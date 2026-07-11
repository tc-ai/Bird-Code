# src/birdcode/permission/gate.py
"""PermissionGate 协议:executor 与 UI 之间的唯一权限接缝。

executor 不碰 ModalScreen/controller——只 await 一个返回 Verdict 的 awaitable。
UiPermissionGate 串联 L1-L5 五层防御:黑名单(硬)→沙箱(硬)→规则→模式→HITL。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from birdcode.permission.blacklist import dangerous_hint, is_dangerous
from birdcode.permission.rules import (
    Rule,
    RuleSet,
    append_local,
    default_yaml_paths,
    load_yaml,
    parse_rule,
)
from birdcode.permission.sandbox import check_path, resolve_roots
from birdcode.permission.verdict import Decision, Verdict
from birdcode.tools.base import Tool
from birdcode.utils.logging import get_logger
from birdcode.utils.paths import find_project_root

log = get_logger("birdcode.permission.gate")  # 进 debug.log


@runtime_checkable
class PermissionGate(Protocol):
    """工具执行前的权限检查。返回 Verdict(action=allow/deny + reason + layer)。"""

    async def check(self, tool: Tool, args: dict[str, object]) -> Verdict: ...


Mode = Literal["plan", "default", "accept-edits", "bypass"]
ModalResult = Literal["approve", "reject", "session"]

# Write/Edit 在 accept-edits 下自动接受;其它 write 工具(含 stage5 Bash)仍需确认。
_AUTO_ACCEPT_IN_ACCEPT_EDITS = {"write_file", "edit_file"}


def _extract_subject(tool: Tool, args: dict[str, object]) -> str | None:
    """Bash→command;path 工具→file_path/path;MCP 工具→None。

    MCP 工具的「path」语义由 server 自定(多为远端/自沙箱的虚拟路径),与本地项目根无关;
    让它进 L2 沙箱会针对本地根解析→「路径越界」硬拒(L2 在 HITL 前,用户无法批准)→
    常见 MCP 文件系统/git/API server 被永久挡死。MCP 写工具改由 L3 命名空间规则 + L5 HITL 把关。
    """
    if tool.name == "bash":
        c = args.get("command")
        return c if isinstance(c, str) else None
    if tool.name.startswith("mcp__"):
        return None
    p = args.get("file_path") or args.get("path")
    return str(p) if isinstance(p, str) else None


def _generalize_bash(command: str) -> str:
    """Bash「Always」泛化:首 token + *。git status→Bash(git *)。偏宽,用户可手改。"""
    tokens = command.strip().split()
    head = tokens[0] if tokens else (command.strip() or command)
    return f"Bash({head} *)"


def _tool_display(tool_name: str) -> str:
    """write_file→Write, edit_file→Edit, delete_file→Delete, read_file→Read, bash→Bash。

    取首段 capitalize:规则串经 parse_rule 小写化后,首段(snake 前缀)能命中
    Rule.matches 的蛇形桥(规则 "write" 命中工具 "write_file"),保证 round-trip。
    全段拼接(WriteFile)会破坏该桥,导致 always/session 规则二次不命中。
    """
    return tool_name.split("_")[0].capitalize()


class UiPermissionGate:
    """L1 黑名单(硬)→ L2 沙箱(硬,仅 path 类工具)→ L3 规则 → L4 模式 → L5 HITL。

    - L2 仅对带 file_path/path 的工具(read/write/edit/delete/glob/grep)生效;Bash 命令
      无法可靠抽出路径,L2 对其不适用——Bash 的硬底由 L1 黑名单 + HITL 兜底。
    - 读类:L3-allow 可先于 L2 放行沙箱外读取(读危险度低且 bash cat 已绕过 L2);
      无命中则 L2 仍拦越界读;之后无条件 allow,不触 L4/L5。
    - 规则(L3)优先于模式(L4):allow 规则即使在 plan 下也放行该具体项。
    - Bash 的「Always」退化为本会话(不持久化),见 check() 内说明。
    """

    def __init__(
        self,
        *,
        get_mode: Callable[[], Awaitable[Mode]],
        request_permission: Callable[[str, str, str | None], Awaitable[ModalResult]],
        yaml_paths: list[Path] | None = None,
        extra_roots: list[Path] | None = None,
        project_root: Path | None = None,
        sandbox_root: Path | None = None,
    ) -> None:
        self._get_mode = get_mode
        self._request_permission = request_permission
        # project_root 先定:yaml_paths 默认锚定它(同源),避免沙箱根与 local-YAML 锚点分歧。
        self._project_root = (project_root or find_project_root()).resolve()
        # 硬隔离:sandbox_root 仅用于 L2 沙箱根(worktree 会话锁 worktree,主仓路径 L2 直接拒,
        # 不进 L4 accept-edits/bypass)。None → L2 用 project_root(旧行为,向后兼容)。
        # 与 project_root 解耦:yaml_paths/permissions 仍锚 project_root,不受沙箱根影响。
        self._sandbox_root = sandbox_root.resolve() if sandbox_root is not None else None
        if yaml_paths is not None:
            self._yaml_paths = list(yaml_paths)
        else:
            self._yaml_paths = default_yaml_paths(self._project_root)
        # local 层 = yaml_paths[0](default_yaml_paths 约定 [local, project, user])。
        # HITL「Always」写这里,且 _reload_local 只重读它(省去全量 _reload 的系统调用)。
        # 兜底锚 project_root(非 Path.cwd()),cwd 漂移(worktree chdir)不影响 local-YAML 定位。
        self._local_path = self._yaml_paths[0] if self._yaml_paths else (
            self._project_root / ".birdcode" / "permissions.local.yaml"
        )
        self._extra_roots = [p.resolve() for p in (extra_roots or [])]
        self._session: list[Rule] = []
        # 串行化并发 L5 HITL:并行 sync 子 agent 共用本 gate(见 runner.build_child_gate
        # allow_hitl=True 复用父 gate),并发写 HITL 靠它排队,一次只弹一个 modal(避免弹窗竞态)。
        self._hitl_lock = asyncio.Lock()
        self._reload()

    def fork_async(self) -> UiPermissionGate:
        """异步子 agent gate:L1-L4 与父一致(同 mode/rules/sandbox),L5 对 bash 放行、
        其余写工具自动拒。

        bash 放行:异步子 agent(如只读 code-reviewer)常用 bash 做只读探查(ls/git log/cat/
        git diff),逐次 HITL 不可行(异步不交互、无法弹菜单);L1 黑名单仍拦危险命令(rm -rf /
        等),故对 bash 返回 approve、其余写工具(write/edit/delete)仍 reject——异步子 agent
        不应后台改文件。L1-L4 复用父 get_mode/yaml_paths/extra_roots/project_root/sandbox_root。
        """
        async def _allow_bash(tool_name: str, _summary: str, _path: str | None) -> ModalResult:
            return "approve" if tool_name == "bash" else "reject"

        return UiPermissionGate(
            get_mode=self._get_mode,
            request_permission=_allow_bash,
            yaml_paths=list(self._yaml_paths),
            extra_roots=list(self._extra_roots),
            project_root=self._project_root,
            sandbox_root=self._sandbox_root,
        )

    def fork_worktree_async(self, *, sandbox_root: Path) -> UiPermissionGate:
        """worktree 异步子 agent gate。L1-L4 复用父,L5 恒 approve(无 HITL),
        sandbox_root=worktree(L2 锁 worktree,主仓路径 L2 拒)。

        不同于 fork_async(无 worktree 的异步子 agent:L5 对写工具 reject,怕毁主仓):worktree
        子 agent 的写被 L2 限定在 worktree 内(主仓路径 L2 硬拦),故写工具可自动批——L2 兜底
        取代 fork_async 的拒写。request_permission 恒 approve(bypass L5,保留 L1/L2/L3)。
        bash 同样自动批(L1 黑名单仍拦 rm -rf / 等)。复用父 get_mode/yaml_paths/extra_roots/
        project_root(skill/permission 仍锚主仓)。
        """
        async def _approve(_tool_name: str, _summary: str, _path: str | None) -> ModalResult:
            return "approve"

        return UiPermissionGate(
            get_mode=self._get_mode,
            request_permission=_approve,
            yaml_paths=list(self._yaml_paths),
            extra_roots=list(self._extra_roots),
            project_root=self._project_root,
            sandbox_root=sandbox_root,  # OVERRIDE:父 sandbox(主仓/父 worktree)→ 子 worktree
        )

    # ---- 规则管理 ----
    # L3 判定算法(源内 deny-wins / 跨源首命中)单一实现在 RuleSet(rules.py:RuleSet.match);
    # gate 只持有一个 RuleSet 并委派 _match/list_rules,不再自维护算法,杜绝双实现漂移。
    def _reload(self) -> None:
        sandbox_base = self._sandbox_root or self._project_root
        self._roots = resolve_roots([sandbox_base, *self._extra_roots])
        # sources[0] = self._session(同引用):add_session 追加后当次 match 立即可见。
        self._ruleset: RuleSet = RuleSet(
            sources=[self._session, *[load_yaml(p) for p in self._yaml_paths]]
        )

    def _reload_local(self) -> None:
        """仅重读 local 层(sources[1] = yaml_paths[0]);roots 与其它 YAML 未变。

        HITL「Always」只动了 local 文件,无需 resolve_roots(系统调用)与重读 project/user。
        """
        if len(self._ruleset.sources) > 1:
            self._ruleset.sources[1] = load_yaml(self._local_path)

    def reload(self) -> None:
        self._reload()

    def list_rules(self) -> list[Rule]:
        return self._ruleset.all_rules()

    def add_session(self, rule_str: str, action: Decision) -> None:
        r = parse_rule(rule_str, action, source="session")
        if r is not None:
            self._session.append(r)  # sources[0] 同引用,追加即对 ruleset 可见

    @property
    def extra_roots(self) -> list[Path]:
        """沙箱额外根的只读副本(供 /clear 重接测试与内省)。"""
        return list(self._extra_roots)

    def replace_extra_root(self, old: Path, new: Path) -> None:
        """原子地把 _extra_roots 中的 old 换成 new(/clear 切到新 session 时调)。

        add_extra_root 只追加不删 → N 次 /clear 累积 N 个旧 session 的 tool-results 根
        (O(N²) 且跨会话残留,LLM 仍可 read_file 旧会话落盘)。replace 移旧加新,根数恒定。
        先算新列表再提交,resolve_roots 抛异常时状态保持一致。
        """
        old_r = old.resolve()
        new_r = new.resolve()
        new_extra = [p for p in self._extra_roots if p != old_r]
        if new_r not in new_extra:
            new_extra.append(new_r)
        self._roots = resolve_roots([self._sandbox_root or self._project_root, *new_extra])
        self._extra_roots = new_extra

    def _match(self, tool_name: str, subject: str | None) -> Decision | None:
        """委派 RuleSet.match(L3 判定的单一实现,见 rules.py:RuleSet.match)。"""
        return self._ruleset.match(tool_name, subject)

    # ---- 主决策 ----
    async def check(self, tool: Tool, args: dict[str, object]) -> Verdict:
        subject = _extract_subject(tool, args)

        # L1 黑名单(仅 Bash;bypass 也拦)
        if tool.name == "bash" and isinstance(subject, str):
            hit, label = is_dangerous(subject)
            if hit:
                return Verdict("deny", dangerous_hint(label), "L1")

        path_str = subject if tool.name != "bash" and isinstance(subject, str) else None
        is_read = tool.kind == "read"

        # 读类:L3 规则先于 L2——显式 allow 规则可放行沙箱外的读取。读危险度低于写,
        # 且 bash(cat)本就绕过 L2;让 YAML allow 规则能细粒度放行项目外读路径(如 vendor
        # 软链、外部参考目录),否则 L2 硬拦既过严又无效。写类仍 L2 硬先于 L3(不可被规则放开)。
        # 结果缓存到 read_decision 供下面复用(_match 是纯函数,无谓二次调用)。
        read_decision: Decision | None = None
        if is_read and path_str is not None:
            read_decision = self._match(tool.name, subject)
            if read_decision == "allow":
                return Verdict("allow", "", "L3-read")
            if read_decision == "deny":
                return Verdict("deny", "规则拒绝。", "L3")

        # L2 沙箱(path 工具,硬;bypass 也拦)。Bash 无 file_path,跳过。
        # 相对路径必须相对【与文件工具 _cwd 相同的基】解析,否则 check_path 内 Path(rel).resolve()
        # 相对进程 cwd 解析会与工具口径分歧:
        # - worktree 子 agent:工具 _cwd=worktree,但进程 cwd 仍为主仓(Phase 2 不 os.chdir)
        #   → 相对 _sandbox_root(=worktree)解析,否则越界误拒。
        # - 主 agent:工具 _cwd=os.getcwd()(Task 2 __init__ 默认)→ 相对 Path.cwd() 解析,
        #   与旧 Path(rel).resolve() 等价(向后兼容)。用 Path.cwd() 而非 _project_root:子目录
        #   启动时 project_root(仓库根)≠ cwd,会与工具 _cwd 分歧、误拒 .. 相对路径。
        if path_str is not None:
            check_subject = path_str
            if not Path(path_str).is_absolute():
                base = self._sandbox_root or Path.cwd()
                check_subject = str(base / path_str)
            err = check_path(check_subject, self._roots)
            if err is not None:
                return Verdict("deny", err, "L2")

        # glob 的 pattern(可含 ..)不在 _extract_subject 抽取的 file_path/path 之列 →
        # 上面 L2 只校验 path 自身(path="."/"sub" 解析进 worktree 内即放行),从不审 pattern →
        # worktree 子 agent glob(path=".", pattern='../../../**') 仍可枚举主仓(读隔离泄漏)。
        # 故对 glob 单独补审 pattern 的 .. 逃逸(基 = args[path] 或 sandbox);不门控 path_str,
        # 否则传任意沙箱内 path 即绕过(复审补丁)。仅 worktree 沙箱(_sandbox_root 非空)
        # 触发;主 agent(sandbox=None)读类宽松(read-default 允许越界读)零回归。grep 的 pattern
        # 是正则(.. 非目录跳级),不在此列;grep 越界靠 path(已由上面 L2 覆盖)。
        if self._sandbox_root is not None and tool.name == "glob":
            pat = args.get("pattern")
            if isinstance(pat, str) and ".." in pat.replace("\\", "/").split("/"):
                gbase: Path = self._sandbox_root
                gpath = args.get("path")
                if isinstance(gpath, str):
                    gp = Path(gpath)
                    gbase = gp if gp.is_absolute() else self._sandbox_root / gp
                # glob 特殊字符(*?[)作字面路径段,不影响 .. 的 resolve
                err = check_path(str((gbase / pat).resolve()), self._roots)
                if err is not None:
                    return Verdict("deny", err, "L2")

        # L3 规则(优先于模式)。读类复用 read_decision(已查);写类现查。
        decision = read_decision if is_read else self._match(tool.name, subject)
        if decision is not None:
            reason = "" if decision == "allow" else "规则拒绝。"
            return Verdict(decision, reason, "L3")

        # 读类:到此默认 allow,不触模式/HITL
        if is_read:
            return Verdict("allow", "", "read-default")

        # L4 模式(写工具)
        mode = await self._get_mode()
        if mode == "plan":
            return Verdict("deny", "plan 模式拒绝写入。", "L4")
        if mode == "bypass":
            return Verdict("allow", "", "L4")
        if mode == "accept-edits" and tool.name in _AUTO_ACCEPT_IN_ACCEPT_EDITS:
            return Verdict("allow", "", "L4")
        # default / accept-edits 的非 edit 工具 → L5

        # L5 HITL(锁:并发写 HITL 串行化,一次只显一个 prompt;主 agent 单流常态非竞争)
        summary = _summarize(tool.name, args)
        async with self._hitl_lock:
            result = await self._request_permission(tool.name, summary, path_str)
        if result == "approve":
            return Verdict("allow", "", "L5")
        if result == "session":
            # 按类别宽放(对齐 Claude Code):edits→Write/Edit/Delete、commands→Bash 等,
            # 一次确认本会话同类全放行;比旧精确规则 Write(/path)/Bash(git *) 更宽。
            self._add_session_by_category(tool)
            return Verdict("allow", "", "L5-session")
        return Verdict("deny", "用户拒绝(Esc/Reject)。", "L5")

    def _rule_str_for(self, tool: Tool, subject: str | None) -> str:
        """HITL Always/Session 的规则串。
        Bash→泛化首 token;MCP 工具→原样 mcp__server__tool(*)(绕过 _tool_display,
        否则 mcp__a__b 塌成 Mcp(...) 误匹配全部 mcp__*);path 工具→精确路径。"""
        if tool.name == "bash":
            return _generalize_bash(subject or "")
        if tool.name.startswith("mcp__"):
            return f"{tool.name}(*)"
        return f"{_tool_display(tool.name)}({subject})"

    def _add_session_for(self, tool: Tool, subject: str | None) -> None:
        self.add_session(self._rule_str_for(tool, subject), "allow")

    def _add_session_by_category(self, tool: Tool) -> None:
        """HITL session 按类别宽放(对齐 Claude Code,取代旧精确 _add_session_for)。

        edits→Write+Edit+Delete、commands→Bash、mcp→mcp__srv__tool、其他→{Tool},
        均 (*) 通配本会话全放行该类别。
        """
        if tool.name in ("write_file", "edit_file", "delete_file"):
            for r in ("Write(*)", "Edit(*)", "Delete(*)"):
                self.add_session(r, "allow")
        elif tool.name == "bash":
            self.add_session("Bash(*)", "allow")
        elif tool.name.startswith("mcp__"):
            self.add_session(f"{tool.name}(*)", "allow")
        else:
            self.add_session(f"{_tool_display(tool.name)}(*)", "allow")

    def _add_persistent_for(self, tool: Tool, subject: str | None) -> None:
        append_local(self._rule_str_for(tool, subject), "allow", path=self._local_path)
        self._reload_local()  # 只重读 local 层(roots/其它 YAML 未变),使本会话后续命中


def _summarize(tool_name: str, args: dict[str, object]) -> str:
    """构造 ModalScreen 摘要,如 Write /path (N 行)。"""
    path = args.get("file_path") or args.get("path") or args.get("command")
    content = args.get("content")
    if isinstance(content, str) and content:
        n = content.count("\n") + 1
        return f"{tool_name} {path} ({n} 行)"
    if isinstance(path, str):
        return f"{tool_name} {path}"
    return f"{tool_name}"
