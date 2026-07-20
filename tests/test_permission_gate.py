# tests/test_permission_gate.py
"""权限门测试:executor↔gate 集成(_StaticGate)+ UiPermissionGate 的 L1-L5 决策流。"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from birdcode.blocks import ToolUseBlock
from birdcode.permission.gate import UiPermissionGate
from birdcode.permission.rules import parse_rule
from birdcode.permission.verdict import Decision, Verdict
from birdcode.tools.base import Tool
from birdcode.tools.executor import ToolExecutor
from birdcode.tools.registry import ToolRegistry

# =====================================================================
# Part 1:executor ↔ gate 集成(用 _StaticGate 固定返回,隔离 gate 内部逻辑)
# =====================================================================


class _WriteInput(BaseModel):
    file_path: str
    content: str


class _PretendWrite(Tool):
    """kind=write 的假工具,execute 会被 permission 拦在前面。"""

    name = "PretendWrite"
    description = "d"
    parameters = _WriteInput
    kind = "write"
    parallel_safe = False

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, *, file_path: str, content: str) -> str:  # type: ignore[override]
        self.executed.append(file_path)
        return "wrote"


class _StaticGate:
    """测试用:固定返回一个 Verdict。实现 PermissionGate 协议(结构化匹配)。"""

    def __init__(self, decision: Decision, reason: str = "") -> None:
        self._d = decision
        self._reason = reason
        self.checked: list[tuple[str, dict]] = []

    async def check(self, tool: Tool, args: dict) -> Verdict:
        self.checked.append((tool.name, args))
        return Verdict(self._d, self._reason, "")


@pytest.mark.asyncio
async def test_write_tool_blocked_by_deny_gate_not_executed():
    tool = _PretendWrite()  # 新实例,executed=[]
    r = ToolRegistry()
    r.register(tool)
    ex = ToolExecutor(r, permission_gate=_StaticGate("deny"))
    res = await ex.execute_batch(
        [ToolUseBlock(id="t1", name="PretendWrite", input={"file_path": "/x", "content": "c"})],
    )
    assert res[0].ok is False
    assert "拒绝" in res[0].error_message
    assert tool.executed == []  # 被 gate 拦住,execute 未调用


@pytest.mark.asyncio
async def test_deny_verdict_reason_echoed_into_tool_result():
    """deny Verdict 的 reason 进 ToolResult.error_message 回传模型。"""
    gate = _StaticGate("deny", reason="L1 黑名单命中。")
    tool = _PretendWrite()
    r = ToolRegistry()
    r.register(tool)
    ex = ToolExecutor(r, permission_gate=gate)
    res = await ex.execute_batch(
        [ToolUseBlock(id="t1", name="PretendWrite", input={"file_path": "/x", "content": "c"})],
    )
    assert res[0].ok is False
    assert "L1 黑名单命中。" in res[0].error_message  # reason 被回传,非默认文案
    assert tool.executed == []


@pytest.mark.asyncio
async def test_deny_verdict_blank_reason_falls_back_to_default_message():
    """reason 为空 → executor 用默认拒绝文案(向后兼容)。"""
    gate = _StaticGate("deny", reason="")
    tool = _PretendWrite()
    r = ToolRegistry()
    r.register(tool)
    ex = ToolExecutor(r, permission_gate=gate)
    res = await ex.execute_batch(
        [ToolUseBlock(id="t1", name="PretendWrite", input={"file_path": "/x", "content": "c"})],
    )
    assert res[0].ok is False
    assert "拒绝" in res[0].error_message


@pytest.mark.asyncio
async def test_write_tool_allowed_by_gate_executed():
    gate = _StaticGate("allow")
    tool = _PretendWrite()
    r = ToolRegistry()
    r.register(tool)
    ex = ToolExecutor(r, permission_gate=gate)
    res = await ex.execute_batch(
        [ToolUseBlock(id="t1", name="PretendWrite", input={"file_path": "/y", "content": "c"})],
    )
    assert res[0].ok and "wrote" in res[0].output
    assert gate.checked == [("PretendWrite", {"file_path": "/y", "content": "c"})]
    assert tool.executed == ["/y"]


@pytest.mark.asyncio
async def test_read_tool_now_goes_through_gate():
    """Step 2:移除 write-only 守卫 → 所有工具(含 read)均经 gate 转发。

    read-default 的「无条件 allow」逻辑在 UiPermissionGate 内(见 Part 2 的
    test_read_skips_mode_and_hitl);executor 只负责无条件转发,不区分 read/write。
    """
    gate = _StaticGate("allow")  # allow 才能让 read 跑到 execute

    class _EmptyInput(BaseModel):
        pass

    class _Read(Tool):
        name = "Peek"
        description = "d"
        parameters = _EmptyInput
        kind = "read"
        parallel_safe = True

        async def execute(self, **args: object) -> str:  # type: ignore[override]
            return "ok"

    r = ToolRegistry()
    r.register(_Read())
    ex = ToolExecutor(r, permission_gate=gate)
    res = await ex.execute_batch([ToolUseBlock(id="r1", name="Peek", input={})])
    assert res[0].ok and res[0].output == "ok"
    assert gate.checked == [("Peek", {})]  # read 也被 check 了


@pytest.mark.asyncio
async def test_no_gate_executes_write_directly():
    """无 gate 注入(默认 None)→ write 直接执行(向后兼容,stage1 行为)。"""
    tool = _PretendWrite()
    r = ToolRegistry()
    r.register(tool)
    ex = ToolExecutor(r)  # permission_gate=None
    res = await ex.execute_batch(
        [ToolUseBlock(id="t1", name="PretendWrite", input={"file_path": "/z", "content": "c"})],
    )
    assert res[0].ok
    assert tool.executed == ["/z"]


# =====================================================================
# Part 2:UiPermissionGate 的 L1-L5 决策流(注入 async stub,无需 app/modal)
# =====================================================================


def _tool(name: str, kind: str = "write") -> Tool:
    """轻量假 Tool:只填 name/kind(够 _extract_subject 与 kind 分支用)。"""

    class _T(Tool):
        async def execute(self, **args: object) -> str:  # type: ignore[override]
            return ""

    t = _T()
    t.name = name
    t.kind = kind  # type: ignore[misc]
    return t


@pytest.fixture
def gate_factory(tmp_path):
    """构造 UiPermissionGate:project_root=tmp_path(沙箱限定其内),yaml 注入 L3 规则。"""

    def _make(
        mode: str = "default", modal: str = "approve", rules_yaml: str = ""
    ) -> UiPermissionGate:
        yf = tmp_path / "p.yaml"
        yf.write_text(rules_yaml, encoding="utf-8")

        async def gm() -> str:
            return mode

        async def rp(tool_name: str, summary: str, path: str | None) -> str:
            return modal

        return UiPermissionGate(
            get_mode=gm,  # type: ignore[arg-type]
            request_permission=rp,  # type: ignore[arg-type]
            yaml_paths=[yf],
            project_root=tmp_path,
        )

    return _make


@pytest.mark.asyncio
async def test_l1_blacklist_even_in_bypass(gate_factory):
    """L1 硬拦:bypass 也不放开危险命令。"""
    g = gate_factory(mode="bypass")
    v = await g.check(_tool("bash"), {"command": "rm -rf /"})
    assert v.action == "deny" and v.layer == "L1"


@pytest.mark.asyncio
async def test_l2_sandbox_blocks_escape(gate_factory, tmp_path):
    """L2 硬拦:写项目根之外的路径被沙箱拒。"""
    g = gate_factory(mode="bypass")
    v = await g.check(_tool("write_file"), {"file_path": str(tmp_path.parent / "x.py")})
    assert v.action == "deny" and v.layer == "L2"


@pytest.mark.asyncio
async def test_l2_sandbox_allows_inside_root(gate_factory, tmp_path):
    """项目根内的路径不触 L2(回归:确保 L2 不会误伤合法路径)。"""
    g = gate_factory(mode="bypass")
    v = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "f")})
    assert v.action == "allow"  # bypass 放行,且未被 L2 拦


@pytest.mark.asyncio
async def test_l3_allow_rule_beats_plan(gate_factory):
    """L3 优先于 L4:allow 规则即使在 plan 模式下也放行该具体项。"""
    g = gate_factory(mode="plan", rules_yaml="allow:\n  - Bash(git *)")
    v = await g.check(_tool("bash"), {"command": "git status"})
    assert v.action == "allow" and v.layer == "L3"


@pytest.mark.asyncio
async def test_l3_deny_rule(gate_factory, tmp_path):
    """L3 deny 规则:Write ./secrets/** 被拒。"""
    g = gate_factory(rules_yaml="deny:\n  - Write(./secrets/**)")
    v = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "secrets" / "k")})
    assert v.action == "deny" and v.layer == "L3"


@pytest.mark.asyncio
async def test_l4_plan_denies_write(gate_factory, tmp_path):
    """plan 模式 → 写工具被拒(L4)。"""
    g = gate_factory(mode="plan")
    v = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "f")})
    assert v.action == "deny" and v.layer == "L4"


@pytest.mark.asyncio
async def test_l4_bypass_allows_write(gate_factory, tmp_path):
    """bypass 模式 → 写工具放行(L4,绕过 HITL)。"""
    g = gate_factory(mode="bypass")
    v = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "f")})
    assert v.action == "allow"


@pytest.mark.asyncio
async def test_l4_accept_edits_auto_allows_write(gate_factory, tmp_path):
    """accept-edits + write_file/edit_file → 自动放行(spec §4.1)。"""
    g = gate_factory(mode="accept-edits")
    v = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "f")})
    assert v.action == "allow" and v.layer == "L4"


@pytest.mark.asyncio
async def test_read_skips_mode_and_hitl(gate_factory, tmp_path):
    """读类:走 L2/L3 后无条件 allow,不触 L4(plan 也不拦)/L5。"""
    g = gate_factory(mode="plan")  # plan 不拦读
    v = await g.check(_tool("read_file", kind="read"), {"file_path": str(tmp_path / "f")})
    assert v.action == "allow"
    assert v.layer == "read-default"


@pytest.mark.asyncio
async def test_l5_approve(gate_factory, tmp_path):
    """default 模式 + write → HITL approve → allow。"""
    g = gate_factory(mode="default", modal="approve")
    v = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "f")})
    assert v.action == "allow" and v.layer == "L5"


@pytest.mark.asyncio
async def test_l5_reject(gate_factory, tmp_path):
    """HITL reject → deny(Esc=Reject)。"""
    g = gate_factory(mode="default", modal="reject")
    v = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "f")})
    assert v.action == "deny"


@pytest.mark.asyncio
async def test_l5_session_persists_in_memory(gate_factory, tmp_path):
    """session 按类别(Write(*))宽放:首次 write_file HITL 放行并加 Write/Edit/Delete(*),
    之后【不同路径】的 write_file/edit_file 都命中 L3(本会话全放行该类别,不再弹)。"""
    g = gate_factory(mode="default", modal="session")
    v1 = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "f")})
    assert v1.action == "allow" and v1.layer == "L5-session"
    v2 = await g.check(_tool("write_file"), {"file_path": str(tmp_path / "other")})
    assert v2.layer == "L3"  # Write(*) 宽放:不同路径也命中
    v3 = await g.check(_tool("edit_file"), {"file_path": str(tmp_path / "yet")})
    assert v3.layer == "L3"  # Edit(*) 同类别宽放


@pytest.mark.asyncio
async def test_l5_session_for_bash_uses_category_pattern(gate_factory):
    """Bash session 按类别(Bash(*))宽放:首次 git status 放行加 Bash(*),之后【任意】
    bash 命令(含非 git)都命中 L3(对齐 Claude Code "all commands during this session")。"""
    g = gate_factory(mode="default", modal="session")
    v = await g.check(_tool("bash"), {"command": "git status"})
    assert v.action == "allow" and v.layer == "L5-session"
    v2 = await g.check(_tool("bash"), {"command": "echo hi"})  # 非 git 也命中 Bash(*)
    assert v2.layer == "L3"


@pytest.mark.asyncio
async def test_read_outside_sandbox_allowed_by_rule(gate_factory, tmp_path):
    """读类:显式 allow 规则可放行沙箱外读取(L3 先于 L2;读危险度低且 bash cat 已绕过)。"""
    g = gate_factory(mode="default", rules_yaml="allow:\n  - Read(*)")
    outside = tmp_path.parent / "external.txt"
    v = await g.check(_tool("read_file", kind="read"), {"file_path": str(outside)})
    assert v.action == "allow" and v.layer == "L3-read"


@pytest.mark.asyncio
async def test_read_outside_sandbox_denied_without_rule(gate_factory, tmp_path):
    """读类:沙箱外读取且无 allow 规则 → 仍被 L2 拦(C 只对显式 allow 放行,不削弱 L2)。"""
    g = gate_factory(mode="default")  # 无规则
    outside = tmp_path.parent / "external.txt"
    v = await g.check(_tool("read_file", kind="read"), {"file_path": str(outside)})
    assert v.action == "deny" and v.layer == "L2"


@pytest.mark.asyncio
async def test_concurrent_hitl_serialized(tmp_path):
    """并发 L5 HITL 经 _hitl_lock 串行:同时两个写 check → request_permission 不并发(防弹窗竞态)。

    并行 sync 子 agent 共用 parent_gate,若都触发写 HITL,锁保证一次只弹一个 modal。
    """
    import asyncio

    live = 0
    max_live = 0

    async def rp(_name: str, _summary: str, _path: str | None) -> str:
        nonlocal live, max_live
        live += 1
        if live > max_live:
            max_live = live
        await asyncio.sleep(0.02)  # 拉长窗口,确保两调用时间重叠(无锁时 max_live 会到 2)
        live -= 1
        return "approve"

    async def gm() -> str:
        return "default"

    g = UiPermissionGate(
        get_mode=gm,  # type: ignore[arg-type]
        request_permission=rp,  # type: ignore[arg-type]
        yaml_paths=[],
        project_root=tmp_path,
    )
    await asyncio.gather(
        g.check(_tool("write_file"), {"file_path": str(tmp_path / "a")}),
        g.check(_tool("write_file"), {"file_path": str(tmp_path / "b")}),
    )
    assert max_live == 1  # 锁串行:同一时刻只有 1 个 request_permission 在跑


# =====================================================================
# Part 3:MCP 工具的 _rule_str_for —— 命名空间作用域(Task 13)
#   mcp__server__tool 不能走 _tool_display(会塌成 Mcp(...),经蛇形桥误匹配全部 mcp__*)。
# =====================================================================


def test_rule_str_for_mcp_emits_full_namespace(gate_factory):
    """MCP 工具 → 原样 mcp__server__tool(*),不塌成 Mcp(None)。"""
    g = gate_factory()
    s = g._rule_str_for(_tool("mcp__grafana__query_prometheus"), None)
    assert s == "mcp__grafana__query_prometheus(*)"


def test_mcp_rule_round_trips_through_matcher(gate_factory):
    """L5 Always/Session 写出的 mcp 规则串 → parse → matches 精确命中、不误伤其它。

    关键回归:mcp__grafana__query_prometheus(*) 绝不能匹配 mcp__grafana__other_tool
    或 mcp__slack__x(若塌成 Mcp(...) 则经蛇形桥命中全部 mcp__*)。
    """
    g = gate_factory()
    s = g._rule_str_for(_tool("mcp__grafana__query_prometheus"), None)
    rule = parse_rule(s, "allow")
    assert rule is not None
    assert rule.matches("mcp__grafana__query_prometheus", None) is True
    assert rule.matches("mcp__grafana__other_tool", None) is False
    assert rule.matches("mcp__slack__x", None) is False


@pytest.mark.asyncio
async def test_mcp_tool_with_path_arg_skips_l2(gate_factory):
    """回归(#3):MCP 工具带 path/file_path 参数不进本地路径沙箱 L2——其路径语义由 server
    自定(远端/虚拟),针对本地根的 L2 会硬拒(用户无法批准)。MCP 写工具改由 L5 HITL 把关。"""
    g = gate_factory(mode="default", modal="approve")
    v = await g.check(_tool("mcp__fs__read_file"), {"path": "/v1/users"})  # 远端式路径
    assert v.action == "allow" and v.layer == "L5"  # 未被 L2 拦,直达 HITL approve


# =====================================================================
# Part 4:fork_async —— 异步子 agent gate(I4:L1-L4 同父,L5 HITL 自动拒,不弹 modal)
# =====================================================================


@pytest.mark.asyncio
async def test_fork_async_rejects_write_in_default_mode_without_modal(tmp_path):
    """fork_async(I4):default 模式 write 工具 → L5 自动拒(不弹 modal)。

    父 gate 的 request_permission 回调被 fork 换成常返 reject 的 coroutine,
    故父回调零调用(异步子 agent 不交互)。写项目根内路径确保走到 L5(非 L2 沙箱拦)。
    """
    calls = {"n": 0}

    async def gm() -> str:
        return "default"

    async def rp(tool_name: str, summary: str, path: str | None) -> str:
        calls["n"] += 1
        return "approve"  # 父会批准;异步 fork 不应调到这

    yf = tmp_path / "p.yaml"
    yf.write_text("", encoding="utf-8")
    parent = UiPermissionGate(
        get_mode=gm,  # type: ignore[arg-type]
        request_permission=rp,  # type: ignore[arg-type]
        yaml_paths=[yf],
        project_root=tmp_path,
    )
    child = parent.fork_async()
    v = await child.check(_tool("write_file"), {"file_path": str(tmp_path / "x.txt")})
    assert v.action == "deny" and v.layer == "L5"  # 走到 L5 且异步拒(非 L2 沙箱)
    assert calls["n"] == 0  # 未弹 modal:父回调零调用


@pytest.mark.asyncio
async def test_resume_session_rule_preauth_skips_hitl(gate_factory):
    """ResumePrompt Yes 预授权:add_session("Resume(*)") → resume_agent 走 L3 放行,不再弹 L5。

    消除"弹两次":用户在 ResumePrompt 已明示续跑,resume_agent 不该紧接着又过权限门 HITL
    (其 docstring 亦写「本工具假定已授权」)。规则串与 gate 自身 _add_session_by_category
    对 resume_agent 产出的 "Resume(*)" 一致(蛇形桥 resume→resume_agent)。
    """
    resume = _tool("resume_agent", "write")
    # 无预授权 + HITL=reject → 走到 L5 → deny(证明原本会弹门)
    g_no = gate_factory(mode="default", modal="reject")
    v_no = await g_no.check(resume, {"agent_id": "sub-1", "direction": "继续"})
    assert v_no.action == "deny" and v_no.layer == "L5"
    # 预授权 → L3 放行,不弹门(即便 HITL 设成 reject)
    g_yes = gate_factory(mode="default", modal="reject")
    g_yes.add_session("Resume(*)", "allow")
    v_yes = await g_yes.check(resume, {"agent_id": "sub-1", "direction": "继续"})
    assert v_yes.action == "allow" and v_yes.layer == "L3"


@pytest.mark.asyncio
async def test_fork_async_allows_read_same_as_parent(gate_factory, tmp_path):
    """fork_async:L1-L4 与父完全一致 → 读类照常 allow(读不触 L5,
    即便异步 L5 会拒)。project_root 内读取走 read-default 放行。"""
    parent = gate_factory(mode="default", modal="reject")  # L5 拒;但读不触 L5
    child = parent.fork_async()
    f = tmp_path / "r.txt"
    f.write_text("x", encoding="utf-8")
    v = await child.check(_tool("read_file", kind="read"), {"file_path": str(f)})
    assert v.action == "allow" and v.layer == "read-default"


@pytest.mark.asyncio
async def test_fork_async_allows_bash(tmp_path):
    """fork_async:bash 放行(异步子 agent 常用只读 bash 探查;L1 仍拦危险,其余 approve)。"""

    async def gm() -> str:
        return "default"

    async def rp(_tool_name: str, _summary: str, _path: str | None) -> str:
        return "approve"  # 父回调不应被调(bash 走 fork 的 _allow_bash)

    yf = tmp_path / "p.yaml"
    yf.write_text("", encoding="utf-8")
    parent = UiPermissionGate(
        get_mode=gm,  # type: ignore[arg-type]
        request_permission=rp,  # type: ignore[arg-type]
        yaml_paths=[yf],
        project_root=tmp_path,
    )
    child = parent.fork_async()
    v = await child.check(_tool("bash"), {"command": "ls -la"})
    assert v.action == "allow" and v.layer == "L5"  # fork 的 _allow_bash 对 bash approve


# =====================================================================
# Part 5:D6 硬隔离(sandbox_root)+ D7 _local_path 锚定 project_root
#   worktree 会话锁 worktree(主仓路径 L2 直接拒,不进 L4 accept-edits/bypass);
#   _local_path 兜底锚 project_root 而非 Path.cwd()。
# =====================================================================


async def test_gate_local_path_anchors_to_project_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """yaml_paths=[] 兜底:_local_path 应锚定 project_root,而非 Path.cwd()(D7)。"""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # cwd ≠ project_root

    async def _mode() -> str:
        return "default"

    async def _perm(_t: str, _s: str, _p: str | None) -> str:
        return "reject"

    gate = UiPermissionGate(
        get_mode=_mode, request_permission=_perm, yaml_paths=[], project_root=tmp_path
    )
    assert gate._local_path.parent.parent == tmp_path.resolve()
    assert gate._local_path.parent.parent != elsewhere.resolve()


async def test_gate_sandbox_root_locks_to_worktree(tmp_path: Path) -> None:
    """sandbox_root=worktree → L2 锁 worktree:主仓路径越界拒,worktree 内放行(D6 硬隔离)。"""
    from birdcode.permission.sandbox import check_path

    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()

    async def _mode() -> str:
        return "default"

    async def _perm(_t: str, _s: str, _p: str | None) -> str:
        return "reject"

    gate = UiPermissionGate(
        get_mode=_mode, request_permission=_perm, project_root=main, sandbox_root=wt
    )
    assert check_path(str(wt / "src" / "f.py"), gate._roots) is None  # worktree 内放行
    assert check_path(str(main / "src" / "f.py"), gate._roots) is not None  # 主仓越界拒


async def test_gate_project_root_main_when_cfg_none_and_worktree(tmp_path: Path) -> None:
    """_cfg=None + worktree 时,on_mount 传 project_root=resolve_main_repo(cwd)=主仓。

    回归 T4-M2:旧代码 _cfg=None → project_root=None → gate 的 find_project_root()
    兜底返 worktree(.git 文件存在),致 yaml_paths 锚 worktree/.birdcode(空,无规则)。
    修复后传 resolve_main_repo(_cwd)=主仓。本测试验证 gate 契约:显式传 project_root=main
    → _project_root=main(非 worktree),即 on_mount 修复所依赖的 gate 行为成立。
    """

    async def _mode() -> str:
        return "default"

    async def _perm(_t: str, _s: str, _p: str | None) -> str:
        return "reject"

    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    gate = UiPermissionGate(
        get_mode=_mode,
        request_permission=_perm,
        yaml_paths=[],
        project_root=main,
        sandbox_root=wt,
    )
    assert gate._project_root == main.resolve()  # 锚主仓,非 worktree
    assert gate._project_root != wt.resolve()
