# src/birdcode/permission/rules.py
"""L3 规则引擎:YAML 三层(user/project/local)+ session 内存层。

格式:Tool(pattern),如 Bash(git *)。pattern 对 path 工具是路径 glob(对 realpath),
对 Bash 是命令 fnmatch。** 沿用 **→* 归一(fnmatch 不认 **,不引新依赖)。

优先级(高→低):session > local > project > user。跨源:首个有命中的源决定;
源内:deny 压过 allow(安全侧倒)。
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from birdcode.permission.verdict import Decision
from birdcode.utils.logging import get_logger
from birdcode.utils.paths import find_project_root

log = get_logger("birdcode.permission.rules")  # 进 debug.log

# tool 组【分段】放宽:MCP 命名空间 mcp__server__tool 的 server 名可含 -/.(my-server、
# github.com),单列宽松分支;mcp__ 前缀外的普通工具仍 [A-Za-z_*]\w* 严格——否则 typo 规则
# (delete-file、edit.file、Bash.git)会被静默接受为死规则,丢失「格式错误」反馈。
_RULE_RE = re.compile(r"^(?P<tool>mcp__[\w.\-]*|[A-Za-z_*]\w*)\((?P<pat>.*)\)$")

# 已警告过的无效 MCP 规则 (source, 原文):去重。_reload_local 每次 HITL Always 都重读
# 本地 YAML→重 parse,同一手写无效规则会被反复警告(N 次批准=N 条噪音);首次后跳过。
_WARNED_INVALID_MCP: set[tuple[str, str]] = set()


@dataclass
class Rule:
    tool: str  # "bash"/"write_file"/"*";存小写
    pattern: str  # "git *" / "./src/**"
    action: Decision
    source: str = ""  # "session"/文件名

    def matches(self, tool_name: str, subject: str | None) -> bool:
        # 工具名:大小写不敏感;规则里写 "Write" 也命中 "write_file"(蛇形扩展)。
        tn = tool_name.lower()
        if self.tool != "*" and self.tool != tn:
            # 蛇形桥(write→write_file)只对普通工具的单段前缀分组有效。MCP 强制命名空间
            # mcp__server__tool 是不可拆字面量:mcp__srv__foo 不得前缀命中 mcp__srv__foo_bar,
            # 否则 HITL 批准前者会越权 auto-approve 后者(同级写工具绕过 HITL,安全洞)。
            if self.tool.startswith("mcp__") or not tn.startswith(self.tool + "_"):
                return False
        if subject is None:
            return self.pattern in ("", "*")
        pat = self.pattern
        # 命令模式(被检工具是 bash):恒大小写敏感、不做路径归一——命令名各平台区分大小写,
        # 且 bash 的 pattern 即便以 ./ 或 / 开头也是命令(如 Bash(./deploy.sh *)),不是路径。
        if tn == "bash":
            return fnmatch.fnmatchcase(subject, pat)
        # 路径模式:./ 或 / 开头做归一(**→*、\→/、补前缀),沿用平台大小写语义
        # (fnmatch 内置 normcase:NTFS 不敏感/POSIX 敏感,与文件系统一致)。
        if pat.startswith("./"):
            # 相对路径 glob:剥 "./",补 "*/" 让它命中绝对路径任意段
            # (write_file/edit_file 的 subject 是 realpath,带项目绝对前缀)。
            pat = pat.replace("**", "*")
            subject = subject.replace("\\", "/")
            pat = "*/" + pat[2:]
        elif pat.startswith("/"):
            pat = pat.replace("**", "*")
            subject = subject.replace("\\", "/")
        return fnmatch.fnmatch(subject, pat)


def parse_rule(s: str, action: Decision, source: str = "") -> Rule | None:
    m = _RULE_RE.match(s.strip())
    if not m:
        log.warning("规则格式错误,已忽略: %r", s)
        return None
    tool = m["tool"].lower()
    pattern = m["pat"]
    # 有效 MCP 规则须是完整 mcp__server__tool(*) 形式:工具名精确匹配、subject
    # 恒 None 只认 pattern=*。两类失效规则要警告(非静默),否则用户误以为受保护而实无防护:
    # ① server 级/裸前缀(mcp__srv(*)、mcp__(*))——精确匹配下不命中任何工具;
    # ② 路径范围(mcp__srv__tool(/etc/**))——subject None 不匹配路径。
    if tool.startswith("mcp__"):
        rest = tool.removeprefix("mcp__")
        # 须正好 2 个非空 __ 段(server + tool):mcp__srv__tool。mcp__srv(1 段)、
        # mcp____tool(段中有空)、mcp__a__b__c(3 段)都不算完整→死规则。
        segments = [seg for seg in rest.split("__") if seg]
        is_full_name = len(segments) == 2
        if not is_full_name or pattern not in ("", "*"):
            # 去重:_reload_local 每次 HITL Always 都重读本地 YAML→重 parse,同一手写无效
            # 规则会被反复警告(N 次批准=N 条噪音)。按 (source, 原文) 去重,首次警告。
            key = (source, s.strip())
            if key not in _WARNED_INVALID_MCP:
                _WARNED_INVALID_MCP.add(key)
                log.warning(
                    "MCP 规则 %r 已加载但永不命中(死规则):须为完整 mcp__server__tool(*) 形式"
                    "(工具名精确匹配、路径由 server 定不做本地范围);不会生效",
                    s,
                )
    return Rule(tool=tool, pattern=pattern, action=action, source=source)


@dataclass
class RuleSet:
    """sources 已按优先级从高到低排(session 在前)。"""

    sources: list[list[Rule]] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        *,
        session: list[Rule] | None = None,
        yaml_paths: list[Path] | None = None,
    ) -> RuleSet:
        layers: list[list[Rule]] = [session or []]
        for p in yaml_paths or []:
            layers.append(load_yaml(p))
        return cls(sources=layers)

    def match(self, tool_name: str, subject: str | None) -> Decision | None:
        for layer in self.sources:
            allow = deny = False
            for r in layer:
                if r.matches(tool_name, subject):
                    if r.action == "deny":
                        deny = True
                    else:
                        allow = True
            if deny:
                return "deny"
            if allow:
                return "allow"
        return None

    def all_rules(self) -> list[Rule]:
        return [r for layer in self.sources for r in layer]


def default_yaml_paths(root: Path | None = None) -> list[Path]:
    """按优先级从高→低:local > project > user。

    local/project 锚定项目根:root 显式传入(由 gate 用其 project_root 调用,保证沙箱根与
    local-YAML 同源);不传则 find_project_root()(标记:.git→pyproject.toml)。user 锚定 home。
    """
    r = root if root is not None else find_project_root()
    return [
        r / ".birdcode" / "permissions.local.yaml",
        r / ".birdcode" / "permissions.yaml",
        Path.home() / ".birdcode" / "permissions.yaml",
    ]


def load_yaml(path: Path) -> list[Rule]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        log.warning("权限 YAML 解析失败 %s: %s(该层视为空)", path, exc)
        return []
    if not isinstance(data, dict):
        log.warning("权限 YAML 顶层非 mapping,视为空: %s", path)
        return []
    out: list[Rule] = []
    for action in ("allow", "deny"):
        for s in data.get(action) or []:
            r = parse_rule(str(s), action, source=path.name)
            if r is not None:
                out.append(r)
    return out


def append_local(rule_str: str, action: Decision, *, path: Path | None = None) -> Path:
    """HITL「Always」:把规则追加到 local YAML(read-modify-write)。

    path 默认 <project_root>/.birdcode/permissions.local.yaml(find_project_root 兜底,
    不随 cwd 漂移);gate 传入其 _local_path 以保证
    「写入」与「读取」命中同一文件(自定义 yaml_paths 时尤其重要,否则 Always 写错
    位置、_reload 读不到)。
    """
    p = path if path is not None else (find_project_root() / ".birdcode" / "permissions.local.yaml")
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, list[str]] = {"allow": [], "deny": []}
    if p.exists():
        try:
            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                for k in ("allow", "deny"):
                    if isinstance(loaded.get(k), list):
                        data[k] = [str(x) for x in loaded[k]]
        except yaml.YAMLError:
            log.warning("local YAML 损坏,重建: %s", p)
    data.setdefault(action, []).append(rule_str)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p
