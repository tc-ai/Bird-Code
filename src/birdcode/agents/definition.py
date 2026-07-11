"""子 agent 定义:persona + 能力边界 + model 覆盖 + 来源层。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal


class AgentSource(StrEnum):
    BUILTIN = "builtin"
    PROJECT = "project"
    USER = "user"


@dataclass(frozen=True)
class AgentDefinition:
    """一个子 agent 类型(= Agent 工具的 subagent_type)。

    - disallowed_tools:从主 registry 减去(黑名单);agent tool 额外排除(靠 is_agent_tool 标记)。
    - model:profile 名;'' = 继承父 profile。
    - kind/parallel_safe:工具化时暴露给 executor 的权限/并行语义。builtin 显式设
      (explore/plan=read+True、gp=write+False);自定义 .md 可选覆盖,缺省 write/False(保守)。
    - run_in_background:同步/异步派生(控制字段,不进 tool schema、不由模型决定)。默认 False
      (同步);builtin 不设(全同步);自定义 .md frontmatter 可选覆盖。
    - source/path:层覆盖与 md 溯源(build_agent_registry 用)。
    """

    name: str
    description: str
    system_prompt: str
    disallowed_tools: tuple[str, ...] = ()
    model: str = ""
    source: AgentSource = AgentSource.BUILTIN
    path: Path | None = None
    kind: Literal["read", "write"] = "write"
    parallel_safe: bool = False
    run_in_background: bool = False
    # —— NEW(skill 专用,默认值保全现有 agent)——
    mode: Literal["inline", "fork"] = "inline"  # 仅 skill 有意义;agent 恒走 fork 路径,忽略
    is_transparent: bool = False  # 加载器按目录打标(skill/=True);非作者字段
    body: str = ""  # 任务模板(含 $ARGUMENTS);agent 留 ""


def split_frontmatter(raw: str) -> tuple[str, str] | None:
    """切 YAML frontmatter 与正文,返回 (frontmatter_text, body_text) 或 None(无合法 fm)。

    仅在【列 0、独立成行】的 ``---``(YAML 文档标记,允许行尾空白/CR)上切分——首行为开标记,
    其后首个列 0 的 ``---`` 为闭标记。缩进的 ``---``(YAML 块标量内容)与行内的 ``---``
    子串(如 ``description: a---b``)都**不**算分隔符。

    旧实现 ``raw.split("---", 2)`` 按子串切:值或正文里的 ``---`` 会让第二次切分落在
    frontmatter 内部 → 描述被截断、正文被污染,skill 静默加载错内容。agents/loader 与
    skill_loader 共用本函数,避免两份解析逻辑漂移。
    """
    if not raw.startswith("---"):
        return None
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != "---":  # 首行须是列 0 的 ---(去尾空白后)
        return None
    close_idx = None
    for i in range(1, len(lines)):
        if lines[i].rstrip() == "---":  # 列 0 的闭标记(缩进的 --- 是块标量内容,不匹配)
            close_idx = i
            break
    if close_idx is None:
        return None
    return "".join(lines[1:close_idx]), "".join(lines[close_idx + 1 :])


def render_body(defn: AgentDefinition, args: str) -> str:
    """把 defn.body 里的 $ARGUMENTS 占位符替换为 args。

    - body 含 $ARGUMENTS:就地替换(保留作者设计的结构;空 args → 占位符变空串)。
    - body 不含 $ARGUMENTS 且 args 非空(去空白后):**追加**到正文末尾——绝不丢弃用户输入。
      (冒烟回归:无占位符的 skill 如 brainstorming,曾把 `/brainstorming <问题>` 的 <问题> 静默吞掉。)
    - body 不含 $ARGUMENTS 且 args 为空:原样返回。
    """
    if "$ARGUMENTS" in defn.body:
        return defn.body.replace("$ARGUMENTS", args)
    if args.strip():
        return f"{defn.body.rstrip()}\n\n{args}"
    return defn.body
