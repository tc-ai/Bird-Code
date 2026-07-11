# src/birdcode/tools/base.py
"""Tool 基类与内部结果结构。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


@dataclass
class ToolResult:
    """工具执行的内部结果（区别于给模型的 ToolResultBlock）。

    三轨(executor 双轨外存后):
      output       → 给 UI 折叠区(head/tail 截到 2000)
      llm_content  → 给 LLM(超阈=占位 / 否则=原文或截断版)
      full_output  → 完整原文(executor 内部;超阈时供 sink 落盘,不进 jsonl)
      persisted_*  → 落盘元信息(超阈时填)

    agent_loop 据此构造 ToolResultBlock(content 用 llm_content)进 history,
    并 emit ProviderEvent.ToolResult(output 仍用 output)给 UI。
    """

    tool_use_id: str
    ok: bool
    output: str = ""
    error_message: str = ""
    truncated: bool = False
    diff: tuple[str, str] | None = None  # stage4:(old, new) 文本,Edit/Write 成功时填
    llm_content: str = ""
    full_output: str = ""
    persisted_path: str | None = None
    persisted_size: int | None = None
    # 顶层 toolUseResult 结构化本地真相:全工具地基,Agent 首个实例;
    # 其他工具按需增量填充(/undo 将填 Edit.structuredPatch)。None=未填充。
    # **不进 message.content / 不发 LLM**;供 UI 回放/搜索/undo。
    tool_use_result: dict[str, Any] | None = None


@dataclass
class ToolOutput:
    """工具的结构化返回(扩展 execute 的 str):text 进三轨,tool_use_result 进结构化真相层。

    Agent 工具首个实例。现有工具继续返回 str(向后兼容,str 仍是合法返回)。
    persisted_* 暂不引入(YAGNI,Agent 工具不用;未来大输出外存再加)。
    """

    text: str
    tool_use_result: dict[str, Any] | None = None


class Tool(ABC):
    """所有工具的基类。parameters 是 Pydantic 模型，即 JSON Schema 的来源。"""

    name: str = ""
    description: str = ""
    parameters: type[BaseModel] = BaseModel
    input_schema: dict[str, Any] | None = None  # MCP 原始 schema;None 走 model_json_schema()
    kind: Literal["read", "write"] = "write"
    parallel_safe: bool = False
    hidden: bool = False  # True 时不暴露给 LLM(to_*_tools 跳过),仅内部/测试可执行
    # 持有【父会话级】状态的工具(如 read_history 的主线 jsonl、tool_search 的父 registry)。
    # True → build_child_registry 排除(不原样共享给子 agent,否则子 agent 读/写父 agent 状态)。
    # 与 fork_for_child 的区别:session_scoped 工具根本不该进子工具表;fork_for_child 处理
    # 「可继承但需隔离每实例状态」的工具(如 BashTool._cwd)。
    session_scoped: bool = False
    # 工具输出持久化阈值(字符数)。超此 → executor 落盘完整结果 + 给 LLM 占位(路径+预览);
    # math.inf → 永不落盘(read_file:靠自身 limit/500KB 护栏界定输出,落盘会致
    #   「读落盘文件→又超阈→循环」)。仿 CC maxResultSizeChars。默认 100K(MCP/未声明工具)。
    max_result_chars: float = 100_000

    def should_defer(self) -> bool:
        """延迟工具:完整 schema 不进模型工具列表,只在 reminder 列名字。内置 False。"""
        return False

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """子 agent 继承本工具时的派生策略。默认返回 self(无状态工具可安全共享实例)。

        cwd(Phase 2 worktree 隔离):worktree 子 agent 派生时传入 worktree dir,持有【每实例
        可变】状态的工具(BashTool._cwd / 6 文件工具 _cwd)覆写为返回带该 cwd 的新实例;无状态
        工具保持默认(忽略 cwd,返回 self)。session_scoped=True 的工具不会走到这里
        (build_child_registry 直接排除)。
        """
        return self

    def schema(self) -> dict[str, Any]:
        """导出给 LLM 的 input_schema。input_schema 优先(MCP 原始 JSON schema);
        否则用 parameters.model_json_schema()。集中在此,registry/tool_search 共用。"""
        if self.input_schema is not None:
            return self.input_schema
        return self.parameters.model_json_schema()

    @abstractmethod
    async def execute(self, **args: object) -> str | ToolOutput:
        """执行工具,返回输出文本 或 ToolOutput(带结构化 tool_use_result)。截断在 Executor 层。"""
        raise NotImplementedError
