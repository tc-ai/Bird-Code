# src/birdcode/agent/provider.py
"""Provider contract: the only seam between the UI and any future LLM/agent backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from birdcode.conversation import Message, Turn


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0       # Anthropic cache_read_input_tokens / OpenAI cached_tokens
    cache_creation_tokens: int = 0   # 仅 Anthropic;OpenAI 恒 0
    cost_usd: float = 0.0


class TurnStart(BaseModel):
    """由 controller 在每轮开头发出（provider 不产出），把本轮 user_text 随事件流
    带回 UI——UI 据此开轮次，不再用共享态延迟读取（修 type-ahead 下用户气泡标错）。"""

    type: Literal["turn_start"] = "turn_start"
    user_text: str


class TextDelta(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingDelta(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str
    signature: str = ""  # Anthropic thinking 签名,随增量带给 agent_loop


class ToolCallStart(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    id: str
    name: str
    args: str = ""


class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    id: str
    ok: bool = True
    summary: str = ""
    output: str = ""  # stage3:截断后的完整正文,供 UI 折叠区显示
    truncated: bool = False  # 透传 executor 的截断标记到 UI


class Done(BaseModel):
    type: Literal["done"] = "done"
    usage: TokenUsage | None = None
    stop_reason: str | None = None  # "end_turn" / "tool_use" / ...


class Error(BaseModel):
    type: Literal["error"] = "error"
    message: str


class Interrupted(BaseModel):
    type: Literal["interrupted"] = "interrupted"


class EditDiff(BaseModel):
    """Edit/Write 成功后的 (old, new) 文本,供 UI 挂 diff 高亮子区。"""

    type: Literal["edit_diff"] = "edit_diff"
    id: str
    old: str
    new: str


class CompactionStart(BaseModel):
    """自动压缩开始:UI 挂一条橙色转圈行(仿工具调用)。

    由 agent_loop 在 ContextManager 真正开始摘要(超阈值)时 emit;
    manual /compact 不走事件流(由命令自身提示)。
    """

    type: Literal["compaction_start"] = "compaction_start"
    reason: str = "auto"  # "auto" | "manual" | "reactive"


class CompactionEnd(BaseModel):
    """自动压缩结束:停转圈、显示压缩结果摘要(橙色)。"""

    type: Literal["compaction_end"] = "compaction_end"
    reason: str = "auto"
    summary: str = ""  # 给 UI 展示的收尾文本,如 "已压缩 32000→8000 token"
    fell_back: bool = False  # 是否走了硬截断兜底(摘要失败/超限)


ProviderEvent = Annotated[
    TurnStart
    | TextDelta
    | ThinkingDelta
    | ToolCallStart
    | ToolResult
    | Done
    | Error
    | Interrupted
    | EditDiff
    | CompactionStart
    | CompactionEnd,
    Field(discriminator="type"),
]


@runtime_checkable
class StreamingProvider(Protocol):
    # NOTE: 必须用普通 def（非 async def）声明返回 AsyncIterator 的协议方法。
    # mypy 会把「无 yield 的 async def」推断成 Callable[..., Coroutine[..., AsyncIterator]]，
    # 与真实 async generator 实现不兼容（结构性匹配失败）。详见 mypy「More Types」。
    def stream(
        self, messages: list[Message], *, history: list[Turn]
    ) -> AsyncIterator[ProviderEvent]: ...

    async def complete(
        self, system: str, user: str, *, max_tokens: int, model: str | None = None
    ) -> str:
        """无工具一次性补全(摘要专用:tool_choice=none,不注入 reminder)。返回纯文本。

        model 非空时覆盖 profile.model(记忆提取用,走便宜模型);None 用 profile.model。
        """
        ...

    async def summarize_with_prefix(
        self, *, prefix: list[Message], instruction: str, max_tokens: int
    ) -> str:
        """复用主对话 system+tools+prefix 缓存的无工具摘要(tool_choice=none)。

        prefix=历史前缀(命中主对话 messages 缓存);instruction=摘要模板(末尾 user,落断点之后)。
        与 complete 的区别:复用主对话 system/tools/prefix(命中 cache),而非独立 system。
        """
        ...
