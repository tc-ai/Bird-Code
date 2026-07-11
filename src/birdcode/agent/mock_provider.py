# src/birdcode/agent/mock_provider.py
"""Deterministic Mock StreamingProvider for the UI milestone.

两阶段（stateful）流，配合 agent_loop 跑通真实的工具往返：
- 第一阶段（user 文本为末条消息）：回显一句 + 发起 echo 工具调用，stop_reason=tool_use
  → agent_loop 把 tool_use 转交 executor 执行并回填 tool_result。
- 第二阶段（末条消息是 tool_result）：再次流式回显用户原文，Done(end_turn) 收尾。

CancelledError 原样上抛做清理。真实 LLM/provider 是后续里程碑；无论后端如何，UI 行为一致。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from birdcode.agent.provider import (
    Done,
    ProviderEvent,
    TextDelta,
    TokenUsage,
    ToolCallStart,
)
from birdcode.blocks import TextBlock, ToolResultBlock
from birdcode.conversation import Message, Turn


class MockProvider:
    def __init__(self, *, delay: float = 0.012) -> None:
        self._delay = delay

    async def complete(
        self, system: str, user: str, *, max_tokens: int, model: str | None = None
    ) -> str:
        """摘要用:把 user 内容包进 <summary>(供 ContextManager 的 _extract_summary 提取逻辑测试)。

        与真实 provider 的 complete 同签名;system/max_tokens 在 mock 下不影响输出
        (确定性回显,不调任何外部服务)。
        """
        return f"<analysis>draft</analysis>\n<summary>{user}</summary>"

    async def stream(
        self, messages: list[Message], *, history: list[Turn]
    ) -> AsyncIterator[ProviderEvent]:
        last_is_result = bool(
            messages
            and messages[-1].content
            and isinstance(messages[-1].content[-1], ToolResultBlock)
        )
        user_text = ""
        if messages:
            user_text = "".join(b.text for b in messages[0].content if isinstance(b, TextBlock))
        try:
            if last_is_result:
                for word in user_text.split() or ["..."]:
                    await asyncio.sleep(self._delay)
                    yield TextDelta(text=word + " ")
                yield Done(
                    usage=TokenUsage(
                        input_tokens=max(1, len(user_text.split())),
                        output_tokens=max(1, len(user_text.split())),
                        cost_usd=0.0001 * max(1, len(user_text.split())),
                    )
                )
            else:
                yield TextDelta(text="我先回显一下: ")
                yield ToolCallStart(id="mock-1", name="echo", args=json.dumps({"text": user_text}))
                yield Done(
                    usage=TokenUsage(
                        input_tokens=max(1, len(user_text.split())),
                        output_tokens=1,
                        cost_usd=0.0,
                    ),
                    stop_reason="tool_use",
                )
        except asyncio.CancelledError:
            # cleanup: nothing to release for the mock; re-raise so the task cancels cleanly
            raise
