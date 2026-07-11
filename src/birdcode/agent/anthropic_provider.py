"""Anthropic 适配器：messages.create(stream=True) raw 事件 → ProviderEvent。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, TypedDict

from anthropic import AsyncAnthropic

from birdcode.agent.base_llm import _BaseLLMProvider
from birdcode.agent.pricing import estimate_cost
from birdcode.agent.provider import (
    Done,
    ProviderEvent,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
    ToolCallStart,
)
from birdcode.blocks import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:
    from birdcode.config.schema import AppConfig, ProviderProfile
    from birdcode.conversation import Message
    from birdcode.tools.registry import ToolRegistry


class _ToolBuf(TypedDict):
    """Streaming tool_use 累积态:id/name 来自 content_block_start,json 由
    input_json_delta 的 partial_json 片段拼接而成,content_block_stop 时整体解析。"""

    id: str
    name: str
    json: str


class AnthropicProvider(_BaseLLMProvider):
    def __init__(
        self,
        profile: ProviderProfile,
        app: AppConfig,
        *,
        client: AsyncAnthropic | None = None,
        registry: ToolRegistry | None = None,
        mcp_instructions: Callable[[], dict[str, str]] | None = None,
        system_override: str | None = None,
    ) -> None:
        super().__init__(
            profile, app, registry=registry, mcp_instructions=mcp_instructions,
            system_override=system_override,
        )
        self._client = client or AsyncAnthropic(base_url=profile.base_url, api_key=profile.api_key)
        self._input_tokens = 0
        self._output_tokens = 0
        self._tool_buf: _ToolBuf | None = None
        self._stop_reason: str | None = None
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0

    def _convert(self, messages: list[Message]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for m in messages:
            if str(m.role) == "system":
                continue
            blocks: list[dict[str, object]] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ThinkingBlock):
                    blocks.append(
                        {"type": "thinking", "thinking": b.text, "signature": b.signature}
                    )
                elif isinstance(b, ToolUseBlock):
                    blocks.append(
                        {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                    )
                elif isinstance(b, ToolResultBlock):
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b.tool_use_id,
                            "content": b.content,
                            "is_error": b.is_error,
                        }
                    )
            out.append({"role": m.role, "content": blocks})
        return out

    async def _open_stream(self, payload: list[dict[str, object]]) -> AsyncIterator[object]:
        self._input_tokens = 0
        self._output_tokens = 0
        self._tool_buf = None
        self._stop_reason = None
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0
        kwargs: dict[str, object] = {
            "model": self._profile.model,
            "max_tokens": self._app.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": self._system_text(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": payload,
            "stream": True,
        }
        if self._profile.thinking is not None:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._profile.thinking.budget_tokens,
            }
        if self._registry is not None:
            kwargs["tools"] = self._registry.to_anthropic_tools(cached=True)
            kwargs["tool_choice"] = {"type": "auto"}
        return await self._client.messages.create(**kwargs)  # type: ignore[no-any-return,call-overload]

    async def complete(
        self, system: str, user: str, *, max_tokens: int, model: str | None = None
    ) -> str:
        """非流式无工具补全(messages.create,stream 默认 False)。拼回 content 里 text blocks。

        摘要专用:不注入 tools / tool_choice / reminder,system 直接用传入字符串
        (ContextManager 的摘要模板,非对话 system_prompt)。model 非空则覆盖 profile.model。
        """
        resp = await self._client.messages.create(
            model=model or self._profile.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            stream=False,
        )
        parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", "") == "text":
                parts.append(str(getattr(block, "text", "") or ""))
        return "".join(parts)

    def _translate(self, raw: object) -> Iterator[ProviderEvent]:
        t = getattr(raw, "type", "")
        if t == "content_block_start":
            block = getattr(raw, "content_block", None)
            if getattr(block, "type", "") == "tool_use":
                self._tool_buf = {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "json": "",
                }
        elif t == "content_block_delta":
            delta = getattr(raw, "delta", None)
            dt = getattr(delta, "type", "")
            if dt == "thinking_delta":
                yield ThinkingDelta(text=str(getattr(delta, "thinking", "") or ""))
            elif dt == "signature_delta":
                yield ThinkingDelta(text="", signature=str(getattr(delta, "signature", "") or ""))
            elif dt == "text_delta":
                yield TextDelta(text=str(getattr(delta, "text", "") or ""))
            elif dt == "input_json_delta" and isinstance(self._tool_buf, dict):
                self._tool_buf["json"] += str(getattr(delta, "partial_json", "") or "")
        elif t == "content_block_stop":
            if isinstance(self._tool_buf, dict):
                try:
                    inp = json.loads(self._tool_buf["json"] or "{}")
                except json.JSONDecodeError:
                    inp = {}
                yield ToolCallStart(
                    id=self._tool_buf["id"],
                    name=self._tool_buf["name"],
                    args=json.dumps(inp, ensure_ascii=False),
                )
                self._tool_buf = None
        elif t == "message_start":
            msg = getattr(raw, "message", None)
            usage = getattr(msg, "usage", None)
            self._input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            self._cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            self._cache_creation_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        elif t == "message_delta":
            usage = getattr(raw, "usage", None)
            self._output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            d = getattr(raw, "delta", None)
            sr = getattr(d, "stop_reason", None) if d is not None else None
            if sr:
                self._stop_reason = str(sr)
        elif t == "message_stop":
            usage = TokenUsage(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cache_read_tokens=self._cache_read_tokens,
                cache_creation_tokens=self._cache_creation_tokens,
            )
            usage.cost_usd = estimate_cost(self._profile.model, usage)
            yield Done(usage=usage, stop_reason=self._stop_reason or "end_turn")
