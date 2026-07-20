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
            profile,
            app,
            registry=registry,
            mcp_instructions=mcp_instructions,
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
                    if self._profile.thinking is None:
                        # profile 未开 thinking:不发 ThinkingBlock(否则 API 400)。免疫切
                        # profile(thinking→非 thinking)后 history 残留 ThinkingBlock 的回归。
                        continue
                    blocks.append(
                        {"type": "thinking", "thinking": b.text, "signature": b.signature}
                    )
                elif isinstance(b, ToolUseBlock):
                    # agent_id 是持久化字段(block_to_dict 写 jsonl),本白名单不带 → 绝不进 API
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

    async def _open_stream(
        self, payload: list[dict[str, object]], *, prior_len: int | None = None
    ) -> AsyncIterator[object]:
        self._input_tokens = 0
        self._output_tokens = 0
        self._tool_buf = None
        self._stop_reason = None
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0
        # 第 3 个 cache 断点(前两个:system/tools):落 prior 末尾(current 之前)的最后 block。
        # prior(history,无 reminder)稳定 → 每轮命中;current(含 reminder)是增量。reminder 不进
        # history,上轮 current 进 history 是干净版 → 前缀连续命中(不被 reminder 打断)。
        if prior_len and 0 < prior_len <= len(payload):
            blocks = payload[prior_len - 1].get("content")
            if blocks:  # 防空(content 被 normalize 剥光)
                blocks[-1]["cache_control"] = {"type": "ephemeral"}
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

    async def summarize_with_prefix(
        self, *, prefix: list[Message], instruction: str, max_tokens: int
    ) -> str:
        """复用主对话 system+tools+prefix 缓存的无工具摘要(tool_choice=none),命中 prompt cache。

        三段同构 _open_stream(system+tools 带 cache 断点 + prefix 末断点)→ 命中主对话缓存。
        镜像 profile.thinking:prefix 含 ThinkingBlock 时必须开 thinking,否则 API 400;
        且 max_tokens 须 > budget 留 text 空间,否则 </summary> 被截断 → 提取失败。instruction
        作末尾 user(落 prefix 断点之后,新内容)。
        """
        from birdcode.agent.base_llm import normalize_messages_for_api
        from birdcode.conversation import Message

        flat = normalize_messages_for_api(
            list(prefix) + [Message(role="user", content=[TextBlock(text=instruction)])]
        )
        prior_len = len(normalize_messages_for_api(list(prefix))) if prefix else 0
        # 同 base_llm.stream:prefix 末若与 instruction(user)同 role → normalize 合并使
        # prior_len==len(flat),断点落末条会纳入 instruction → 前移一条落纯 prefix 稳定区。
        if prior_len and prior_len == len(flat) and len(flat) > 1:
            prior_len = len(flat) - 1
        payload = self._convert(flat)
        # 第 3 个 cache 断点:prefix 末(instruction 之前)→ 命中主对话 prior 缓存。
        if prior_len and 0 < prior_len <= len(payload):
            blocks = payload[prior_len - 1].get("content")
            if blocks:
                blocks[-1]["cache_control"] = {"type": "ephemeral"}
        max_tok = max_tokens
        if self._profile.thinking is not None:
            # 须 > budget 留 text 空间,否则摘要被 thinking 占满 → </summary> 截断 → 提取失败。
            max_tok = max(max_tokens, self._profile.thinking.budget_tokens + 4096)
        kwargs: dict[str, object] = {
            "model": self._profile.model,
            "max_tokens": max_tok,
            "system": [
                {
                    "type": "text",
                    "text": self._system_text(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": payload,
            "stream": False,
        }
        if self._profile.thinking is not None:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._profile.thinking.budget_tokens,
            }
        if self._registry is not None:
            kwargs["tools"] = self._registry.to_anthropic_tools(cached=True)
            kwargs["tool_choice"] = {"type": "none"}
        resp = await self._client.messages.create(**kwargs)  # type: ignore[no-any-return,call-overload]
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
