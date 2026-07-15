"""OpenAI 适配器：chat.completions.create(stream=True) chunk → ProviderEvent。

防御性捕获 delta.reasoning_content（官方 SDK 可能不类型化，但 DeepSeek/vLLM 等会吐）。
用 max_completion_tokens（max_tokens 已弃用且与 o 系列不兼容）。

工具调用流式拼装：delta.tool_calls 按 index 累积（首片带 id+name，后续带 arguments
分片），finish_reason=tool_calls 到达时整体 flush 成 ToolCallStart。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, TypedDict

from openai import AsyncOpenAI

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
    """流式 tool_call 累积态：id/name 来自首片，json 由 arguments 分片拼接，
    finish_reason=tool_calls 到达时整体解析成 ToolCallStart。"""

    id: str
    name: str
    json: str


class OpenAIProvider(_BaseLLMProvider):
    def __init__(
        self,
        profile: ProviderProfile,
        app: AppConfig,
        *,
        client: AsyncOpenAI | None = None,
        registry: ToolRegistry | None = None,
        mcp_instructions: Callable[[], dict[str, str]] | None = None,
        system_override: str | None = None,
    ) -> None:
        super().__init__(
            profile, app, registry=registry, mcp_instructions=mcp_instructions,
            system_override=system_override,
        )
        self._client = client or AsyncOpenAI(base_url=profile.base_url, api_key=profile.api_key)
        self._tool_bufs: dict[int, _ToolBuf] = {}
        self._stop_reason: str | None = None
        # 某些 OpenAI 兼容端点不发 trailing usage,finish_reason chunk 之后
        # 流即结束。用 _finish_seen 标记「已见 finish_reason」,由 _finish_stream 钩子在
        # 流末尾补发 Done;_done_emitted 防止与后续 usage 重复(若两条路径都触发)。
        self._finish_seen: bool = False
        self._done_emitted: bool = False

    def _convert(self, messages: list[Message]) -> list[dict[str, object]]:
        # system prompt 由 _open_stream 统一前置，此处只转对话消息。
        out: list[dict[str, object]] = []
        for m in messages:
            if str(m.role) == "system":
                continue
            texts: list[str] = []
            reasoning: list[str] = []
            tool_uses: list[dict[str, object]] = []
            results: list[ToolResultBlock] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    texts.append(b.text)
                elif isinstance(b, ThinkingBlock):
                    reasoning.append(b.text)
                elif isinstance(b, ToolUseBlock):
                    tool_uses.append(
                        {
                            "id": b.id,
                            "type": "function",
                            "function": {"name": b.name, "arguments": json.dumps(b.input)},
                        }
                    )
                elif isinstance(b, ToolResultBlock):
                    results.append(b)
            msg: dict[str, object] = {"role": m.role}
            if texts:
                msg["content"] = "".join(texts)
            if reasoning:
                msg["reasoning_content"] = "".join(reasoning)
            if tool_uses:
                msg["tool_calls"] = tool_uses
            # 只含 tool_result 的消息不产空壳（role=tool 结果消息紧随其后独立发出）。
            if len(msg) > 1:
                out.append(msg)
            for r in results:
                out.append({"role": "tool", "tool_call_id": r.tool_use_id, "content": r.content})
        return out

    async def _open_stream(
        self, payload: list[dict[str, object]], *, prior_len: int | None = None  # noqa: ARG002 - OpenAI 自动缓存,无需显式断点
    ) -> AsyncIterator[object]:
        self._tool_bufs = {}
        self._stop_reason = None
        self._finish_seen = False
        self._done_emitted = False
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": self._system_text(),
            },
            *payload,
        ]
        kwargs: dict[str, object] = {
            "model": self._profile.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self._app.max_tokens,
        }
        if self._profile.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._profile.reasoning_effort
        if self._registry is not None:
            kwargs["tools"] = self._registry.to_openai_tools()
        return await self._client.chat.completions.create(**kwargs)  # type: ignore[no-any-return,call-overload]

    async def complete(
        self, system: str, user: str, *, max_tokens: int, model: str | None = None
    ) -> str:
        """非流式无工具补全(chat.completions.create,stream 默认 False)。

        摘要专用:不注入 tools / reasoning_effort / reminder。max_completion_tokens
        与 _open_stream 一致(max_tokens 在 o 系列已弃用)。model 非空则覆盖 profile.model。
        """
        resp = await self._client.chat.completions.create(
            model=model or self._profile.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=max_tokens,
            stream=False,
        )
        choices = getattr(resp, "choices", []) or []
        if not choices:
            return ""
        return str(getattr(choices[0].message, "content", "") or "")

    async def summarize_with_prefix(
        self, *, prefix: list[Message], instruction: str, max_tokens: int
    ) -> str:
        """复用主对话 system+tools+prefix 的无工具摘要(tool_choice=none),命中自动 prefix cache。

        OpenAI 自动缓存(无需显式断点);镜像 reasoning_effort;instruction 作末尾 user。
        """
        from birdcode.agent.base_llm import normalize_messages_for_api
        from birdcode.conversation import Message

        flat = normalize_messages_for_api(
            list(prefix) + [Message(role="user", content=[TextBlock(text=instruction)])]
        )
        payload = self._convert(flat)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": self._system_text()},
            *payload,
        ]
        max_tok = max_tokens
        if self._profile.reasoning_effort is not None:
            # reasoning 占 completion 配额,不提升则 reasoning 吃满 → </summary> 截断 → 提取失败。
            # 与 Anthropic 路径(budget+4096)对齐:加固定 margin 留 text 空间。
            max_tok = max_tokens + 4096
        kwargs: dict[str, object] = {
            "model": self._profile.model,
            "messages": messages,
            "max_completion_tokens": max_tok,
            "stream": False,
        }
        if self._profile.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._profile.reasoning_effort
        if self._registry is not None:
            kwargs["tools"] = self._registry.to_openai_tools()
            kwargs["tool_choice"] = "none"
        resp = await self._client.chat.completions.create(**kwargs)  # type: ignore[no-any-return,call-overload]
        choices = getattr(resp, "choices", []) or []
        if not choices:
            return ""
        return str(getattr(choices[0].message, "content", "") or "")

    def _translate(self, raw: object) -> Iterator[ProviderEvent]:
        usage = getattr(raw, "usage", None)
        choices = getattr(raw, "choices", None)
        if usage is not None and not choices:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            # input_tokens 归一为「真实全量请求大小」(BirdCode 到处当全量用:last_in /
            # /context 脚注)。两种 provider 语义:
            # - deepseek 实测:prompt_tokens 只报未命中(miss),命中在 cached → 全量 = miss+hit。
            # - OpenAI 标准:prompt_tokens 已含 cached → 全量 = prompt_tokens。
            # 判据:prompt_tokens < cached 必是 deepseek 语义(全量不可能小于命中部分)。
            full_input = prompt_tokens + cached if prompt_tokens < cached else prompt_tokens
            done_usage = TokenUsage(
                input_tokens=full_input,
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                cache_read_tokens=cached,
            )
            done_usage.cost_usd = estimate_cost(self._profile.model, done_usage)
            yield from self._emit_done(done_usage)
            return
        for choice in choices or []:
            delta = getattr(choice, "delta", None)
            fr = getattr(choice, "finish_reason", None)
            if fr:
                self._stop_reason = "tool_use" if fr == "tool_calls" else "end_turn"
                # finish_reason 到达即流将结束,标 _finish_seen,由
                # _finish_stream 钩子在流末尾补发 Done(若无 trailing usage)。
                # 不在此直发,否则会抢在 trailing usage 之前发 usage=None 的 Done,
                # 破坏既有 usage 断言。
                self._finish_seen = True
            if delta is None:
                continue
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = int(getattr(tc, "index", 0))
                buf = self._tool_bufs.setdefault(idx, {"id": "", "name": "", "json": ""})
                if getattr(tc, "id", None):
                    buf["id"] = str(tc.id)
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        buf["name"] = str(fn.name)
                    if getattr(fn, "arguments", None):
                        buf["json"] += str(fn.arguments)
            if self._stop_reason == "tool_use":
                for idx in sorted(self._tool_bufs):
                    buf = self._tool_bufs[idx]
                    try:
                        inp = json.loads(buf["json"] or "{}")
                    except json.JSONDecodeError:
                        inp = {}
                    yield ToolCallStart(
                        id=buf["id"], name=buf["name"], args=json.dumps(inp, ensure_ascii=False)
                    )
                self._tool_bufs.clear()
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield ThinkingDelta(text=str(reasoning))
            content = getattr(delta, "content", None)
            if content:
                yield TextDelta(text=str(content))

    def _emit_done(self, usage: TokenUsage | None) -> Iterator[ProviderEvent]:
        """收尾 Done 的唯一出口:_done_emitted 保证一流只发一次。

        usage=None 表示无 trailing usage(仅 finish_reason 到达),仍发 Done 让 UI 收尾。
        """
        if self._done_emitted:
            return
        self._done_emitted = True
        yield Done(usage=usage, stop_reason=self._stop_reason or "end_turn")

    def _finish_stream(self) -> Iterator[ProviderEvent]:
        """流正常结束的收尾钩子(override base 空实现)。

        当 finish_reason 已到达但全程没收到 trailing usage 时,流末尾补一次 Done,
        保证 UI 一定收到收尾。_done_emitted 防止与既有 usage 路径重复:
        若 trailing usage 已发过 Done,此处被吞 → 既有 usage 测试仍带 usage。
        """
        if self._finish_seen:
            yield from self._emit_done(None)
