"""_BaseLLMProvider：真实 LLM 适配器的通用骨架。

封装：消息转换、首 token 前重试、事件循环、usage/Error 收尾。子类实现
_convert / _open_stream / _translate 与客户端构造。
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING

from anthropic import (
    APIConnectionError as _AnthropicConn,
)
from anthropic import (
    APIStatusError as _AnthropicStatus,
)
from anthropic import (
    APITimeoutError as _AnthropicTimeout,
)
from anthropic import (
    AuthenticationError as _AnthropicAuth,
)
from anthropic import (
    BadRequestError as _AnthropicBadRequest,
)
from anthropic import (
    NotFoundError as _AnthropicNotFound,
)
from anthropic import (
    PermissionDeniedError as _AnthropicPerm,
)
from anthropic import (
    RateLimitError as _AnthropicRateLimit,
)
from openai import (
    APIConnectionError as _OpenAIConn,
)
from openai import (
    APIStatusError as _OpenAIStatus,
)
from openai import (
    APITimeoutError as _OpenAITimeout,
)
from openai import (
    AuthenticationError as _OpenAIAuth,
)
from openai import (
    BadRequestError as _OpenAIBadRequest,
)
from openai import (
    NotFoundError as _OpenAINotFound,
)
from openai import (
    PermissionDeniedError as _OpenAIPerm,
)
from openai import (
    RateLimitError as _OpenAIRateLimit,
)

from birdcode.agent.context import ContextOverflow
from birdcode.agent.provider import Error, ProviderEvent, TextDelta, ThinkingDelta
from birdcode.agent.system_prompt import build_system_reminder
from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message
from birdcode.memory.loader import load_memory_index
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    from birdcode.config.schema import AppConfig, ProviderProfile
    from birdcode.conversation import Turn
    from birdcode.tools.registry import ToolRegistry

log = get_logger("birdcode.agent.base")

_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    _AnthropicRateLimit,
    _OpenAIRateLimit,
    _AnthropicTimeout,
    _OpenAITimeout,
    _AnthropicConn,
    _OpenAIConn,
)
_PERMANENT_TYPES: tuple[type[BaseException], ...] = (
    _AnthropicAuth,
    _OpenAIAuth,
    _AnthropicBadRequest,
    _OpenAIBadRequest,
    _AnthropicPerm,
    _OpenAIPerm,
    _AnthropicNotFound,
    _OpenAINotFound,
)
_STATUS_TYPES: tuple[type[BaseException], ...] = (_AnthropicStatus, _OpenAIStatus)

# 可重试的 HTTP 状态码：超时/限流/服务端错误。注意 429 必须在此——它是
# RateLimitError 的状态码，但 RateLimitError 同时也是 APIStatusError 的子类，
# 必须显式列为可重试，否则会被「< 500 即永久」的旧逻辑误判为永久错误。
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

# 上下文超限关键词(大小写不敏感)。Anthropic 历史用 400 + "prompt is too long";
# OpenAI 常用 413 或 400 + "context length"。只认 413 会让 Anthropic 后端漏检。
_OVERFLOW_MARKERS: tuple[str, ...] = ("prompt is too long", "context length", "maximum context")


def normalize_messages_for_api(messages: list[Message]) -> list[Message]:
    """规整消息序列,确保满足 LLM API 约束(残缺/损坏/legacy 数据的安全网)。

    - 合并连续同 role 消息(避免「连续两条 user」→ Anthropic 400,旧病根)。
    - 剥离无配对 tool_use 的孤儿 tool_result(其 tool_use 行可能因残缺被 decode 跳过;
      避免「tool_result 无 tool_use」→ 400,旧病根)。
    - 剥离无配对 tool_result 的孤儿 tool_use(中段残缺:decode 跳过损坏 result 行 →
      其前 tool_use 沦为孤儿;repair 只看末尾 Turn 够不到 → normalize 在 flatten 层兜底)。
      assistant 整条剥空则丢弃该消息(不留空 content,否则 400)。
    纯函数:不改输入,返回新列表(Message/Blocks 均复制)。stream 展平 prior+current
    后调用。与 codec.repair_trailing_edge(末尾自愈 + 落盘)互补:repair 修常见崩溃尾部,
    normalize 兜任何残缺/中段/legacy 残留,保证发往 API 的 payload 恒满足交替约束。
    """

    if not messages:
        return []
    use_ids = {b.id for m in messages for b in m.content if isinstance(b, ToolUseBlock)}
    result_ids = {
        b.tool_use_id for m in messages for b in m.content if isinstance(b, ToolResultBlock)
    }
    merged: list[Message] = []
    for m in messages:
        # 剥离孤儿 tool_result(无配对 tool_use)+ 孤儿 tool_use(无配对 tool_result)。
        # 双向对称:repair 只看末尾 Turn,够不到中段残缺(decode 跳过损坏 result 行 →
        # 其前 tool_use 沦为中段孤儿);normalize 在 flatten 层兜任何位置的双向违规。
        blocks = [
            b
            for b in m.content
            if not (isinstance(b, ToolResultBlock) and b.tool_use_id not in use_ids)
            and not (isinstance(b, ToolUseBlock) and b.id not in result_ids)
        ]
        if not blocks:
            continue  # 整条被剥空 → 跳过(避免空消息触发 400)
        if merged and merged[-1].role == m.role:
            merged[-1].content.extend(blocks)  # 合并连续同 role
        else:
            merged.append(Message(role=m.role, content=list(blocks), synthetic=m.synthetic))
    return merged


class _BaseLLMProvider:
    def __init__(
        self,
        profile: ProviderProfile,
        app: AppConfig,
        *,
        registry: ToolRegistry | None = None,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        mcp_instructions: Callable[[], dict[str, str]] | None = None,
        system_override: str | None = None,
    ) -> None:
        self._profile = profile
        self._app = app
        self._registry = registry
        self._mcp_instructions = mcp_instructions
        self._system_override = system_override
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    @property
    def profile(self) -> ProviderProfile:
        """当前 profile(供子 agent 继承 model:resolve_profile 用)。"""
        return self._profile

    def _system_text(self) -> str:
        """组装 system 文本。system_override(子 agent persona)前置到 project_instructions
        位——不走 build_system_text 的 override 逃生口(那会旁路 modules/env/mcp+项目规则),
        保留主 agent 同等的模块/环境,仅把 persona 加在最前。

        子 agent(system_override 非 None)额外走模块精简版:_IDENTITY→子 agent 身份、
        去 _TOOLS(build_system_text(subagent=True))。"""
        from birdcode.agent.system_prompt import build_system_text

        is_subagent = self._system_override is not None
        if is_subagent:
            pi = self._app.project_instructions
            project_instructions = (
                f"{self._system_override}\n\n{pi}" if pi else self._system_override
            )
            override = None
        else:
            project_instructions = self._app.project_instructions
            override = self._app.system_prompt or None
        return build_system_text(
            override=override,
            mcp_instructions=(self._mcp_instructions() if self._mcp_instructions else None),
            project_instructions=project_instructions,
            subagent=is_subagent,
        )

    # —— 子类实现 ——
    def _convert(self, messages: list[Message]) -> list[dict[str, object]]:
        raise NotImplementedError

    async def _open_stream(
        self, payload: list[dict[str, object]], *, prior_len: int | None = None
    ) -> AsyncIterator[object]:
        # prior_len:payload 中 history(prior) 的条数(current 之前);子类据此在 prior 末尾
        # 放 cache_control 断点(prior 稳定命中;current/reminder 是增量)。None=不缓存 messages。
        raise NotImplementedError

    def _translate(self, raw: object) -> Iterator[ProviderEvent]:
        raise NotImplementedError

    def _finish_stream(self) -> Iterator[ProviderEvent]:
        """流正常结束的收尾钩子。默认无操作;子类(OpenAI)可 override 补发收尾事件。"""
        return iter(())

    async def complete(
        self, system: str, user: str, *, max_tokens: int, model: str | None = None
    ) -> str:
        """无工具一次性补全(摘要专用)。子类用各自 client 非流式 API override 实现。"""
        raise NotImplementedError

    async def summarize_with_prefix(
        self, *, prefix: list[Message], instruction: str, max_tokens: int
    ) -> str:
        """复用主对话 system+tools+prefix 缓存的无工具摘要(tool_choice=none),命中 prompt cache。

        prefix=历史前缀(命中主对话 messages 缓存,因主对话 prior 末有断点);instruction=摘要
        模板(末尾 user,落断点之后)。镜像 profile.thinking(否则发含 ThinkingBlock 的 prefix →
        Anthropic 400)。子类用各自 client 非流式 API override。
        """
        raise NotImplementedError

    def _is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, _RETRYABLE_TYPES):
            return True
        if isinstance(exc, _STATUS_TYPES):
            return int(exc.status_code) in _RETRYABLE_STATUS  # type: ignore[attr-defined]
        return False

    def _is_permanent(self, exc: BaseException) -> bool:
        # 互斥：可重试的绝不算永久（修复 429 被误判为永久的关键）。
        if self._is_retryable(exc):
            return False
        if isinstance(exc, _PERMANENT_TYPES):
            return True
        # 其余任意 status error 都视作不可重试的 4xx。
        if isinstance(exc, _STATUS_TYPES):
            return True
        return False

    @staticmethod
    def _is_context_overflow(exc: BaseException) -> bool:
        """是否上下文超限:413,或 400 + 超限关键词。

        Anthropic 的 "prompt is too long" 历史用 400 BadRequestError(不是 413),
        只认 413 会让 Reactive 在 Anthropic 后端永不触发。故双兜底。
        """
        status = getattr(exc, "status_code", None)
        if status == 413:
            return True
        if status == 400:
            msg = str(exc).lower()
            return any(m in msg for m in _OVERFLOW_MARKERS)
        return False

    async def _inject_reminder(self, messages: list[Message]) -> list[Message]:
        """给当前轮首条 user 消息末尾追加 <system-reminder>(末尾位置,recency 注意力更高)。

        大模型对上下文首尾注意力权重更高;reminder 放在 user 消息末尾(而非前置),首轮里它就是
        最后一条内容,离生成点最近、最显眼。构造副本、不 mutate 原 Message;每次 stream 从干净
        messages[0] 重建 → 天然幂等,不累积。动态环境位于缓存断点之后,不破坏 system+tools 前缀缓存。
        记忆索引每轮现读磁盘(见 _memory_index)注入此 reminder——提取一旦落盘,下一轮 stream 即见。
        """
        if not messages:
            return messages
        memory_idx = self._memory_index()
        reminder = await build_system_reminder(
            registry=self._registry, memory=memory_idx,
            subagent=self._system_override is not None,
        )
        if not reminder:
            return messages
        first = messages[0]
        if first.role != "user":
            return messages
        new_first = Message(
            role=first.role, content=[*first.content, TextBlock(text=reminder)]
        )
        return [new_first, *messages[1:]]

    def _memory_index(self) -> str:
        """每轮现读记忆索引(项目级 + 用户级),供 <system-reminder> 注入。

        从 cfg.project_root 派生两个 memory 目录、扫 .md 现读——故记忆更新本会话内及时可见
        (提取落盘后下一轮即见),代价是记忆块不进前缀缓存(每轮重发)。无 cfg / project_root → ""。
        读取失败只 log,绝不杀 stream(记忆是锦上添花,非必需)。
        """
        project_root = self._app.project_root if self._app is not None else None
        if project_root is None:
            return ""
        try:
            return load_memory_index(project_root)
        except Exception:  # noqa: BLE001 - 记忆读取失败不杀主循环
            log.warning("读取记忆索引失败,本轮 reminder 不带记忆", exc_info=True)
            return ""

    async def stream(
        self, messages: list[Message], *, history: list[Turn]
    ) -> AsyncIterator[ProviderEvent]:
        prior = [m for turn in history for m in turn.messages]
        current = await self._inject_reminder(list(messages))
        # normalize:合并连续同 role + 剥离孤儿 tool_result,兜住残缺/legacy/中段违规
        # (repair 只修末尾且落盘;normalize 保证发往 API 的 payload 恒满足交替约束)。
        flat = normalize_messages_for_api(prior + current)
        # prior_len:flat 中 current 之前的条数,供 _open_stream 在 prior 末放 cache 断点。
        # 注意不变性 normalize(prior+current)[:n]==normalize(prior) 并非总成立:normalize 会
        # 合并连续同 role 消息。工具循环中 history 末常是 user(tool_result)、current 首条是
        # user(text)→ 同 role 被并入 prior 末条,使 prior_len==len(flat)。此时断点若落末条
        # 会纳入 current(reminder 每轮变→cache miss),前移一条让断点落在纯 prior 稳定区。
        prior_len = len(normalize_messages_for_api(prior)) if prior else 0
        if prior_len and prior_len == len(flat) and len(flat) > 1:
            prior_len = len(flat) - 1
        payload = self._convert(flat)
        attempt = 0
        while True:
            yielded_token = False
            try:
                raw_stream = await self._open_stream(payload, prior_len=prior_len)
                async for raw in raw_stream:
                    for ev in self._translate(raw):
                        if isinstance(ev, (TextDelta, ThinkingDelta)):
                            yielded_token = True
                        yield ev
                # 流正常结束:让子类补尾(OpenAI 在无 trailing usage 时补 Done)。
                for ev in self._finish_stream():
                    yield ev
                return
            except Exception as exc:  # noqa: BLE001 — 统一收口转 Error
                if self._is_context_overflow(exc):
                    # 上下文超限:抛 ContextOverflow 交 agent_loop 做 Reactive(硬截断+重试),
                    # 不在此 yield Error(否则安全网失效)。
                    raise ContextOverflow(str(exc) or "context overflow") from exc
                if self._is_permanent(exc):
                    yield Error(message=self._err(exc))
                    return
                if self._is_retryable(exc) and not yielded_token and attempt < self._max_retries:
                    attempt += 1
                    await asyncio.sleep(self._backoff(attempt))
                    log.warning("provider 可重试错误，第 %d 次重试: %s", attempt, exc)
                    continue
                yield Error(message=self._err(exc))
                return

    def _backoff(self, attempt: int) -> float:
        delay: float = self._backoff_base * (2 ** (attempt - 1)) + random.uniform(
            0, self._backoff_base
        )
        return delay

    @staticmethod
    def _err(exc: BaseException) -> str:
        return str(exc) or exc.__class__.__name__
