# tests/test_agent_loop_context.py
"""Task 9: agent_loop 接入 ContextManager。

4 个场景:
1) loop 每轮调 maybe_compact;
2) overflow → react → 重试同轮 stream(成功);
3) 不传 context(旧路径/未启用持久化):行为不变,ContextOverflow 直接抛交上层;
4) 重试上限:连续 overflow→react→重试不能无限,截了 tail 仍超限时 emit Error + return(防死循环)。
"""
import asyncio

import pytest

from birdcode.agent.agent_loop import run_agent_loop
from birdcode.agent.context import CompactionResult, ContextOverflow
from birdcode.agent.provider import Done, Error, TextDelta
from birdcode.blocks import TextBlock
from birdcode.conversation import Message, Turn


class _CountingCtx:
    """假 ContextManager:记录 maybe_compact / react 调用次数。

    react 模拟硬截断(丢 history 前段,留末条),不调真 LLM。所有方法签名对齐
    birdcode.agent.context.ContextManager,确保 agent_loop 走真路径。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.react_calls = 0

    async def maybe_compact(self, *, history, current, last_in, on_activity=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return None

    async def react(self, *, history):  # type: ignore[no-untyped-def]
        self.react_calls += 1
        del history[:-1]  # 硬截断:留末条(若 history 为空或单条则为 no-op)
        return CompactionResult("reactive", 0, 0, "", "", True)


class _Provider:
    """第一次 stream 抛 ContextOverflow,第二次正常吐 text + Done。"""

    def __init__(self) -> None:
        self._n = 0

    async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
        self._n += 1
        if self._n == 1:
            raise ContextOverflow("too long")
        yield TextDelta(text="ok ")
        yield Done()


class _AlwaysOverflowProvider:
    """每次 stream 都抛 ContextOverflow(测重试上限)。"""

    def __init__(self) -> None:
        self._n = 0

    async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
        self._n += 1
        raise ContextOverflow(f"too long #{self._n}")
        yield  # 不可达:仅为了让本函数被识别为 async generator(否则 stream() 返回 coroutine
        # 而非 async iterator;raise 需在 __anext__ 内触发才能被外层 async for 的 try/except
        # 捕获)。真实 provider 有 yield 故无此问题。


async def _run(turn, ctx, prov):  # type: ignore[no-untyped-def]
    events: list = []

    async def emit(ev):  # type: ignore[no-untyped-def]
        events.append(ev)

    await run_agent_loop(
        provider=prov, turn=turn, history=[], executor=None, emit=emit, context=ctx,
    )
    return events


def test_loop_calls_maybe_compact_each_round():
    """每轮 stream 前调一次 maybe_compact(至少一次)。"""
    ctx = _CountingCtx()
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="hi")])])
    asyncio.run(_run(turn, ctx, _Provider()))
    assert ctx.calls >= 1


def test_loop_catches_context_overflow_calls_react_and_retries():
    """overflow → react 硬截断 → 重试同轮 stream → 成功。react 仅调一次。"""
    ctx = _CountingCtx()
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="hi")])])
    asyncio.run(_run(turn, ctx, _Provider()))
    assert ctx.react_calls == 1  # 第一次 overflow → react → 重试成功


def test_loop_without_context_unchanged_behavior():
    """不传 context(旧路径/未启用持久化):行为不变,ContextOverflow 直接抛交上层。"""
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="hi")])])
    prov = _Provider()

    async def go() -> None:
        async def emit(ev):  # type: ignore[no-untyped-def]
            pass

        await run_agent_loop(
            provider=prov, turn=turn, history=[], executor=None, emit=emit, context=None,
        )

    with pytest.raises(ContextOverflow):
        asyncio.run(go())


def test_loop_react_retry_cap_stops_infinite_overflow():
    """重试上限:连续 overflow→react→重试不能无限。

    截了 tail 仍超限(估算误差 / 尾段仍太大)时,达上限(2 次重试)后不再 react,
    emit Error 提示 /clear 并 return。防 react→continue→overflow 死循环。
    """
    ctx = _CountingCtx()
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="hi")])])
    prov = _AlwaysOverflowProvider()
    events: list = []

    async def go() -> None:
        async def emit(ev):  # type: ignore[no-untyped-def]
            events.append(ev)

        await run_agent_loop(
            provider=prov, turn=turn, history=[], executor=None, emit=emit, context=ctx,
        )

    asyncio.run(go())
    # 上限 2 次重试:react 最多 2 次,stream 最多 3 次(原始 + 2 次重试)
    assert ctx.react_calls <= 2
    assert prov._n <= 3
    # 最终 emit Error 且提示 /clear
    errs = [e for e in events if isinstance(e, Error)]
    assert errs and "/clear" in errs[-1].message
