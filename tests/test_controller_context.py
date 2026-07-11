# tests/test_controller_context.py
"""Task 10:TurnController 把 context 透传给 run_agent_loop(context=...)。"""

import pytest

from birdcode.agent.provider import Done
from birdcode.blocks import TextBlock
from birdcode.conversation import Message, Turn, TurnController


class _Prov:
    """最小 provider:stream 立即结束(本测试不关心 provider 行为,只验 context 透传)。"""

    async def stream(self, messages, *, history):  # noqa: ARG002
        yield Done()


@pytest.mark.asyncio
async def test_controller_passes_context_to_run_agent_loop(monkeypatch):
    """TurnController(..., context=ctx) → run_agent_loop 收到 context=ctx。"""
    captured: dict = {}

    async def fake_loop(  # noqa: ARG001 - 只捕 context,其余参数忽略
        *, provider, turn, history, executor, emit, on_message=None, context=None, **kwargs
    ):
        captured["context"] = context
        await emit(Done())

    # _consume 内每次都 `from birdcode.agent.agent_loop import run_agent_loop` 重读属性,
    # 故 monkeypatch 模块级属性即可拦截。
    monkeypatch.setattr("birdcode.agent.agent_loop.run_agent_loop", fake_loop)

    class _Ctx:
        pass

    ctx = _Ctx()

    async def on_event(ev): ...  # noqa: ARG001
    async def on_status(): ...

    ctrl = TurnController(_Prov(), on_event=on_event, on_status=on_status, context=ctx)
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="hi")])])
    await ctrl._consume(turn)

    assert captured["context"] is ctx


@pytest.mark.asyncio
async def test_controller_defaults_context_to_none_when_not_passed(monkeypatch):
    """不传 context(旧路径/无 store)→ run_agent_loop 收到 context=None,行为不变。"""
    captured: dict = {}

    async def fake_loop(  # noqa: ARG001
        *, provider, turn, history, executor, emit, on_message=None, context=None, **kwargs
    ):
        captured["context"] = context
        await emit(Done())

    monkeypatch.setattr("birdcode.agent.agent_loop.run_agent_loop", fake_loop)

    async def on_event(ev): ...  # noqa: ARG001
    async def on_status(): ...

    # 不传 context(向后兼容:旧调用风格)
    ctrl = TurnController(_Prov(), on_event=on_event, on_status=on_status)
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="hi")])])
    await ctrl._consume(turn)

    assert captured["context"] is None
