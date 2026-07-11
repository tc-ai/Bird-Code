import asyncio

import pytest

from birdcode.agent.provider import (
    Done,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
    ToolCallStart,
)
from birdcode.blocks import TextBlock
from birdcode.ui.app import BirdApp
from birdcode.ui.widgets.thinking_block import ThinkingBlock


class ThinkingFakeProvider:
    """TurnStart(controller) → ThinkingDelta×2 → TextDelta×2 → Done."""

    async def stream(self, messages, *, history):
        yield ThinkingDelta(text="reasoning 1 ")
        yield ThinkingDelta(text="reasoning 2 ")
        yield TextDelta(text="answer ")
        yield TextDelta(text="here")
        yield Done(usage=TokenUsage(input_tokens=1, output_tokens=2))


@pytest.mark.asyncio
async def test_thinking_block_mounted_and_history_excludes_thinking():
    provider = ThinkingFakeProvider()
    async with BirdApp(provider).run_test(size=(80, 24)) as pilot:
        ctrl = pilot.app.controller
        await ctrl.submit("hi")
        await pilot.pause()
        blocks = list(pilot.app.query(ThinkingBlock))
        assert len(blocks) == 1  # 一个思考块
        assert blocks[0]._text  # 收到了 reasoning
        # history 只含正文，不含 thinking
        last = ctrl.history[-1]
        assistant = [m for m in last.messages if m.role == "assistant"]
        text = "".join(b.text for b in assistant[0].content if isinstance(b, TextBlock))
        assert assistant and "answer here" in text
        assert "reasoning" not in text


class MultiRoundThinkingProvider:
    """一次回复内两段 thinking(中间夹 text → _finish_thinking → 下一 ThinkingDelta 起新块)。"""

    async def stream(self, messages, *, history):
        yield ThinkingDelta(text="第一段思考 ")
        yield TextDelta(text="中间文本")  # 触发 _finish_thinking → _thinking=None
        yield ThinkingDelta(text="第二段思考 ")  # 新块,_thinking_round +1
        yield TextDelta(text="结论")
        yield Done(usage=TokenUsage(input_tokens=1, output_tokens=2))


@pytest.mark.asyncio
async def test_thinking_round_resets_each_turn():
    """每条用户消息的思考轮次从 1 重新起算,不延续上一问。

    回归:begin_assistant_turn 未重置 _thinking_round → 第二问首段思考被标「第 3 轮」。
    修复后两轮各自 [1, 2];修复前会是 [1, 2, 3, 4]。
    """
    provider = MultiRoundThinkingProvider()
    async with BirdApp(provider).run_test(size=(80, 24)) as pilot:
        await pilot.app.controller.submit("第一条")
        await pilot.pause()
        assert [b._round for b in pilot.app.query(ThinkingBlock)] == [1, 2]

        await pilot.app.controller.submit("第二条")
        await pilot.pause()
        assert [b._round for b in pilot.app.query(ThinkingBlock)] == [1, 2, 1, 2]


@pytest.mark.asyncio
async def test_clear_removes_thinking_blocks():
    """/clear 必须清掉散落挂在 #scroll 的 ThinkingBlock。

    回归:ThinkingBlock 直接 mount 在 #scroll（非 Turn 内），旧的 query(".turn")
    抓不到它 → /clear 后残留、仍可点击展开。
    """
    provider = ThinkingFakeProvider()
    async with BirdApp(provider).run_test(size=(80, 24)) as pilot:
        await pilot.app.controller.submit("hi")
        await pilot.pause()
        assert len(list(pilot.app.query(ThinkingBlock))) == 1
        await pilot.app._dispatch(pilot.app._command_registry.resolve("/clear"), "")
        await pilot.pause()
        assert list(pilot.app.query(ThinkingBlock)) == []


class _TwoRoundProvider:
    """第1轮 thinking+tool_use(echo);第2轮 thinking+text(总结)。验第2轮 text 不丢。"""

    def __init__(self) -> None:
        self.n = 0

    async def stream(self, messages, *, history):
        self.n += 1
        if self.n == 1:
            yield ThinkingDelta(text="r1 think ")
            yield ToolCallStart(id="t1", name="echo", args='{"text": "x"}')
            yield Done(usage=TokenUsage(input_tokens=1, output_tokens=1), stop_reason="tool_use")
        else:
            yield ThinkingDelta(text="r2 think ")
            yield TextDelta(text="这是总结")
            yield Done(usage=TokenUsage(input_tokens=1, output_tokens=1))


@pytest.mark.asyncio
async def test_second_round_text_not_silently_dropped():
    """回归:多轮中第1轮 Done 清了 _md_stream,第2轮 TextDelta(总结)若不重建 md 会被丢弃。"""
    from birdcode.ui.widgets.turn import Turn

    provider = _TwoRoundProvider()
    async with BirdApp(provider).run_test(size=(80, 24)) as pilot:
        await pilot.app.controller.submit("读并总结")
        await pilot.pause()
        # 模型确实输出了总结(history 佐证 agent_loop 收到了 TextDelta)
        assert "这是总结" in pilot.app.transcript()
        # UI 也显示了出来:第2轮重建了 Turn(修复前只有 1 个 Turn,总结被丢)
        turns = list(pilot.app.query(Turn))
        assert len(turns) >= 2


@pytest.mark.asyncio
async def test_interrupt_marks_spinning_tool_lines():
    """Esc 中断时,转 spinner 的工具行应停并标「⚡ 中断」。

    回归:中断时正在执行的工具没机会收到 ToolResult,spinner 一直转、盖在 [interrupted]
    上方,用户看不出已中断。修复:Interrupted 事件主动 finish 这些工具行。
    """
    from pydantic import BaseModel

    from birdcode.tools import Tool, ToolRegistry
    from birdcode.tools.executor import ToolExecutor
    from birdcode.ui.widgets.tool_line import ToolLine

    class _NoInput(BaseModel):
        pass

    class _SlowTool(Tool):
        name = "slow"
        description = "slow"
        parameters = _NoInput
        kind = "read"
        parallel_safe = True

        async def execute(self, **a):  # type: ignore[override]
            await asyncio.sleep(10)
            return "done"

    class _SlowProvider:
        async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
            yield ToolCallStart(id="s1", name="slow", args="{}")
            yield Done(usage=TokenUsage(), stop_reason="tool_use")

    reg = ToolRegistry()
    reg.register(_SlowTool())
    async with BirdApp(_SlowProvider()).run_test(size=(80, 24)) as pilot:
        pilot.app.controller._executor = ToolExecutor(reg)
        asyncio.create_task(pilot.app.controller.submit("run"))
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()
        await pilot.press("escape")  # Esc 中断
        await asyncio.sleep(0.5)
        await pilot.pause()
        lines = list(pilot.app.query(ToolLine))
        assert lines and "⚡ 中断" in lines[0]._last_summary
        assert lines[0]._timer is None  # spinner 已停
