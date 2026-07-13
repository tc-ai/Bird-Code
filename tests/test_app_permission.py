# tests/test_app_permission.py
import pytest

from birdcode.ui.app import BirdApp


def test_mode_reactive_defaults_default():
    # reactive 是类属性;核默认值(Textual reactive 用 _default 存默认)。
    # 默认档改为 "default":逐次 HITL 确认(更安全的首启体验)。
    assert BirdApp.mode._default == "default"  # noqa: SLF001


@pytest.mark.asyncio
async def test_on_event_edit_diff_calls_set_diff():
    """收到 ToolResult → EditDiff → 对应 ToolLine.set_diff 被调用(agent_loop 真实顺序)。"""
    from birdcode.agent.provider import EditDiff, ToolCallStart, ToolResult

    app = BirdApp(provider=_DummyProvider())
    async with app.run_test() as pilot:
        await app._on_event(ToolCallStart(id="tu1", name="edit_file", args="{}"))
        await pilot.pause()
        line = app._tools.get("tu1")
        assert line is not None
        # ToolResult 先到(agent_loop 真实顺序),把 line 暂存进 _diff_pending
        await app._on_event(ToolResult(id="tu1", ok=True, summary="ok"))
        await pilot.pause()
        called: list = []
        line.set_diff = lambda old, new: called.append((old, new))  # type: ignore[method-assign]
        await app._on_event(EditDiff(id="tu1", old="a\n", new="b\n"))
        await pilot.pause()
        assert called == [("a\n", "b\n")]


@pytest.mark.asyncio
async def test_shift_tab_cycles_mode():
    """shift+tab 绑定(priority)抢占焦点逆向导航,循环切档 default→plan→accept-edits→bypass→default。

    已用一次性探针确认 pilot.press('shift+tab') 在本环境触发 action_cycle_mode。
    """
    app = BirdApp(provider=_DummyProvider())
    async with app.run_test() as pilot:
        assert app.mode == "default"  # 起点(默认档)
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.mode == "plan"  # default → plan
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.mode == "accept-edits"  # plan → accept-edits
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.mode == "bypass"  # accept-edits → bypass
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.mode == "default"  # bypass → default(回绕到起点)


class _DummyProvider:
    """最小 provider,只满足 TurnController 构造签名;不实际 stream。"""

    async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
        return
        yield  # 让它成为 async generator
