# tests/test_turn.py
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static

from birdcode.ui.icons import select_icons
from birdcode.ui.widgets.tool_line import ToolLine
from birdcode.ui.widgets.turn import Turn


@pytest.mark.asyncio
async def test_turn_holds_children():
    icons = select_icons(unicode_supported=True)

    class _App(App):
        def compose(self) -> ComposeResult:
            yield Turn()

    async with _App().run_test() as pilot:
        await pilot.pause()
        turn = pilot.app.query_one(Turn)
        await turn.add_user("hello")
        md = await turn.add_assistant_markdown()
        tool = await turn.add_tool(icons)
        await pilot.pause()
        assert isinstance(turn.query_one(".user", Static), Static)
        assert isinstance(md, Markdown)
        assert isinstance(tool, ToolLine)
        assert "turn" in turn.classes


@pytest.mark.asyncio
async def test_add_user_empty_text_is_noop():
    """#9:空 user_text 不应挂空 .user Static(wake 轮 TurnStart(user_text='') 不当用户气泡)。"""
    from textual.app import App, ComposeResult

    class _App(App):
        def compose(self) -> ComposeResult:
            yield Turn()

    async with _App().run_test() as pilot:
        await pilot.pause()
        turn = pilot.app.query_one(Turn)
        await turn.add_user("")
        await pilot.pause()
        assert list(turn.query(".user")) == [], "空文本不应渲染 .user Static"
        # 非空文本照常渲染
        await turn.add_user("hi")
        await pilot.pause()
        assert len(list(turn.query(".user"))) == 1
