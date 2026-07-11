# tests/test_input_area.py
import pytest
from textual import on
from textual.app import App, ComposeResult

from birdcode.ui.widgets.input_area import InputArea


class _Harness(App):
    submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputArea()

    @on(InputArea.Submitted)
    def _record_submit(self, event: InputArea.Submitted) -> None:
        self.submitted.append(event.text)


@pytest.mark.asyncio
async def test_typing_and_submit():
    _Harness.submitted = []
    async with _Harness().run_test() as pilot:
        ia = pilot.app.query_one(InputArea)
        await pilot.press("h", "i")
        assert ia.text == "hi"
        await pilot.press("enter")
        assert ia.text == ""  # 提交后清空
        assert _Harness.submitted == ["hi"]


@pytest.mark.asyncio
async def test_backslash_enter_continues():
    _Harness.submitted = []
    async with _Harness().run_test() as pilot:
        ia = pilot.app.query_one(InputArea)
        await pilot.press("a", "\\", "enter")  # 续行：不提交
        assert "\n" in ia.text
        assert _Harness.submitted == []


@pytest.mark.asyncio
async def test_height_grows_with_lines():
    async with _Harness().run_test() as pilot:
        ia = pilot.app.query_one(InputArea)
        await pilot.press("a", "\\", "enter", "b", "\\", "enter", "c")
        await pilot.pause()
        assert ia.size.height >= 3  # 至少 3 行（随行数伸缩）
