# tests/test_separator.py
import pytest
from textual.app import App, ComposeResult
from textual.strip import Strip

from birdcode.ui.widgets.separator import Separator, separator_segments


def test_separator_segments_fill_width():
    segs = separator_segments(width=5, char="-")
    s = Strip(segs)
    assert s.cell_length == 5


@pytest.mark.asyncio
async def test_separator_renders_full_width():
    class _App(App):
        def compose(self) -> ComposeResult:
            yield Separator(char="-")

    async with _App().run_test(size=(20, 3)) as pilot:
        await pilot.pause()
        sep = pilot.app.query_one(Separator)
        line = sep.render_line(0)
        assert line.cell_length == 20
