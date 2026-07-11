import pytest
from textual.app import App, ComposeResult

from birdcode.ui.icons import select_icons
from birdcode.ui.widgets.thinking_block import (
    ThinkingBlock,
    format_thinking_done,
    format_thinking_running,
)


def test_format_running():
    assert format_thinking_running("⠋", 3) == "⠋ Thinking… (3s)"


def test_format_done_unicode():
    ic = select_icons(unicode_supported=True)
    assert format_thinking_done(ic, seconds=5).startswith("✻ Thought for 5s")


def test_format_done_ascii():
    ic = select_icons(unicode_supported=False)
    assert format_thinking_done(ic, seconds=1).startswith("* Thought for 1s")


class _BlockApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ThinkingBlock(select_icons(unicode_supported=True))


@pytest.mark.asyncio
async def test_toggle_flips_expanded():
    async with _BlockApp().run_test() as pilot:
        block = pilot.app.query_one(ThinkingBlock)
        assert block._expanded is False
        block.toggle()
        await pilot.pause()
        assert block._expanded is True
        block.toggle()
        await pilot.pause()
        assert block._expanded is False


@pytest.mark.asyncio
async def test_click_toggles_expanded():
    async with _BlockApp().run_test() as pilot:
        block = pilot.app.query_one(ThinkingBlock)
        assert block._expanded is False
        await pilot.click(ThinkingBlock)
        await pilot.pause()
        assert block._expanded is True


@pytest.mark.asyncio
async def test_thinking_text_with_brackets_renders_without_markup_error():
    """widget markup=False:思考正文含 [TextBlock(..)] 渲染期不抛 MarkupError。

    回归 df3e574:模型 reasoning 可能含 [..](如代码片段),默认 markup 解析会崩溃。
    需 run_test 触发真实渲染(渲染期才抛 MarkupError)。
    """
    async with _BlockApp().run_test() as pilot:
        block = pilot.app.query_one(ThinkingBlock)
        block.write("[TextBlock(text='x')] reasoning")  # 含方括号
        block.toggle()  # 展开 → _refresh 渲染含方括号的正文
        await pilot.pause()  # 触发真实渲染;markup=False 下不解析方括号
        assert "[TextBlock" in "".join(block._text)  # 字面保留


@pytest.mark.asyncio
async def test_click_body_keeps_expanded_click_head_collapses():
    """展开后点正文(y≥1)不收起(便于选中);点 head 行(y=0)才收起。"""
    async with _BlockApp().run_test() as pilot:
        block = pilot.app.query_one(ThinkingBlock)
        block.write("some reasoning line here")
        block.toggle()  # 展开
        await pilot.pause()
        assert block._expanded is True
        # 点正文(第 1 行)→ 保持展开(不被误触收起)
        await pilot.click(ThinkingBlock, offset=(0, 1))
        await pilot.pause()
        assert block._expanded is True
        # 点 head(第 0 行)→ 收起
        await pilot.click(ThinkingBlock, offset=(0, 0))
        await pilot.pause()
        assert block._expanded is False
