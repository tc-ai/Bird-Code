# tests/test_tool_line.py
import pytest

from birdcode.ui.icons import select_icons
from birdcode.ui.widgets.tool_line import format_tool_running


def test_format_running():
    assert format_tool_running("echo", "hi", "⠋") == "⠋ echo(hi)"


def test_render_result_ok():
    """结果行(无工具名时)形如 '⎿ ✓ done' —— 断言真实渲染路径 _render_text。"""
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl.finish(ok=True, summary="done")
    assert tl._render_text().startswith("  ⎿ ✓ done")


def test_render_result_fail():
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=False))
    tl.finish(ok=False, summary="boom")
    out = tl._render_text()
    assert "x" in out and "boom" in out


def test_render_result_includes_tool_name():
    """_render_text 含工具名,让用户看清是哪个工具的结果(生产改进点)。"""
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl._name = "read"  # start() 起定时器需事件循环,直接置名验证渲染
    tl.finish(True, "test.md")
    text = tl._render_text()
    assert "read" in text and "test.md" in text


# ---- stage2: ToolLine 接收 truncated(为 stage3 折叠 UI 预留) ----


def test_tool_line_finish_accepts_truncated():
    from birdcode.ui.widgets.tool_line import ToolLine

    line = ToolLine(select_icons(unicode_supported=True))
    line.finish(ok=True, summary="ok", truncated=True)
    assert getattr(line, "_truncated", False) is True


def test_tool_line_finish_truncated_defaults_false():
    from birdcode.ui.widgets.tool_line import ToolLine

    line = ToolLine(select_icons(unicode_supported=True))
    line.finish(ok=True, summary="ok")
    assert getattr(line, "_truncated", False) is False


# ---- stage3: ToolLine.set_output 可折叠结果区 ----


def test_set_output_stores_text_and_truncated():
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl.set_output("hello\nworld", truncated=False)
    assert tl._output == "hello\nworld"
    assert tl._truncated is False
    assert tl._has_output is True


def test_set_output_default_collapsed():
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl.set_output("x" * 15000)
    assert tl._expanded is False  # 默认收起


def test_toggle_flips_expanded():
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl.set_output("x" * 15000)  # 默认收起(不论长短)
    assert tl._expanded is False
    tl.toggle()
    assert tl._expanded is True
    tl.toggle()
    assert tl._expanded is False


def test_set_output_short_default_collapsed():
    """工具输出默认收起(不论长短);点击展开才看正文。"""
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl.set_output("line1\nline2")  # 2 行,短
    assert tl._expanded is False  # 默认收起
    text = tl._render_text()
    assert "展开" in text  # 收起态显示展开提示
    assert "line1" not in text  # 正文不直接可见


def test_render_collapsed_shows_expand_hint():
    """长输出收起时显示「展开」提示;展开后正文出现。"""
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    # 不调 start(它起 Textual 定时器,需事件循环);finish + set_output 足以验证折叠渲染
    tl.finish(True, "✓ 大文件")
    long_body = "\n".join(f"line{i}" for i in range(100))  # 100 行,默认收起
    tl.set_output(long_body, truncated=True)
    assert tl._expanded is False
    collapsed = tl._render_text()
    assert "展开" in collapsed
    assert "已截断" in collapsed  # 截断标记透传
    # 收起态不应露出完整正文
    tl.toggle()
    expanded = tl._render_text()
    assert "line0" in expanded


def test_render_no_output_just_head():
    """无 set_output 时,_render_text 退化为普通结果行(向后兼容)。"""
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl.finish(True, "ok")
    text = tl._render_text()
    assert "ok" in text
    assert "展开" not in text


def test_set_output_always_collapsed_regardless_of_size():
    """工具输出默认收起,不论行数/字符数(改自原「短输出自动展开」)。"""
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    body = "\n".join(f"line {i}: " + "x" * 90 for i in range(34))  # 34 行 ~3400 字符
    tl.set_output(body)
    assert tl._expanded is False  # 默认收起
    assert "展开" in tl._render_text()


# ---- markup=False:输出含 [TextBlock(..)] 方括号时不 MarkupError 崩溃(df3e574 回归) ----


@pytest.mark.asyncio
async def test_set_output_with_brackets_renders_without_markup_error():
    """widget markup=False:含 [TextBlock(..)] 的正文渲染期不抛 MarkupError。

    回归 df3e574 前的崩溃 —— 点击展开触发 re-render,[TextBlock] 被当 Rich tag 解析。
    需 run_test 触发真实渲染(无 app 上下文时 MarkupError 不会在 update 时抛出)。
    """
    from textual.app import App, ComposeResult

    from birdcode.ui.widgets.tool_line import ToolLine

    class _LineApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ToolLine(select_icons(unicode_supported=True))

    async with _LineApp().run_test() as pilot:
        line = pilot.app.query_one(ToolLine)
        line.finish(True, "test.md")
        line.set_output("[TextBlock(text='x')] body")  # 默认收起
        line.toggle()  # 展开看正文(含方括号)
        await pilot.pause()  # 触发真实渲染;markup=False 下不解析 [TextBlock]
        assert "[TextBlock" in line._render_text()  # 方括号被当字面量保留


# ---- Read 代码语法高亮:按 file_path 扩展名用 Rich Syntax 着色 ----


def test_guess_language_read_python():
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl._name = "read_file"
    tl._args = '{"file_path": "src/birdcode/conversation.py"}'
    assert tl._guess_language() == "python"


def test_guess_language_unknown_ext_returns_none():
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl._name = "read_file"
    tl._args = '{"file_path": "notes.unknownext"}'
    assert tl._guess_language() is None


def test_guess_language_non_read_returns_none():
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl._name = "grep"
    tl._args = '{"pattern": "x"}'
    assert tl._guess_language() is None


@pytest.mark.asyncio
async def test_read_output_renders_syntax_highlight():
    """Read .py 展开 → _render_content 返回含 Syntax 的 Group(语法高亮),渲染不抛。"""
    from rich.console import Group
    from rich.syntax import Syntax
    from textual.app import App, ComposeResult

    from birdcode.ui.widgets.tool_line import ToolLine

    class _LineApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ToolLine(select_icons(unicode_supported=True))

    async with _LineApp().run_test() as pilot:
        line = pilot.app.query_one(ToolLine)
        line._name = "read_file"
        line._args = '{"file_path": "conversation.py"}'
        line.finish(True, "conversation.py")
        line.set_output("import asyncio\nclass Foo:\n    pass\n")
        line.toggle()  # 展开(默认收起)
        await pilot.pause()  # 触发真实渲染(高亮),不抛即通过
        assert line._language == "python"
        content = line._render_display()
        assert isinstance(content, Group)
        assert any(isinstance(p, Syntax) for p in content.renderables)


@pytest.mark.asyncio
async def test_click_body_keeps_expanded_click_head_collapses():
    """展开后点正文(y≥1)不收起(便于选中代码);点 head 行(y=0)才收起。"""
    from textual.app import App, ComposeResult

    from birdcode.ui.widgets.tool_line import ToolLine

    class _LineApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ToolLine(select_icons(unicode_supported=True))

    async with _LineApp().run_test() as pilot:
        line = pilot.app.query_one(ToolLine)
        line.finish(True, "x.py")
        line.set_output("line0\nline1\nline2\n")
        line.toggle()  # 展开(默认收起)
        await pilot.pause()
        assert line._expanded is True
        # 点正文(y=2)→ 保持展开(不被误触收起)
        await pilot.click(ToolLine, offset=(0, 2))
        await pilot.pause()
        assert line._expanded is True
        # 点 head(y=0)→ 收起
        await pilot.click(ToolLine, offset=(0, 0))
        await pilot.pause()
        assert line._expanded is False


def test_body_render_cached_not_rebuilt_on_toggle():
    """正文渲染对象缓存:toggle 不重建(避免每次展开重跑 pygments lexer 解析大文件)。"""
    from birdcode.ui.widgets.tool_line import ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl._name = "read_file"
    tl._args = '{"file_path": "x.py"}'
    tl.finish(True, "x.py")
    tl.set_output("import os\n")
    first = tl._body_render
    assert first is not None
    tl.toggle()
    tl.toggle()
    assert tl._body_render is first  # 同一对象(缓存,toggle 没重建)


def test_build_body_render_truncates_to_max_lines():
    """展开时正文最多 _MAX_EXPAND_LINES 行,超出截断 + 提示(免撑满终端)。"""
    from rich.syntax import Syntax

    from birdcode.ui.widgets.tool_line import _MAX_EXPAND_LINES, ToolLine

    tl = ToolLine(select_icons(unicode_supported=True))
    tl.set_output("\n".join(f"line{i}" for i in range(30)))  # 30 行
    br = tl._body_render
    code = br.code if isinstance(br, Syntax) else br.plain
    assert f"line{_MAX_EXPAND_LINES - 1}" in code  # 第 15 行(0-14)在
    assert f"line{_MAX_EXPAND_LINES}" not in code  # 第 16 行被截
    assert "共 30 行" in code  # 截断提示
