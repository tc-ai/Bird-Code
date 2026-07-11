# tests/test_edit_diff_ui.py
import pytest
from rich.syntax import Syntax

from birdcode.ui.icons import select_icons
from birdcode.ui.widgets.tool_line import ToolLine


def test_set_diff_produces_syntax_with_addition_and_deletion():
    """set_diff 后,diff 子区应含 Syntax 渲染对象,+/- 行被 diff lexer 染色。"""
    syntax = ToolLine._build_diff_syntax("old line\n", "new line\n", "f.txt")
    assert isinstance(syntax, Syntax)
    assert syntax.code.startswith("---")  # unified diff header
    assert "old line" in syntax.code and "new line" in syntax.code


def test_set_diff_new_file_shows_all_additions():
    """新文件(old="")→ diff 全是 + 行。"""
    syntax = ToolLine._build_diff_syntax("", "a\nb\n", "new.txt")
    assert "-a" not in syntax.code and "-b" not in syntax.code
    assert "+a" in syntax.code and "+b" in syntax.code


def test_set_diff_no_change_returns_empty_or_minimal():
    """old==new → 无 hunk(unified_diff 空或仅 header)。"""
    syntax = ToolLine._build_diff_syntax("x\n", "x\n", "f.txt")
    # unified_diff 在无改动时返回空(或仅 header,取决于 fromfile/tofile)
    assert "@@" not in syntax.code  # 无 hunk header


@pytest.mark.asyncio
async def test_set_diff_attaches_diff_to_render_group():
    """set_diff 后 diff 并入 _render_display 的 Group(不再 mount 子区)。

    回归:set_diff 曾用 self.mount(diff-region) 挂子区到 Static,Write/Edit 成功后
    工具行「一闪而过」。改为持有 _diff_syntax 进 Group 渲染。
    """
    from rich.console import Group
    from textual.app import App

    class _Host(App[None]):
        pass

    host = _Host()
    async with host.run_test() as pilot:
        line = ToolLine(select_icons(unicode_supported=True))
        await host.mount(line)
        line.start("write_file", '{"file_path": "tmp.txt"}')
        line.finish(True, "tmp.txt")
        # 真实流程:ToolResult(set_output 确认消息)先于 EditDiff(set_diff);默认收起,展开才看 diff
        line.set_output("已写入 tmp.txt")
        line.toggle()  # 展开(默认收起)
        line.set_diff("", "new content\n")
        await pilot.pause()
        assert line._diff_syntax is not None
        content = line._render_display()
        assert isinstance(content, Group)
        assert line._diff_syntax in content.renderables  # diff 进了渲染 Group
        # 不再 mount diff-region 子区(Static 非容器,曾致渲染不稳)
        assert list(line.query(".diff-region")) == []
