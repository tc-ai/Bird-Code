# tests/test_edit_tool.py
import pytest

from birdcode.tools.edit_tool import EditTool


@pytest.mark.asyncio
async def test_edit_unique_replace(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    out = await EditTool().execute(file_path=str(f), old_string="return 1", new_string="return 2")
    assert f.read_text(encoding="utf-8") == "def foo():\n    return 2\n"
    assert "已编辑" in out


@pytest.mark.asyncio
async def test_edit_zero_match_gives_self_correct_hint(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    out = await EditTool().execute(
        file_path=str(f),
        old_string="return 99",
        new_string="return 2",  # 不存在
    )
    assert "0 匹配" in out
    assert "<hint>" in out
    assert "缩进" in out or "Read" in out  # §3.5 引导核对缩进/用 Read


@pytest.mark.asyncio
async def test_edit_multiple_matches_gives_hint(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\nx = 1\n", encoding="utf-8")
    out = await EditTool().execute(
        file_path=str(f),
        old_string="x = 1",
        new_string="x = 2",  # 匹配 2 次
    )
    assert "2" in out and "唯一" in out
    assert "<hint>" in out
    assert "replace_all" in out or "上下文" in out  # §3.5 引导扩大上下文或 replace_all


@pytest.mark.asyncio
async def test_edit_replace_all(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("TODO\nTODO\nkeep\n", encoding="utf-8")
    out = await EditTool().execute(
        file_path=str(f),
        old_string="TODO",
        new_string="DONE",
        replace_all=True,
    )
    assert f.read_text(encoding="utf-8") == "DONE\nDONE\nkeep\n"
    assert "已编辑" in out


@pytest.mark.asyncio
async def test_edit_file_not_found(tmp_path):
    out = await EditTool().execute(
        file_path=str(tmp_path / "nope.py"), old_string="a", new_string="b"
    )
    assert "不存在" in out or "找不到" in out


def test_edit_tool_metadata():
    t = EditTool()
    assert t.name == "edit_file"
    assert t.kind == "write"
    assert t.parallel_safe is False
    d = t.description
    assert "替换" in d or "Search/Replace" in d  # 核心作用
