# tests/test_write_tool.py
from pathlib import Path

import pytest

from birdcode.tools.write_tool import WriteTool


@pytest.mark.asyncio
async def test_write_creates_new_file(tmp_path):
    f = tmp_path / "new.py"
    out = await WriteTool().execute(file_path=str(f), content="print('hi')\n")
    assert f.read_text(encoding="utf-8") == "print('hi')\n"
    assert "成功" not in out or "已写入" in out  # 返回「已写入」
    assert "1" in out  # 行数提示


@pytest.mark.asyncio
async def test_write_overwrites_existing_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("OLD", encoding="utf-8")
    await WriteTool().execute(file_path=str(f), content="NEW")
    assert f.read_text(encoding="utf-8") == "NEW"  # 覆盖,非追加


@pytest.mark.asyncio
async def test_write_missing_parent_dir_gives_hint(tmp_path):
    f = tmp_path / "no_such_dir" / "f.txt"
    out = await WriteTool().execute(file_path=str(f), content="x")
    assert "父目录不存在" in out
    assert "<hint>" in out
    assert "mkdir" in out  # §3.4 提示词


@pytest.mark.asyncio
async def test_write_permission_error_gives_hint(tmp_path, monkeypatch):
    """模拟权限拒绝 → §3.4 用户拒绝/权限提示词。"""
    f = tmp_path / "f.txt"

    def _raise(*_a, **_kw):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "write_text", _raise)
    out = await WriteTool().execute(file_path=str(f), content="x")
    assert "拒绝" in out or "权限" in out
    assert "<hint>" in out


def test_write_tool_metadata():
    t = WriteTool()
    assert t.name == "write_file"
    assert t.kind == "write"
    assert t.parallel_safe is False
    d = t.description
    assert "覆盖" in d or "写入" in d  # 核心作用
    assert "新建" in d or "重写" in d  # 用途
