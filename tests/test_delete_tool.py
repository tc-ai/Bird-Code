# tests/test_delete_tool.py
from pathlib import Path

import pytest

from birdcode.tools.delete_tool import DeleteTool


@pytest.mark.asyncio
async def test_delete_removes_existing_file(tmp_path):
    f = tmp_path / "trash.txt"
    f.write_text("x", encoding="utf-8")
    out = await DeleteTool().execute(file_path=str(f))
    assert not f.exists()
    assert "已删除" in out


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_error_not_raise(tmp_path):
    """文件不存在 → 友好提示词,不抛异常(让模型自纠,不崩循环)。"""
    f = tmp_path / "nope.txt"
    out = await DeleteTool().execute(file_path=str(f))
    assert "不存在" in out
    assert "<error>" in out
    assert "<hint>" in out


@pytest.mark.asyncio
async def test_delete_directory_returns_error_with_bash_hint(tmp_path):
    """delete_file 仅删文件;目录 → 报错并引导用 bash rm -r。"""
    d = tmp_path / "subdir"
    d.mkdir()
    out = await DeleteTool().execute(file_path=str(d))
    assert d.exists()  # 目录未被删
    assert "目录" in out
    assert "rm -r" in out


@pytest.mark.asyncio
async def test_delete_permission_error_gives_hint(tmp_path, monkeypatch):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")

    def _raise(*_a, **_kw):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "unlink", _raise)
    out = await DeleteTool().execute(file_path=str(f))
    assert "拒绝" in out or "权限" in out
    assert "<hint>" in out


def test_delete_tool_metadata():
    t = DeleteTool()
    assert t.name == "delete_file"
    assert t.kind == "write"
    assert t.parallel_safe is False
    assert "删除" in t.description


def test_default_registry_registers_delete():
    from birdcode.tools.executor import default_registry

    r = default_registry()
    t = r.get("delete_file")
    assert t is not None
    assert t.kind == "write"
    assert t.parallel_safe is False


@pytest.mark.asyncio
async def test_executor_runs_delete_serially(tmp_path):
    """delete_file 是 parallel_safe=False → Executor 串行执行(经 write 权限门,
    但 ToolExecutor 未配 gate 时跳过)。"""
    from birdcode.blocks import ToolUseBlock
    from birdcode.tools.executor import ToolExecutor, default_registry

    f = tmp_path / "victim.txt"
    f.write_text("x", encoding="utf-8")
    ex = ToolExecutor(default_registry())
    res = await ex.execute_batch(
        [ToolUseBlock(id="d1", name="delete_file", input={"file_path": str(f)})]
    )
    assert len(res) == 1
    assert res[0].ok is True
    assert not f.exists()
