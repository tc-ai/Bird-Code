"""对真实仓库(BirdCode 自身)跑只读工具的集成验收 —— 非 tmp 合成 fixture。"""

import shutil
from pathlib import Path

import pytest

from birdcode.tools.glob_tool import GlobTool
from birdcode.tools.grep_tool import GrepTool
from birdcode.tools.read_tool import ReadTool

_REPO_ROOT = Path(__file__).resolve().parents[1]  # 仓库根(含 src/ tests/)
_HAS_RG = shutil.which("rg") is not None


async def test_glob_finds_python_sources() -> None:
    out = await GlobTool().execute(pattern="src/**/*.py", path=str(_REPO_ROOT))
    files = [ln for ln in out.splitlines() if ln.strip()]
    assert any("tools" in f for f in files)
    assert any("glob_tool.py" in f for f in files)


@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
async def test_grep_finds_globtool_class() -> None:
    out = await GrepTool().execute(
        pattern="class GlobTool",
        path=str(_REPO_ROOT / "src"),
        output_mode="files_with_matches",
    )
    assert "glob_tool.py" in out


async def test_read_reads_base_tool() -> None:
    out = await ReadTool().execute(
        file_path=str(_REPO_ROOT / "src" / "birdcode" / "tools" / "base.py")
    )
    assert "class Tool" in out
    assert "class ToolResult" in out


@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
async def test_grep_count_mode_on_realrepo() -> None:
    out = await GrepTool().execute(
        pattern="async def",
        path=str(_REPO_ROOT / "src" / "birdcode" / "tools"),
        output_mode="count",
    )
    assert ":" in out  # file:count 行
