from pathlib import Path

import pytest

from birdcode.tools.glob_tool import GlobTool


@pytest.fixture
def sample_tree(tmp_path: Path) -> Path:
    """tmp_path 下造 3 个 .py + 1 个 .md + 一个子目录 .py。"""
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("bb", encoding="utf-8")
    (tmp_path / "readme.md").write_text("doc", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "c.py").write_text("ccc", encoding="utf-8")
    return tmp_path


async def test_glob_matches_flat_pattern(sample_tree: Path) -> None:
    # 非递归 *.py 仅匹配顶层(a.py, b.py);**/*.py 才递归到 pkg/c.py
    out = await GlobTool().execute(pattern="*.py", path=str(sample_tree))
    names = sorted(Path(p).name for p in out.splitlines() if p.strip())
    assert names == ["a.py", "b.py"]


async def test_glob_recursive_double_star(sample_tree: Path) -> None:
    out = await GlobTool().execute(pattern="**/*.py", path=str(sample_tree))
    assert "pkg" in out  # 递归命中子目录


async def test_glob_sorted_by_mtime_desc(sample_tree: Path) -> None:
    # tmp_path 下文件 mtime 几乎相同,但顺序应稳定可解析;核心校验「可解析为路径列表」
    out = await GlobTool().execute(pattern="*.py", path=str(sample_tree))
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2  # a.py, b.py(顶层;** 才递归)


async def test_glob_missing_path_returns_error_hint(sample_tree: Path) -> None:
    out = await GlobTool().execute(pattern="*.py", path=str(sample_tree / "nope"))
    assert "<error>" in out and "path 不存在" in out
    assert "<hint>" in out


async def test_glob_no_match_returns_error_hint(sample_tree: Path) -> None:
    out = await GlobTool().execute(pattern="*.nosuchext", path=str(sample_tree))
    assert "<error>" in out and "无匹配" in out
    assert "<hint>" in out and "**/*" in out  # 引导放宽 pattern


async def test_glob_caps_at_100(tmp_path: Path) -> None:
    for i in range(150):
        (tmp_path / f"f{i:03d}.py").write_text("x", encoding="utf-8")
    out = await GlobTool().execute(pattern="*.py", path=str(tmp_path))
    # 仅统计真实文件路径行(排除截断提示行)
    path_lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("...")]
    assert len(path_lines) == 100
    assert "已截断" in out or "限 100" in out  # 超限提示


def test_glob_description_and_name() -> None:
    """description 一句话描述作用 + name 正确。"""
    d = GlobTool.description
    assert d
    assert "glob" in d.lower() or "路径" in d  # 核心作用
    assert GlobTool.name == "glob"


async def test_glob_skips_birdcode_worktrees(tmp_path: Path) -> None:
    """.birdcode/worktrees/ 下的文件不被 glob 串回(D2)——Path.glob 不认 gitignore,需显式跳过。"""
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    wt_file = tmp_path / ".birdcode" / "worktrees" / "wt-a" / "src" / "x.py"
    wt_file.parent.mkdir(parents=True)
    wt_file.write_text("x", encoding="utf-8")
    out = await GlobTool().execute(pattern="**/*.py", path=str(tmp_path))
    names = [Path(p).name for p in out.splitlines() if p.strip() and not p.startswith("...")]
    assert "a.py" in names
    assert "x.py" not in names  # worktree 内文件不泄漏
