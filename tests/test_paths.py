# tests/test_paths.py
"""项目根发现:标记向上查找(.git → pyproject.toml)。

注:.birdcode 不作标记(与 ~/.birdcode 用户配置目录冲突),单独存在不锚定。
"""

from pathlib import Path

from birdcode.utils.paths import find_project_root


def test_finds_git_marker(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "src"
    sub.mkdir()
    assert find_project_root(sub) == tmp_path.resolve()


def test_finds_pyproject_marker(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    assert find_project_root(sub) == tmp_path.resolve()


def test_birdcode_dir_is_not_a_marker(tmp_path: Path):
    """`.birdcode` 不作项目标记(与 ~/.birdcode 冲突):单独存在不锚定,回退到 start。"""
    (tmp_path / ".birdcode").mkdir()
    sub = tmp_path / "a"
    sub.mkdir()
    # 只有 .birdcode、无 .git/pyproject → 回退到 start,不把 tmp_path 当根
    assert find_project_root(sub) == sub.resolve()


def test_priority_git_over_pyproject_at_same_level(tmp_path: Path):
    """同级 .git + pyproject.toml → .git 优先(标记优先级)。"""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    assert find_project_root(tmp_path) == tmp_path.resolve()


def test_nearest_marker_wins(tmp_path: Path):
    """外层 .git、内层也 .git → 内层(最近的标记)胜。"""
    (tmp_path / ".git").mkdir()
    inner = tmp_path / "sub"
    inner.mkdir()
    (inner / ".git").mkdir()
    leaf = inner / "deep"
    leaf.mkdir()
    assert find_project_root(leaf) == inner.resolve()
