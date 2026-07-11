"""记忆索引启动加载测试:两级优先级、缺失跳过、200 行/25KB 上限、空。"""
from __future__ import annotations

from pathlib import Path

import pytest

from birdcode.memory import loader as loader_mod
from birdcode.memory.loader import load_memory_index
from birdcode.memory.store import rebuild_index, write_memory


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", lambda: h)
    return h


def _seed(dir_: Path, name, desc, type_):
    write_memory(dir_, name=name, description=desc, type_=type_, body="b")
    rebuild_index(dir_)


def test_no_memory_returns_empty(tmp_path, home):
    assert load_memory_index(tmp_path) == ""


def test_loads_project_level(tmp_path, home):
    _seed(tmp_path / ".birdcode" / "memory", "ci", "用 GitHub Actions", "project")
    out = load_memory_index(tmp_path)
    assert "用 GitHub Actions" in out
    assert out.startswith("记忆（")


def test_loads_user_level(tmp_path, home):
    _seed(home / ".birdcode" / "memory", "notab", "不要用 tab", "user")
    out = load_memory_index(tmp_path)
    assert "不要用 tab" in out


def test_project_level_before_user_level(tmp_path, home):
    _seed(tmp_path / ".birdcode" / "memory", "proj", "PROJECT_MARKER", "project")
    _seed(home / ".birdcode" / "memory", "usr", "USER_MARKER", "user")
    out = load_memory_index(tmp_path)
    assert out.index("PROJECT_MARKER") < out.index("USER_MARKER")


def test_line_cap_truncates_with_warning(tmp_path, home, monkeypatch):
    monkeypatch.setattr(loader_mod, "MAX_INDEX_LINES", 5)
    d = tmp_path / ".birdcode" / "memory"
    for i in range(20):
        write_memory(d, name=f"m{i:02d}", description=f"条目{i}", type_="project", body="b")
    rebuild_index(d)
    out = load_memory_index(tmp_path)
    assert "已截断" in out
    assert out.count("\n") < 20


def test_byte_cap_truncates_with_warning(tmp_path, home, monkeypatch):
    monkeypatch.setattr(loader_mod, "MAX_INDEX_BYTES", 200)
    d = tmp_path / ".birdcode" / "memory"
    write_memory(d, name="big", description="X" * 500, type_="project", body="b")
    rebuild_index(d)
    out = load_memory_index(tmp_path)
    assert "已截断" in out
    assert len(out.encode("utf-8")) < 600


def test_cache_stability(tmp_path, home):
    _seed(tmp_path / ".birdcode" / "memory", "x", "desc", "project")
    assert load_memory_index(tmp_path) == load_memory_index(tmp_path)


# —— F7:扫 .md 而非读 MEMORY.md(派生物删/损/手改不影响加载)——


def test_load_ignores_deleted_memory_md(tmp_path, home):
    # 删掉派生的 MEMORY.md → 仍从 .md 文件加载
    d = tmp_path / ".birdcode" / "memory"
    _seed(d, "ci", "用 GH Actions", "project")
    (d / "MEMORY.md").unlink()
    out = load_memory_index(tmp_path)
    assert "用 GH Actions" in out


def test_load_picks_up_manually_added_md(tmp_path, home):
    # 手动加 .md(不经提取、无 MEMORY.md)→ 启动加载可见
    d = tmp_path / ".birdcode" / "memory"
    write_memory(d, name="manual", description="手加的", type_="project", body="b")
    out = load_memory_index(tmp_path)
    assert "手加的" in out
