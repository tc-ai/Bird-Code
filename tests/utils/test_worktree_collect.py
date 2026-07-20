# tests/utils/test_worktree_collect.py
from __future__ import annotations

from pathlib import Path

import pytest

from birdcode.utils import worktree as wt_mod
from birdcode.utils.worktree import collect_worktree_changes, current_head_sha


async def _fake_git(outputs: dict[str, tuple[int, str, str]]):
    """按 args 关键字匹配,返回预设 (rc,out,err)。未命中→(128,"","")。"""

    async def _fake(args: list[str], *, cwd: Path | None = None, timeout: float = 60.0):
        key = " ".join(args)
        for k, v in outputs.items():
            if k in key:
                return v
        return (128, "", "no mock")

    return _fake


@pytest.mark.asyncio
async def test_committed_true_when_rev_list_positive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wt_mod,
        "_run_git",
        await _fake_git(
            {
                "rev-list --count": (0, "3\n", ""),
                "diff --name-only": (0, "a.py\nb.py\n", ""),
                "status --porcelain": (0, "", ""),
            }
        ),
    )
    artifacts, committed = await collect_worktree_changes(tmp_path, "base123")
    assert committed is True
    assert artifacts == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_committed_false_when_rev_list_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wt_mod,
        "_run_git",
        await _fake_git(
            {
                "rev-list --count": (0, "0\n", ""),
                "diff --name-only": (0, "", ""),
                "status --porcelain": (0, "?? sort.py\n", ""),
            }
        ),
    )
    artifacts, committed = await collect_worktree_changes(tmp_path, "base123")
    assert committed is False
    assert artifacts == ["sort.py"]


@pytest.mark.asyncio
async def test_union_dedup_committed_and_uncommitted(tmp_path, monkeypatch):
    """已提交 a.py + 未提交 a.py(M)与 b.py(??)→ 去重 {a.py, b.py}。"""
    monkeypatch.setattr(
        wt_mod,
        "_run_git",
        await _fake_git(
            {
                "rev-list --count": (0, "1\n", ""),
                "diff --name-only": (0, "a.py\n", ""),
                "status --porcelain": (0, " M a.py\n?? b.py\n", ""),
            }
        ),
    )
    artifacts, _ = await collect_worktree_changes(tmp_path, "base123")
    assert sorted(artifacts) == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_porcelain_rename_takes_dest(tmp_path, monkeypatch):
    """rename `R  old -> new` → 取 new。"""
    monkeypatch.setattr(
        wt_mod,
        "_run_git",
        await _fake_git(
            {
                "rev-list --count": (0, "0\n", ""),
                "diff --name-only": (0, "", ""),
                "status --porcelain": (0, "R  old.py -> new.py\n", ""),
            }
        ),
    )
    artifacts, _ = await collect_worktree_changes(tmp_path, "base123")
    assert artifacts == ["new.py"]


@pytest.mark.asyncio
async def test_git_failure_degrades_to_empty(tmp_path, monkeypatch):
    """rev-list rc!=0 → ([], False),不抛。"""
    monkeypatch.setattr(
        wt_mod,
        "_run_git",
        await _fake_git(
            {
                "rev-list --count": (128, "", "git error"),
            }
        ),
    )
    artifacts, committed = await collect_worktree_changes(tmp_path, "base123")
    assert artifacts == []
    assert committed is False


@pytest.mark.asyncio
async def test_truncate_over_max_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wt_mod,
        "_run_git",
        await _fake_git(
            {
                "rev-list --count": (0, "0\n", ""),
                "diff --name-only": (0, "", ""),
                "status --porcelain": (0, "".join(f"?? f{i}.py\n" for i in range(25)), ""),
            }
        ),
    )
    artifacts, _ = await collect_worktree_changes(tmp_path, "base123", max_files=20)
    assert len(artifacts) == 21  # 20 个文件 + 1 个截断标记行
    assert "截断" in artifacts[-1]


@pytest.mark.asyncio
async def test_nonexistent_wt_returns_empty(tmp_path, monkeypatch):
    artifacts, committed = await collect_worktree_changes(tmp_path / "nope", "base123")
    assert artifacts == []
    assert committed is False


@pytest.mark.asyncio
async def test_current_head_sha_returns_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wt_mod,
        "_run_git",
        await _fake_git(
            {
                "rev-parse HEAD": (0, "abc1234\n", ""),
            }
        ),
    )
    assert await current_head_sha(tmp_path) == "abc1234"


@pytest.mark.asyncio
async def test_current_head_sha_none_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(wt_mod, "_run_git", await _fake_git({}))
    assert await current_head_sha(tmp_path) is None
