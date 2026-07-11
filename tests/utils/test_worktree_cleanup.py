# tests/utils/test_worktree_cleanup.py
"""Task 7 Part A: cleanup_subagent_worktree §6.6 单测。

四场景(真实 git:subprocess init repo + commit):
  1. success+clean+已合并 → 删目录 + branch -d 成功(分支没)。
  2. success+dirty → 留目录 + 留分支(不毁活)。
  3. success+clean+未合并 commit → 删目录 + 留分支(commit 不丢,git branch -d 失败)。
  4. failed → 留目录 + 留分支(回报主 agent 定夺)。

绝不 git branch -D(防丢未合并 commit)。
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from birdcode.utils import worktree as wt_mod

_HAS_GIT = shutil.which("git") is not None


def _init_repo(main: Path) -> None:
    """init 一个 git 仓库 + 一个初始 commit(给 worktree 基线)。"""
    main.mkdir()
    for c in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=main, check=True, capture_output=True)
    (main / "f").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "i"], cwd=main, check=True, capture_output=True)


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_cleanup_success_clean_removes_dir_and_branch_when_merged(tmp_path):
    """success+clean+已合并 → 删目录 + branch -d 成功(分支没)。"""
    main = tmp_path / "main"
    _init_repo(main)
    wt = await wt_mod.create_worktree(main, "sub-abc", base_ref="HEAD")
    await wt_mod.lock_worktree(main, wt, "subagent active")
    # 无改动(clean)、分支基于 HEAD(已含在 HEAD)→ 视为已合并
    await wt_mod.cleanup_subagent_worktree(main, wt, succeeded=True)
    assert not wt.exists()
    out = subprocess.run(
        ["git", "branch", "--list", "worktree-sub-abc"],
        cwd=main, capture_output=True, text=True,
    )
    assert "worktree-sub-abc" not in out.stdout  # branch -d 删了


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_cleanup_success_dirty_keeps_dir_and_branch(tmp_path):
    """success+dirty → 留目录 + 留分支(不毁活)。"""
    main = tmp_path / "main"
    _init_repo(main)
    wt = await wt_mod.create_worktree(main, "sub-dirty", base_ref="HEAD")
    await wt_mod.lock_worktree(main, wt, "subagent active")
    (wt / "untracked.txt").write_text("u", encoding="utf-8")  # 脏
    await wt_mod.cleanup_subagent_worktree(main, wt, succeeded=True)
    assert wt.exists()  # 脏 → 留目录
    out = subprocess.run(
        ["git", "branch", "--list", "worktree-sub-dirty"],
        cwd=main, capture_output=True, text=True,
    )
    assert "worktree-sub-dirty" in out.stdout  # 留分支


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_cleanup_success_unmerged_keeps_branch(tmp_path):
    """success+clean+未合并 commit → 删目录 + 留分支(commit 不丢,git branch -d 失败)。"""
    main = tmp_path / "main"
    _init_repo(main)
    wt = await wt_mod.create_worktree(main, "sub-unmerged", base_ref="HEAD")
    await wt_mod.lock_worktree(main, wt, "subagent active")
    (wt / "new.txt").write_text("y", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=wt, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "work"], cwd=wt, check=True, capture_output=True)
    await wt_mod.cleanup_subagent_worktree(main, wt, succeeded=True)
    assert not wt.exists()  # clean working tree → 删目录
    out = subprocess.run(
        ["git", "log", "--oneline", "worktree-sub-unmerged"],
        cwd=main, capture_output=True, text=True,
    )
    assert "work" in out.stdout  # 分支+commit 在(branch -d 因未合并失败→留)


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_cleanup_failed_keeps_dir_and_branch(tmp_path):
    """failed → 留目录 + 留分支(回报主 agent 定夺)。"""
    main = tmp_path / "main"
    _init_repo(main)
    wt = await wt_mod.create_worktree(main, "sub-fail", base_ref="HEAD")
    await wt_mod.lock_worktree(main, wt, "subagent active")
    await wt_mod.cleanup_subagent_worktree(main, wt, succeeded=False)
    assert wt.exists()
    out = subprocess.run(
        ["git", "branch", "--list", "worktree-sub-fail"],
        cwd=main, capture_output=True, text=True,
    )
    assert "worktree-sub-fail" in out.stdout
