# tests/test_cli_worktree.py
import shutil
from pathlib import Path

import pytest

from birdcode.cli.app import (
    _apply_worktree_env,
    _build_session_ctx,
    _cleanup_worktree,
    _resolve_project_root,
)

_HAS_GIT = shutil.which("git") is not None


def test_resolve_project_root_in_worktree_returns_main(tmp_path: Path) -> None:
    """worktree 里 → resolve_main_repo 返主仓(D7/D10)。"""
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt-a").mkdir(parents=True)
    wt = tmp_path / "wt-a"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt-a", encoding="utf-8")
    assert _resolve_project_root(wt) == main.resolve()


def test_resolve_project_root_normal_repo_uses_find_project_root(tmp_path: Path) -> None:
    """非 worktree → find_project_root(走 .git 标记)。"""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert _resolve_project_root(tmp_path) == tmp_path.resolve()


def test_apply_worktree_env_sets_venv_vars(tmp_path: Path, monkeypatch) -> None:
    """worktree → 设 UV_PROJECT_ENVIRONMENT=<main>/.venv + UV_NO_SYNC=1(D4)。"""
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt-a").mkdir(parents=True)
    wt = tmp_path / "wt-a"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt-a", encoding="utf-8")
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.delenv("UV_NO_SYNC", raising=False)
    _apply_worktree_env(wt)
    import os

    assert os.environ["UV_PROJECT_ENVIRONMENT"] == str(main.resolve() / ".venv")
    assert os.environ["UV_NO_SYNC"] == "1"


def test_apply_worktree_env_noop_outside_worktree(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.delenv("UV_NO_SYNC", raising=False)
    _apply_worktree_env(tmp_path)
    import os

    assert "UV_PROJECT_ENVIRONMENT" not in os.environ
    assert "UV_NO_SYNC" not in os.environ


def test_build_session_ctx_cwd_none_uses_project_root(tmp_path: Path) -> None:
    """非 worktree(含子目录启动)→ 传 cwd=None,SessionContext.cwd 退回 project_root。

    锁定向后兼容:run_tui 仅在 worktree 时传 cwd,否则传 None,使会话存储键
    (encode_cwd(cwd))与改造前一致——不论从 repo 哪个子目录启动都共享 project_root 的会话,
    不会因 `cd repo/src && birdcode` 把 cwd 钉成 repo/src 而碎片化。
    """
    project_root = tmp_path / "repo"
    project_root.mkdir()
    # cwd=None → 默认路径:SessionContext.cwd == project_root
    ctx = _build_session_ctx("sid", project_root, cwd=None)
    assert ctx.cwd == str(project_root)
    # 对照:若误传子目录(subdir),cwd 会偏离 project_root——正是本修复规避的情形。
    subdir = project_root / "src"
    subdir.mkdir()
    assert _build_session_ctx("sid", project_root, cwd=subdir).cwd == str(subdir)


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_cleanup_removes_clean_worktree(tmp_path: Path) -> None:
    """干净 + 非交互 → 自动删(remove_worktree 走非 force 路径)。"""
    import subprocess

    main = tmp_path / "main"
    main.mkdir()
    for c in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=main, check=True, capture_output=True)
    (main / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=main, check=True, capture_output=True)
    from birdcode.utils.worktree import create_worktree

    wt = await create_worktree(main, "rm-clean", base_ref="HEAD")
    await _cleanup_worktree(main, wt, interactive=False)
    assert not wt.exists()


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_cleanup_keeps_dirty_when_noninteractive(tmp_path: Path) -> None:
    """脏 + 非交互 → 保留(不毁活)。"""
    import subprocess

    main = tmp_path / "main"
    main.mkdir()
    for c in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=main, check=True, capture_output=True)
    (main / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=main, check=True, capture_output=True)
    from birdcode.utils.worktree import create_worktree

    wt = await create_worktree(main, "dirty-wt", base_ref="HEAD")
    (wt / "untracked.txt").write_text("u", encoding="utf-8")  # 脏
    await _cleanup_worktree(main, wt, interactive=False)
    assert wt.exists()  # 保留


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_cleanup_keeps_branch(tmp_path: Path) -> None:
    """F2 正确修法:干净退出删目录、**留分支**(commit 不丢,re-add 时 checkout 复用)。"""
    import subprocess

    main = tmp_path / "main"
    main.mkdir()
    for c in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=main, check=True, capture_output=True)
    (main / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=main, check=True, capture_output=True)
    from birdcode.utils.worktree import create_worktree

    wt = await create_worktree(main, "br-keep", base_ref="HEAD")
    await _cleanup_worktree(main, wt, interactive=False)
    assert not wt.exists()  # F1:目录删了(cwd 已还原离 wt)
    out2 = subprocess.run(
        ["git", "branch", "--list", "worktree-br-keep"],
        cwd=main, capture_output=True, text=True,
    )
    assert "worktree-br-keep" in out2.stdout  # F2 proper:分支保留(不删,commit 不丢)


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_create_worktree_reuse_preserves_commit(tmp_path: Path) -> None:
    """F2 端到端:建→commit→清理(留分支)→再建同名→checkout 复用,commit 内容回来。"""
    import subprocess

    main = tmp_path / "main"
    main.mkdir()
    for c in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=main, check=True, capture_output=True)
    (main / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=main, check=True, capture_output=True)
    from birdcode.utils.worktree import create_worktree

    wt1 = await create_worktree(main, "reuse", base_ref="HEAD")
    (wt1 / "new.txt").write_text("y", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=wt1, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "work"], cwd=wt1, check=True, capture_output=True)
    await _cleanup_worktree(main, wt1, interactive=False)  # 删目录,留分支
    assert not wt1.exists()
    # 分支仍带 commit
    out = subprocess.run(
        ["git", "log", "--oneline", "worktree-reuse"],
        cwd=main, capture_output=True, text=True,
    )
    assert "work" in out.stdout
    # 再建同名 → checkout 复用,commit 内容回来
    wt2 = await create_worktree(main, "reuse", base_ref="HEAD")
    assert (wt2 / "new.txt").exists()
    assert (wt2 / "new.txt").read_text(encoding="utf-8") == "y"
    await _cleanup_worktree(main, wt2, interactive=False)


def test_default_config_paths_uses_project_root(tmp_path: Path) -> None:
    """#16:传 project_root(主仓)→ 项目层锚主仓,不靠 find_project_root()。"""
    from birdcode.config.loader import _default_config_paths

    main = tmp_path / "main"
    paths = _default_config_paths(project_root=main)
    assert paths[0] == Path.home() / ".birdcode" / "config.yaml"
    assert paths[1] == main / ".birdcode" / "config.yaml"


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_cleanup_restores_cwd_before_remove(tmp_path: Path) -> None:
    """F1:进程 cwd 在 wt 里时,_cleanup_worktree 须先 chdir(main) 再 remove。

    否则 Windows 占用目录锁 → `git worktree remove` 失败(回归:831d45e 每次干净退出都
    RuntimeError)。本测试主动 chdir 进 wt 模拟该场景,验证还原后能删干净。
    """
    import os
    import subprocess

    main = tmp_path / "main"
    main.mkdir()
    for c in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=main, check=True, capture_output=True)
    (main / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=main, check=True, capture_output=True)
    from birdcode.utils.worktree import create_worktree

    wt = await create_worktree(main, "cwd-lock", base_ref="HEAD")
    orig = Path.cwd()
    try:
        os.chdir(wt)  # 模拟进程 cwd 占用 wt(Windows 锁场景)
        await _cleanup_worktree(main, wt, interactive=False)
        assert not wt.exists()  # F1:chdir(main) 后 remove 成功
    finally:
        os.chdir(orig)  # 复原,免污染后续测试


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
def test_build_session_ctx_git_branch_uses_cwd_not_project_root(tmp_path: Path) -> None:
    """worktree 场景:cwd=worktree → git_branch 读 worktree 分支,非主仓 main 分支。

    回归:git rev-parse 的 cwd 曾误用 project_root(=主仓),导致 worktree 会话的
    jsonl 记录 git_branch=main 而非 worktree-<name>。修复后 rev-parse 与 SessionContext.cwd
    同参(cwd),worktree → 读 worktree 分支。
    """
    import asyncio
    import subprocess

    main = tmp_path / "main"
    main.mkdir()
    for c in (
        ["git", "init", "-b", "main"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=main, check=True, capture_output=True)
    (main / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=main, check=True, capture_output=True)
    from birdcode.utils.worktree import create_worktree

    wt = asyncio.run(create_worktree(main, "test", base_ref="HEAD"))
    # 主仓分支 = main;worktree 分支 = worktree-test
    ctx = _build_session_ctx("sid", main, cwd=wt)
    assert ctx.git_branch == "worktree-test"
    assert ctx.git_branch != "main"


def test_resolve_session_id_worktree_resume_routing(tmp_path: Path) -> None:
    """F10:--resume <sid> + worktree_name → 只在 worktree/<name>/ 找;主仓找不到。"""
    from birdcode.cli.app import ProfileError, _resolve_session_id
    from birdcode.session import paths

    root, proj = tmp_path, tmp_path
    foo_jf = paths.session_jsonl(root, "s1", proj, worktree_name="foo")
    foo_jf.parent.mkdir(parents=True, exist_ok=True)
    foo_jf.write_text("{}\n", encoding="utf-8")
    # worktree_name="foo" → 找到
    sid, load = _resolve_session_id(
        continue_last=False, resume="s1", project_root=proj, root=root, worktree_name="foo"
    )
    assert sid == "s1" and load is True
    # worktree_name=None(主仓)→ 找不到 → ProfileError(堵 -c 错位)
    with pytest.raises(ProfileError):
        _resolve_session_id(
            continue_last=False, resume="s1", project_root=proj, root=root, worktree_name=None
        )


def test_list_sessions_worktree_isolation(tmp_path: Path) -> None:
    """F10:list_sessions(worktree_name=...) 只列 worktree/<name>/;默认只列主仓。"""
    from birdcode.session import paths
    from birdcode.session.store import SessionStore

    root, proj = tmp_path, tmp_path
    for sid, wt in [("s-main", None), ("s-foo", "foo")]:
        jf = paths.session_jsonl(root, sid, proj, worktree_name=wt)
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text("{}\n", encoding="utf-8")  # 非空(list_sessions 跳过空)
        jf.with_suffix(".meta").write_text('{"first_user":"x","turn_count":1}', encoding="utf-8")
    assert {s.session_id for s in SessionStore.list_sessions(root, proj)} == {"s-main"}
    foo_ids = {s.session_id for s in SessionStore.list_sessions(root, proj, worktree_name="foo")}
    assert foo_ids == {"s-foo"}


def test_session_store_derives_worktree_name(tmp_path: Path) -> None:
    """F10:SessionStore 从 ctx.cwd 推 worktree_name(worktree ctx → 'foo';主仓 → None)。"""
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    main = tmp_path / "main"
    (main / ".git").mkdir(parents=True)
    wt = tmp_path / "foo"
    (main / ".git" / "worktrees" / "foo").mkdir(parents=True)
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/foo", encoding="utf-8")
    s_wt = SessionStore(
        SessionContext(session_id="s", cwd=str(wt), version="0.1.0", git_branch="worktree-foo"),
        tmp_path, root=tmp_path,
    )
    assert s_wt.worktree_name == "foo"
    assert "worktree" in s_wt.jsonl_path.parts and "foo" in s_wt.jsonl_path.parts
    s_main = SessionStore(
        SessionContext(session_id="s", cwd=str(main), version="0.1.0", git_branch="main"),
        tmp_path, root=tmp_path,
    )
    assert s_main.worktree_name is None
    assert "worktree" not in s_main.jsonl_path.parts
