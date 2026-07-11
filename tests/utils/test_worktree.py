# tests/utils/test_worktree.py
import shutil
from pathlib import Path

import pytest

from birdcode.utils import worktree as wt_mod
from birdcode.utils.worktree import is_worktree, resolve_main_repo

_HAS_GIT = shutil.which("git") is not None


def test_is_worktree_false_for_normal_repo(tmp_path: Path) -> None:
    """普通 repo:.git 是目录 → 非 worktree。"""
    (tmp_path / ".git").mkdir()
    assert is_worktree(tmp_path) is False


def test_is_worktree_false_when_no_git(tmp_path: Path) -> None:
    assert is_worktree(tmp_path) is False


def test_is_worktree_true_for_gitdir_file(tmp_path: Path) -> None:
    """worktree:.git 是文件,内容 `gitdir: ...` → 是 worktree。"""
    (tmp_path / ".git").write_text("gitdir: /main/.git/worktrees/wt-a", encoding="utf-8")
    assert is_worktree(tmp_path) is True


def test_is_worktree_false_if_git_file_not_gitdir(tmp_path: Path) -> None:
    (tmp_path / ".git").write_text("something else", encoding="utf-8")
    assert is_worktree(tmp_path) is False


def test_is_worktree_false_for_submodule(tmp_path: Path) -> None:
    """F5:子模块 .git(`gitdir: .../.git/modules/<n>`)不判为 worktree。

    回归:旧版只看 `gitdir:` 开头,子模块也是 gitdir 文件 → 误判 → 错 project_root/sandbox
    + 退出 remove 抛错。
    """
    (tmp_path / ".git").write_text(
        "gitdir: /main/.git/modules/sub", encoding="utf-8"
    )
    assert is_worktree(tmp_path) is False


def test_resolve_main_repo_from_absolute_gitdir(tmp_path: Path) -> None:
    """`gitdir: <main>/.git/worktrees/<name>` → 主仓 = <main>。"""
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt-a").mkdir(parents=True)
    wt = tmp_path / "wt-a"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt-a", encoding="utf-8")
    assert resolve_main_repo(wt) == main.resolve()


def test_is_worktree_true_for_separate_git_dir(tmp_path: Path) -> None:
    """#7:separate-git-dir 的 worktree(gitdir `.../.real.git/worktrees/<n>`)也判为 worktree。"""
    (tmp_path / ".git").write_text(
        "gitdir: /main/.real.git/worktrees/wt-x", encoding="utf-8"
    )
    assert is_worktree(tmp_path) is True


def test_resolve_main_repo_separate_git_dir(tmp_path: Path) -> None:
    """#7:separate-git-dir worktree → resolve 到主仓(找 worktrees 段,非 .git 段)。"""
    main = tmp_path / "main"
    (main / ".real.git" / "worktrees" / "wt-x").mkdir(parents=True)
    wt = tmp_path / "wt-x"
    wt.mkdir()
    (wt / ".git").write_text(
        f"gitdir: {main}/.real.git/worktrees/wt-x", encoding="utf-8"
    )
    assert resolve_main_repo(wt) == main.resolve()


def test_resolve_main_repo_none_for_non_worktree(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert resolve_main_repo(tmp_path) is None


def test_worktree_store_name_returns_dirname(tmp_path: Path) -> None:
    """F10:worktree cwd → 目录名(--worktree foo 的 dir basename=foo)。"""
    from birdcode.utils.worktree import worktree_store_name

    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "foo").mkdir(parents=True)
    wt = tmp_path / "foo"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/foo", encoding="utf-8")
    assert worktree_store_name(wt) == "foo"
    assert worktree_store_name(str(wt)) == "foo"  # str 也行(ctx.cwd 是 str)


def test_worktree_store_name_none_for_main(tmp_path: Path) -> None:
    """F10:主仓/非 worktree cwd → None(存主仓目录,不进 worktree/)。"""
    from birdcode.utils.worktree import worktree_store_name

    (tmp_path / ".git").mkdir()  # 主仓(.git 目录,非 worktree)
    assert worktree_store_name(tmp_path) is None


async def test_create_worktree_calls_git_add(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """create_worktree 新分支:rev-parse 查不存在 → `git worktree add -b <branch> <path> <ref>`。"""
    (tmp_path / ".git").mkdir()  # 模拟 git 仓库(过 .git 预检)
    calls: list[tuple[list[str], Path | None]] = []

    async def _fake_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        calls.append((args, cwd))
        if args[:2] == ["rev-parse", "--verify"]:
            return (1, "", "no such branch")  # 分支不存在 → 走 -b
        return 0, "", ""

    monkeypatch.setattr(wt_mod, "_run_git", _fake_run)
    out = await wt_mod.create_worktree(tmp_path, "feat-a", base_ref="HEAD")
    assert out == tmp_path / ".birdcode" / "worktrees" / "feat-a"
    assert calls[0][0] == ["rev-parse", "--verify", "refs/heads/worktree-feat-a"]
    assert calls[1][0] == ["worktree", "add", "-b", "worktree-feat-a", str(out), "HEAD"]
    assert calls[1][1] == tmp_path


async def test_create_worktree_falls_back_to_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """origin/HEAD 失败(无 remote)→ 回退本地 HEAD(D3)。#4:超时不回退。"""
    (tmp_path / ".git").mkdir()  # 模拟 git 仓库(过 .git 预检)
    # origin/HEAD 败 → 回退 HEAD 成
    add_seq: list[tuple[int, str, str]] = [(1, "", "no origin"), (0, "", "")]

    async def _fake_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        if args[:2] == ["rev-parse", "--verify"]:
            return (1, "", "no branch")  # 分支不存在 → -b
        return add_seq.pop(0)  # worktree add 调用

    monkeypatch.setattr(wt_mod, "_run_git", _fake_run)
    await wt_mod.create_worktree(tmp_path, "x", base_ref="origin/HEAD")
    # origin/HEAD 失败 → 回退 HEAD,两次 add 都消费
    assert add_seq == []


async def test_create_worktree_no_head_fallback_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#4:超时(rc=128,err=git 超时)【不】回退 HEAD(超时非缺 ref,回退会 double 挂)。"""
    (tmp_path / ".git").mkdir()

    async def _fake_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        if args[:2] == ["rev-parse", "--verify"]:
            return (1, "", "no branch")
        return (128, "", "git 超时")  # worktree add 超时

    monkeypatch.setattr(wt_mod, "_run_git", _fake_run)
    with pytest.raises(RuntimeError, match="git worktree add 失败"):
        await wt_mod.create_worktree(tmp_path, "t", base_ref="origin/HEAD")


async def test_create_worktree_raises_if_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".git").mkdir()  # 模拟 git 仓库(过 .git 预检)
    (tmp_path / ".birdcode" / "worktrees" / "dup").mkdir(parents=True)
    monkeypatch.setattr(wt_mod, "_run_git", _fake_run_ok)
    with pytest.raises(FileExistsError):
        await wt_mod.create_worktree(tmp_path, "dup")


async def test_create_worktree_raises_non_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """非 git 目录(无 .git)→ 早 raise RuntimeError,不调 _run_git(纯检查,无需 git)。"""

    async def _boom(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        raise AssertionError("should not call _run_git for non-git")

    monkeypatch.setattr(wt_mod, "_run_git", _boom)
    with pytest.raises(RuntimeError, match="需在 git 仓库"):
        await wt_mod.create_worktree(tmp_path, "x", base_ref="HEAD")


async def _fake_run_ok(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    return 0, "", ""


async def test_remove_worktree_force_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    async def _fake_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(wt_mod, "_run_git", _fake_run)
    wt = tmp_path / ".birdcode" / "worktrees" / "x"
    await wt_mod.remove_worktree(tmp_path, wt, force=True)
    assert ["worktree", "remove", str(wt), "--force"] in calls


async def test_remove_worktree_retries_double_force_when_first_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F6:force 路径首次失败(如 locked,单 --force 仍拒)→ 重试 -ff 兜底。(#3:-ff 仅 force=True)"""
    calls: list[list[str]] = []
    seq: list[tuple[int, str, str]] = [(1, "", "locked"), (0, "", "")]  # 首败,-ff 成功

    async def _fake_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        calls.append(args)
        return seq.pop(0)

    monkeypatch.setattr(wt_mod, "_run_git", _fake_run)
    wt = tmp_path / ".birdcode" / "worktrees" / "lk"
    wt.mkdir(parents=True)
    await wt_mod.remove_worktree(tmp_path, wt, force=True)  # force=True 才 -ff
    assert ["worktree", "remove", str(wt), "--force"] in calls
    assert ["worktree", "remove", str(wt), "--force", "--force"] in calls


async def test_remove_worktree_clean_path_no_double_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#3:force=False(干净路径)首败不 -ff → prune + raise(不静默强删 TOCTOU 脏)。"""
    calls: list[list[str]] = []

    async def _fake_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        calls.append(args)
        return (1, "", "dirty")  # 首败(可能 TOCTOU 变脏)

    monkeypatch.setattr(wt_mod, "_run_git", _fake_run)
    wt = tmp_path / ".birdcode" / "worktrees" / "d"
    wt.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        await wt_mod.remove_worktree(tmp_path, wt, force=False)
    # 干净路径不 -ff(否则 TOCTOU 脏被静默强删)
    assert ["worktree", "remove", str(wt), "--force", "--force"] not in calls


async def test_create_worktree_reuses_existing_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F2 正确修法:分支已存在 → checkout 复用(`git worktree add <path> <branch>`,无 -b)。"""
    (tmp_path / ".git").mkdir()
    calls: list[list[str]] = []

    async def _fake_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        calls.append(args)
        if args[:2] == ["rev-parse", "--verify"]:
            return (0, "exists", "")  # 分支已存在 → checkout 复用
        return 0, "", ""

    monkeypatch.setattr(wt_mod, "_run_git", _fake_run)
    out = await wt_mod.create_worktree(tmp_path, "feat/x", base_ref="HEAD")
    add_calls = [c for c in calls if c[:2] == ["worktree", "add"]]
    assert add_calls == [["worktree", "add", str(out), "worktree-feat/x"]]  # 无 -b


async def test_create_worktree_fast_restore_skips_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """快速恢复:dir 已存在 + 合法 worktree + 在 worktree-<name> 分支 → 直接复用,不调 git。"""
    main = tmp_path / "main"
    main.mkdir()
    (main / ".git").mkdir()
    name = "fr"
    wt = main / ".birdcode" / "worktrees" / name
    wt.mkdir(parents=True)
    gitdir = main / ".git" / "worktrees" / name
    gitdir.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {gitdir}", encoding="utf-8")
    (gitdir / "HEAD").write_text("ref: refs/heads/worktree-fr", encoding="utf-8")

    async def _boom(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        raise AssertionError(f"快速恢复不应调 git,却调了 {args}")

    monkeypatch.setattr(wt_mod, "_run_git", _boom)
    out = await wt_mod.create_worktree(main, name, base_ref="HEAD")
    assert out == wt  # 复用,无 git 子进程


async def test_create_worktree_fast_restore_branch_mismatch_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """快速恢复:dir 在但 HEAD 不是 worktree-<name>(切了别的分支)→ 报状态不符,不复用。"""
    main = tmp_path / "main"
    main.mkdir()
    (main / ".git").mkdir()
    name = "fr"
    wt = main / ".birdcode" / "worktrees" / name
    wt.mkdir(parents=True)
    gitdir = main / ".git" / "worktrees" / name
    gitdir.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {gitdir}", encoding="utf-8")
    (gitdir / "HEAD").write_text("ref: refs/heads/other-branch", encoding="utf-8")  # 不符

    async def _boom(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        raise AssertionError(f"状态不符不应调 git,却调了 {args}")

    monkeypatch.setattr(wt_mod, "_run_git", _boom)
    with pytest.raises(FileExistsError, match="状态不符"):
        await wt_mod.create_worktree(main, name, base_ref="HEAD")


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_create_worktree_fast_restore_real(tmp_path: Path) -> None:
    """快速恢复(真机):建 → 再建同名 → 直接复用(不报已存在、不重跑 git worktree add)。"""
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
    wt1 = await wt_mod.create_worktree(main, "fr", base_ref="HEAD")
    wt2 = await wt_mod.create_worktree(main, "fr", base_ref="HEAD")  # dir 在 → 快速恢复
    assert wt1 == wt2


async def test_list_worktrees_parses_porcelain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    porcelain = (
        "worktree /r/.birdcode/worktrees/a\nHEAD abc\nbranch refs/heads/worktree-a\n\n"
        "worktree /r\nHEAD def\nbranch refs/heads/main\n"
    )

    async def _fake_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        return 0, porcelain, ""

    monkeypatch.setattr(wt_mod, "_run_git", _fake_run)
    rows = await wt_mod.list_worktrees(tmp_path)
    assert rows[0]["path"] == "/r/.birdcode/worktrees/a"
    assert rows[0]["branch"] == "refs/heads/worktree-a"
    assert rows[1]["path"] == "/r"


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_create_worktree_real(tmp_path: Path) -> None:
    """真实 git:init repo + commit → create_worktree → 是 worktree 且带主仓文件。"""
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
    created = await wt_mod.create_worktree(main, "wt-a", base_ref="HEAD")
    assert wt_mod.is_worktree(created)
    assert (created / "f.txt").exists()


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_remove_worktree_real(tmp_path: Path) -> None:
    """真实 git:建后删 → 目录消失 + list 不再含它。"""
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
    created = await wt_mod.create_worktree(main, "rm-me", base_ref="HEAD")
    await wt_mod.remove_worktree(main, created)
    assert not created.exists()


async def test_run_git_sets_no_prompt_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_run_git 设 GIT_TERMINAL_PROMPT=0 + GIT_ASKPASS=''(凭据防挂三重防御之一)。"""
    from typing import Any

    captured: dict[str, Any] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"")

        async def wait(self) -> int:
            return 0

        def kill(self) -> None:
            pass

    async def _fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_exec)
    await wt_mod._run_git(["status"], cwd=tmp_path)
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_ASKPASS"] == ""
