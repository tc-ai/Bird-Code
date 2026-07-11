# tests/agent/system_prompt/test_env_cwd.py
"""env.py 的 cwd 隔离(Phase 2 worktree §6.5)。

_AGENT_CWD contextvar + 按 resolved cwd 键的 lru_cache,保证 worktree 子 agent
系统提示的环境块显示自己的 cwd/git,而非主仓进程 cwd。
"""

from birdcode.agent.system_prompt import env


def test_stable_env_reflects_cwd(tmp_path):
    """gather_stable_env(cwd=wt) → '工作目录: wt'(非进程 cwd)。"""
    wt = str(tmp_path / "wt")
    s = env.gather_stable_env(cwd=wt)
    assert f"工作目录: {wt}" in s


def test_stable_env_uses_contextvar_when_no_arg(tmp_path, monkeypatch):
    """_AGENT_CWD set → gather_stable_env() 无参也读到 worktree cwd。"""
    wt = str(tmp_path / "wt")
    tok = env._AGENT_CWD.set(wt)
    try:
        s = env.gather_stable_env()
        assert f"工作目录: {wt}" in s
    finally:
        env._AGENT_CWD.reset(tok)


def test_stable_env_cache_keyed_by_cwd(tmp_path):
    """不同 cwd → 不同缓存条目(主仓/worktree 共存,不串)。"""
    main = str(tmp_path / "main")
    wt = str(tmp_path / "wt")
    s_main = env.gather_stable_env(cwd=main)
    s_wt = env.gather_stable_env(cwd=wt)
    assert main in s_main and wt not in s_main
    assert wt in s_wt and main not in s_wt
    # 再取一次主仓——仍是主仓(缓存不串)
    assert env.gather_stable_env(cwd=main) == s_main


async def test_git_branch_dirty_uses_contextvar(tmp_path):
    """_AGENT_CWD set → _git_branch_dirty 读 worktree 的 git status。"""
    import subprocess

    main = tmp_path / "main"
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
    tok = env._AGENT_CWD.set(str(main))
    try:
        out = await env._git_branch_dirty()
    finally:
        env._AGENT_CWD.reset(tok)
    assert "clean" in out  # worktree cwd 的 git status
