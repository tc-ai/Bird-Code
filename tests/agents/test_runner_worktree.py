# tests/agents/test_runner_worktree.py
"""Phase 2 §6.2/§6.7:build_child_registry(cwd=) + build_child_gate(worktree_dir=)。

cwd 透传 fork_for_child(cwd=)→ BashTool/6 文件工具的 _cwd 注入(worktree 隔离)。
worktree_dir 非空 → fork_worktree_async(sandbox_root=worktree_dir)(L2 锁 worktree,
主仓路径 L2 拒);worktree_dir=None → 原 fork_async 路径(L5 拒写,怕毁主仓);
allow_hitl=True → 复用父 gate(同步,含 HITL)。
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from birdcode.agents import runner
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.runner import SubagentRunner, build_child_gate, build_child_registry
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.permission.gate import UiPermissionGate
from birdcode.session.models import SessionContext
from birdcode.tools.bash_tool import BashTool
from birdcode.tools.registry import ToolRegistry
from birdcode.tools.write_tool import WriteTool

_HAS_GIT = shutil.which("git") is not None

# ---------------- build_child_registry(cwd=) ----------------


def test_build_child_registry_passes_cwd_to_fork(tmp_path: Path) -> None:
    """cwd=worktree_dir → 子工具经 fork_for_child(cwd=worktree_dir) 派生,BashTool._cwd=worktree。

    端到端验:不 mock fork_for_child,直接观察子 bash 实例的 _cwd 被注入了 worktree dir
    (BashTool.fork_for_child: child._cwd = cwd or self._cwd)。cwd 非 None → 覆盖父快照。
    """
    reg = ToolRegistry()
    reg.register(BashTool())
    wt = str(tmp_path / "wt")
    child = build_child_registry(reg, disallowed=(), cwd=wt)
    child_bash = child.get("bash")
    assert child_bash is not None
    assert child_bash is not reg.get("bash")  # 全新实例(fork,非共享)
    assert child_bash._cwd == wt  # noqa: SLF001 - cwd 注入 fork_for_child → _cwd


def test_build_child_registry_cwd_none_default_snapshots_parent() -> None:
    """cwd=None(非 worktree)→ fork_for_child(cwd=None),BashTool 快照父 _cwd(旧行为,向后兼容)。

    None → BashTool.fork_for_child 的 `cwd or self._cwd` 取父快照,与 Phase 1 行为一致。
    """
    reg = ToolRegistry()
    reg.register(BashTool())
    parent_bash = reg.get("bash")
    assert parent_bash is not None
    parent_bash._cwd = "/fake/parent/cwd"  # noqa: SLF001
    child = build_child_registry(reg, disallowed=())  # cwd 默认 None
    child_bash = child.get("bash")
    assert child_bash is not None
    assert child_bash is not parent_bash  # 全新实例
    assert child_bash._cwd == "/fake/parent/cwd"  # noqa: SLF001 - None → 快照父 _cwd


# ---------------- build_child_gate(worktree_dir=) ----------------


def _make_parent_gate(main: Path) -> UiPermissionGate:
    """父 gate:project_root=sandbox_root=main(主仓)。父 HITL 恒 reject。

    恒 reject 用于对比:worktree gate 应恒 approve(不调父回调);fork_async gate 对写 reject。
    """

    async def _mode() -> str:
        return "default"

    async def _perm(_t: str, _s: str, _p: str | None) -> str:
        return "reject"

    return UiPermissionGate(
        get_mode=_mode,  # type: ignore[arg-type]
        request_permission=_perm,  # type: ignore[arg-type]
        project_root=main,
        sandbox_root=main,
    )


async def test_build_child_gate_worktree_dir_routes_to_fork_worktree_async(tmp_path: Path) -> None:
    """worktree_dir 非空 → fork_worktree_async(sandbox_root=worktree_dir)。

    加强(避免「g is not parent」假阳性:fork_async 也产新 gate ≠ parent):用主仓路径写
    L2 拒 来证明 sandbox=worktree。fork_worktree_async 的 sandbox_root=worktree → 主仓路径
    不在 roots → L2 硬拒;若误走 fork_async(sandbox=父=main),主仓路径会过 L2 到 L5 拒
    (layer=L5 非 L2)。layer=="L2" 唯一区分二者。
    """
    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    parent = _make_parent_gate(main)
    g = build_child_gate(parent, allow_hitl=False, worktree_dir=wt)
    assert g is not parent  # 新 gate
    assert g is not None
    # 主仓路径写 → L2 拒(sandbox_root=worktree,main 不在 roots)。
    v = await g.check(WriteTool(), {"file_path": str(main / "f.txt"), "content": "x"})
    assert v.action == "deny"
    assert v.layer == "L2"  # 唯一标识 fork_worktree_async(fork_async 会是 L5)


def test_build_child_gate_no_worktree_uses_fork_async(tmp_path: Path) -> None:
    """worktree_dir=None,allow_hitl=False → 原 fork_async 路径(产新 gate ≠ 父)。

    worktree_dir=None 排除 worktree 分支;allow_hitl=False 排除复用父分支;唯一剩余路径
    即 fork_async。g is not parent 证明走了 fork_async(产新 gate)。
    """
    main = tmp_path / "main"
    main.mkdir()
    parent = _make_parent_gate(main)
    g = build_child_gate(parent, allow_hitl=False, worktree_dir=None)
    assert g is not parent  # fork_async 产新 gate


def test_build_child_gate_allow_hitl_returns_parent(tmp_path: Path) -> None:
    """allow_hitl=True(同步子 agent)且无 worktree_dir → 复用父 gate(不 fork,含 HITL)。向后兼容。

    注:worktree_dir 非空时优先走 worktree 分支(见
    test_build_child_gate_worktree_wins_over_allow_hitl),故 allow_hitl 复用父
    仅在 worktree_dir=None(纯同步)时成立。
    """
    main = tmp_path / "main"
    main.mkdir()
    parent = _make_parent_gate(main)
    g = build_child_gate(parent, allow_hitl=True)
    assert g is parent  # 同步:复用父(L1-L5 全照常,含 HITL)


async def test_build_child_gate_worktree_wins_over_allow_hitl(tmp_path: Path) -> None:
    """worktree_dir + allow_hitl=True(sync+worktree 误组合)→ worktree 分支胜(sandbox=worktree)。

    F6 回归:原 allow_hitl 在前 → 该组合回退父 gate(sandbox=主仓,与工具 _cwd=worktree 口径分歧)。
    生产中 execute 对 isolation=worktree force-async 使该组合不出现;build_child_gate 自身应
    健壮:worktree_dir 非空即用 fork_worktree_async。layer=="L2" 证明 sandbox=worktree(主仓路径
    写被 L2 拒);若回退父 gate(sandbox=主仓)则会过 L2 到 L5。
    """
    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    parent = _make_parent_gate(main)
    g = build_child_gate(parent, allow_hitl=True, worktree_dir=wt)
    assert g is not parent  # 走 worktree 分支(非复用父)
    v = await g.check(WriteTool(), {"file_path": str(main / "f.txt"), "content": "x"})
    assert v.action == "deny"
    assert v.layer == "L2"  # sandbox=worktree → 主仓路径 L2 拒


# ---------------- SubagentRunner.run() worktree 生命周期(Task 7)----------------


def _init_repo(main: Path) -> None:
    """init 一个 git 仓库 + 一个初始 commit(给 worktree 基线)。"""
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


def _wt_cfg() -> AppConfig:
    return AppConfig(
        providers={
            "p": ProviderProfile(
                name="p", protocol="anthropic", model="m",
                base_url="http://x", api_key="k",
            )
        },
        default="p",
    )


class _ParentProviderStub:
    """满足 ParentProvider 协议(仅 .profile)。"""

    def __init__(self, profile: ProviderProfile) -> None:
        self.profile = profile


def _fake_build_provider(  # noqa: ARG001
    profile, app, *, registry=None, system_override=None, mcp_instructions=None,
):
    """避免真实 API:返回仅含 .profile 的桩(run_agent_loop 也被 mock,不会 stream)。"""
    return type("P", (), {"profile": profile})()


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_runner_worktree_isolation_writes_in_worktree(tmp_path, monkeypatch):
    """isolation=worktree → 子 agent 文件写进 worktree(非主仓);结束清理(clean → dir 删)。

    mock run_agent_loop 在 worktree 内写文件 + commit(clean working tree → cleanup 能删 dir)。
    断言:文件路径在 <主仓>/.birdcode/worktrees/sub-*/ 下;主仓无该文件;run 结束后 worktree
    目录清理(removed);_AGENT_CWD 复位;report completed。
    """
    main = tmp_path / "main"
    _init_repo(main)

    cfg = _wt_cfg()
    ctx = SessionContext(session_id="s1", cwd=str(main), version="", git_branch=None)
    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")

    monkeypatch.setattr(runner, "build_provider", _fake_build_provider)

    captured: dict[str, Path | None] = {"write_path": None}

    async def _fake_loop(**kwargs):  # noqa: ARG001
        from birdcode.agent.system_prompt.env import _AGENT_CWD

        wt_dir = main / ".birdcode" / "worktrees" / r.agent_id
        # worktree 应在 build_provider 前创建(run_agent_loop 调用时已存在)
        assert wt_dir.exists(), "worktree 目录应在 run_agent_loop 前创建"
        # _AGENT_CWD 应已设为 worktree dir(系统提示 env 块用 worktree cwd)
        assert _AGENT_CWD.get() == str(wt_dir), "_AGENT_CWD 应在 run_agent_loop 前设为 worktree dir"
        write_path = wt_dir / "output.txt"
        write_path.write_text("子 agent 的产出", encoding="utf-8")
        captured["write_path"] = write_path
        # commit(clean working tree → remove_worktree 能删)
        subprocess.run(["git", "add", "."], cwd=wt_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "work"], cwd=wt_dir, check=True, capture_output=True)

    monkeypatch.setattr(runner, "run_agent_loop", _fake_loop)

    r = SubagentRunner(
        defn=defn, prompt="写 output.txt", description="d", tool_use_id="call_wt",
        model_override="", spawn_depth=1, is_async=False,
        isolation="worktree",
        parent_provider=_ParentProviderStub(cfg.providers["p"]),
        parent_registry=None, parent_gate=None, cfg=cfg, app=None, ctx=ctx,
        project_root=main, root=tmp_path,
    )
    report = await r.run()

    # 文件落在 worktree 下(非主仓)
    write_path = captured["write_path"]
    assert write_path is not None
    assert ".birdcode" in write_path.parts
    assert "worktrees" in write_path.parts
    # 主仓没有 output.txt
    assert not (main / "output.txt").exists()
    # worktree 目录已清理(clean working tree → remove 成功)
    wt_dir = main / ".birdcode" / "worktrees" / r.agent_id
    assert not wt_dir.exists()
    # _AGENT_CWD 已复位(finally reset → None)
    from birdcode.agent.system_prompt.env import _AGENT_CWD
    assert _AGENT_CWD.get() is None
    # report completed
    assert report.status == "completed"


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_runner_worktree_kept_on_failure(tmp_path, monkeypatch):
    """isolation=worktree + run_agent_loop 抛异常 → worktree 保留(failed → 留目录)。

    异常被 except Exception 捕获 → status='error';finally cleanup(succeeded=False)→ 留目录。
    _AGENT_CWD 仍复位(finally 即使失败也跑 reset)。
    """
    main = tmp_path / "main"
    _init_repo(main)

    cfg = _wt_cfg()
    ctx = SessionContext(session_id="s2", cwd=str(main), version="", git_branch=None)
    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")

    monkeypatch.setattr(runner, "build_provider", _fake_build_provider)

    async def _boom_loop(**kwargs):  # noqa: ARG001
        raise RuntimeError("子 agent 崩了")

    monkeypatch.setattr(runner, "run_agent_loop", _boom_loop)

    r = SubagentRunner(
        defn=defn, prompt="会崩", description="d", tool_use_id="call_wt2",
        model_override="", spawn_depth=1, is_async=False,
        isolation="worktree",
        parent_provider=_ParentProviderStub(cfg.providers["p"]),
        parent_registry=None, parent_gate=None, cfg=cfg, app=None, ctx=ctx,
        project_root=main, root=tmp_path,
    )
    report = await r.run()

    # 异常被捕获 → error 报告
    assert report.status == "error"
    # worktree 保留(failed → 留目录)
    wt_dir = main / ".birdcode" / "worktrees" / r.agent_id
    assert wt_dir.exists()
    # _AGENT_CWD 仍复位(finally reset 即使失败也跑)
    from birdcode.agent.system_prompt.env import _AGENT_CWD
    assert _AGENT_CWD.get() is None


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_runner_worktree_cleanup_cancel_preserves_completed_report(tmp_path, monkeypatch):
    """Esc 落在已完成 worktree 子 agent 的清理窗口 → 不丢已完成报告(F2 回归)。

    finally 内 cleanup await 未 shield 时,cancel 经 _run_git 抛 CancelledError 冲出 finally →
    return report 不执行 → _supervise 改写成 'cancelled'(已完成报告丢失),且 worktree 可能留
    git-locked。修后:shield 让清理跑完,report 非 None(completed)→ 交付报告而非上传取消。
    """
    main = tmp_path / "main"
    _init_repo(main)

    cfg = _wt_cfg()
    ctx = SessionContext(session_id="s3", cwd=str(main), version="", git_branch=None)
    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")

    monkeypatch.setattr(runner, "build_provider", _fake_build_provider)

    async def _fake_loop(**kwargs):  # noqa: ARG001
        pass  # 正常完成(无文件写,working tree clean)

    monkeypatch.setattr(runner, "run_agent_loop", _fake_loop)

    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def _blocking_cleanup(main_repo, wt, *, succeeded):  # noqa: ARG001
        cleanup_entered.set()
        await cleanup_release.wait()  # 阻塞,模拟清理进行中被 Esc

    monkeypatch.setattr(runner, "cleanup_subagent_worktree", _blocking_cleanup)

    r = SubagentRunner(
        defn=defn, prompt="ok", description="d", tool_use_id="call_c",
        model_override="", spawn_depth=1, is_async=False, isolation="worktree",
        parent_provider=_ParentProviderStub(cfg.providers["p"]),
        parent_registry=None, parent_gate=None, cfg=cfg, app=None, ctx=ctx,
        project_root=main, root=tmp_path,
    )
    task = asyncio.create_task(r.run())
    await cleanup_entered.wait()  # 确认进入清理窗口(runner 悬在 shield 上)
    task.cancel()  # Esc 落在清理窗口
    cleanup_release.set()  # 让被 shield 保护的清理完成
    report = await task  # 修后:返回 completed(非抛 CancelledError)
    assert report.status == "completed"


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_runner_worktree_cleanup_exception_preserves_completed_report(
    tmp_path, monkeypatch
):
    """清理抛非取消异常(_run_git OSError/EMFILE/git 不可用)→ 不掩盖已完成报告(F2 完整版)。

    回归复审(04a546c):原外层 except 只接 CancelledError,cleanup 抛 OSError 时穿透 shield
    逃出 finally,替换掉 return report → _supervise 报 error(已完成报告被掩盖)。修后:except
    接 (CancelledError, Exception),report 非 None → 交付。与取消路径对称。
    """
    main = tmp_path / "main"
    _init_repo(main)

    cfg = _wt_cfg()
    ctx = SessionContext(session_id="s4", cwd=str(main), version="", git_branch=None)
    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")

    monkeypatch.setattr(runner, "build_provider", _fake_build_provider)

    async def _fake_loop(**kwargs):  # noqa: ARG001
        pass  # 正常完成(无文件写)

    monkeypatch.setattr(runner, "run_agent_loop", _fake_loop)

    async def _boom_cleanup(main_repo, wt, *, succeeded):  # noqa: ARG001
        raise OSError("git 不可用 / EMFILE")

    monkeypatch.setattr(runner, "cleanup_subagent_worktree", _boom_cleanup)

    r = SubagentRunner(
        defn=defn, prompt="ok", description="d", tool_use_id="call_e",
        model_override="", spawn_depth=1, is_async=False, isolation="worktree",
        parent_provider=_ParentProviderStub(cfg.providers["p"]),
        parent_registry=None, parent_gate=None, cfg=cfg, app=None, ctx=ctx,
        project_root=main, root=tmp_path,
    )
    report = await r.run()
    assert report.status == "completed"  # 不被清理的 OSError 掩盖成 error
