# tests/test_agent_worktree_e2e.py
"""端到端:SubagentRunner(isolation=worktree) → 真实 WriteTool 写进 worktree、主仓不变、清理正确。

与 Task 7(test_runner_worktree.py)的关键区别:
- Task 7 的 mock _fake_loop 直接 write_path.write_text()(绕过 WriteTool/executor)。
- 本 e2e 的 mock 通过 executor.execute_batch([ToolUseBlock(name="write_file")]) 驱动【真实】
  WriteTool.execute()——相对路径经 WriteTool._cwd(worktree dir,由 fork_for_child(cwd=) 注入)解析,
  验证全链:build_child_registry(cwd=) → fork_for_child → ToolExecutor → WriteTool
  → 文件落 worktree。

测试层级:SubagentRunner(AgentTool.execute 路由已由 Task 6 覆盖;permission gate 的 worktree 锁定
已由 Task 7 单测覆盖)。parent_gate=None——聚焦文件路径隔离;gate 路径由 Task 7 独立覆盖。
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from birdcode.agent.system_prompt.env import _AGENT_CWD
from birdcode.agents import runner
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.runner import SubagentRunner
from birdcode.blocks import ToolUseBlock
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.permission.gate import ModalResult, Mode, UiPermissionGate
from birdcode.session.models import SessionContext
from birdcode.tools.registry import ToolRegistry
from birdcode.tools.write_tool import WriteTool

_HAS_GIT = shutil.which("git") is not None


def _init_repo(main: Path) -> None:
    """init git repo + 初始 commit src/foo.py(给 worktree 基线 + src/ 目录)。"""
    main.mkdir()
    for c in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=main, check=True, capture_output=True)
    (main / "src").mkdir()
    (main / "src" / "foo.py").write_text("# original\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=main, check=True, capture_output=True)


def _cfg() -> AppConfig:
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


def _parent_registry() -> ToolRegistry:
    """父工具表:仅含 WriteTool。

    build_child_registry 会经 fork_for_child(cwd=worktree_dir) 派生子 WriteTool——
    其 _cwd=worktree,相对路径 src/xxx 解析到 worktree(非主仓)。
    """
    reg = ToolRegistry()
    reg.register(WriteTool())
    return reg


# ---------------- Test 1: 单 worktree 子 agent 写隔离 + 清理 ----------------


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_worktree_subagent_isolates_writes(tmp_path, monkeypatch):
    """isolation=worktree → 真实 WriteTool 经 executor 写进 worktree(非主仓);clean → 清理。

    mock run_agent_loop 通过 executor.execute_batch 驱动真实 WriteTool.execute(
    file_path="src/bar.py"):
    WriteTool._cwd 已被 fork_for_child(cwd=worktree_dir) 注入 → 相对路径解析到 worktree。
    commit → clean working tree → cleanup_subagent_worktree(succeeded=True) 删 worktree 目录。
    断言:文件落在 <主仓>/.birdcode/worktrees/<agent_id>/src/bar.py;主仓 src/bar.py 不存在;
    主仓 src/foo.py 内容不变;clean → worktree 目录被清理;report completed。
    """
    main = tmp_path / "main"
    _init_repo(main)

    cfg = _cfg()
    ctx = SessionContext(session_id="s1", cwd=str(main), version="", git_branch=None)
    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")

    monkeypatch.setattr(runner, "build_provider", _fake_build_provider)

    captured: dict[str, str | None] = {"write_output": None, "wt_path": None}

    async def _fake_loop(**kwargs):
        executor = kwargs["executor"]
        wt_cwd = _AGENT_CWD.get()
        assert wt_cwd is not None, "_AGENT_CWD 应在 run_agent_loop 前设为 worktree dir"
        # 驱动真实 WriteTool 经 executor 执行(相对路径 src/bar.py → 解析到 worktree)
        tu = ToolUseBlock(
            id="call_write", name="write_file",
            input={"file_path": "src/bar.py", "content": "# subagent output\n"},
        )
        results = await executor.execute_batch([tu])
        assert results[0].ok, f"write_file 应成功: {results[0].error_message}"
        captured["write_output"] = results[0].output
        # DURING mock(cleanup 前验证):文件落在 worktree 内(非主仓)
        wt_path = Path(wt_cwd) / "src" / "bar.py"
        assert wt_path.exists(), "文件应写在 worktree 内(WriteTool._cwd=worktree 解析相对路径)"
        assert wt_path.read_text(encoding="utf-8") == "# subagent output\n"
        captured["wt_path"] = str(wt_path)
        # commit → clean working tree → cleanup 能删目录(§6.6)
        subprocess.run(["git", "add", "."], cwd=wt_cwd, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "work"], cwd=wt_cwd, check=True, capture_output=True)

    monkeypatch.setattr(runner, "run_agent_loop", _fake_loop)

    r = SubagentRunner(
        defn=defn, prompt="写 src/bar.py", description="d", tool_use_id="call_wt",
        model_override="", spawn_depth=1, is_async=True,
        isolation="worktree",
        parent_provider=_ParentProviderStub(cfg.providers["p"]),
        parent_registry=_parent_registry(), parent_gate=None, cfg=cfg, app=None, ctx=ctx,
        project_root=main, root=tmp_path,
    )
    report = await r.run()

    # write_file 真实执行成功(executor 驱动了 WriteTool)
    assert captured["write_output"] is not None
    assert "已写入" in captured["write_output"]
    # 文件路径在 worktree 下(captured during mock,cleanup 前)
    assert captured["wt_path"] is not None
    assert ".birdcode" in captured["wt_path"]
    assert "worktrees" in captured["wt_path"]
    # 主仓 src/bar.py 不存在(写没落到主仓)
    assert not (main / "src" / "bar.py").exists()
    # 主仓 src/foo.py 内容不变(主仓未被踩)
    assert (main / "src" / "foo.py").read_text(encoding="utf-8") == "# original\n"
    # clean working tree → cleanup 删 worktree 目录
    wt_dir = main / ".birdcode" / "worktrees" / r.agent_id
    assert not wt_dir.exists()
    # _AGENT_CWD 已复位(finally reset → None)
    assert _AGENT_CWD.get() is None
    # report completed
    assert report.status == "completed"


# ---------------- Test 2: 两个并发 worktree 子 agent 不互踩 ----------------


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_two_worktree_subagents_dont_collide(tmp_path, monkeypatch):
    """两个并发 isolation=worktree 子 agent → 各自 worktree、同名文件内容不串。

    两个 SubagentRunner 并发(asyncio.gather),各自 isolation=worktree → 各自 worktree(独立 agent_id
    → 独立目录)。mock run_agent_loop 各驱动真实 WriteTool 写 src/out.txt(同名),
    内容按 agent_id 区分。
    不 commit(dirty → cleanup 留目录)→ run() 后可读取各自 worktree 的文件断言内容不串。
    断言:两 worktree 目录不同;各自 src/out.txt 内容独立不串;主仓 src/out.txt 不存在;主仓不变。
    """
    main = tmp_path / "main"
    _init_repo(main)

    cfg = _cfg()
    ctx = SessionContext(session_id="s2", cwd=str(main), version="", git_branch=None)
    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")

    monkeypatch.setattr(runner, "build_provider", _fake_build_provider)

    # agent_id → 唯一内容(runner __init__ 生成 agent_id,run() 前注册)
    contents: dict[str, str] = {}

    async def _fake_loop(**kwargs):
        executor = kwargs["executor"]
        wt_cwd = _AGENT_CWD.get()
        assert wt_cwd is not None, "_AGENT_CWD 应在 run_agent_loop 前设为 worktree dir"
        agent_id = Path(wt_cwd).name
        content = contents[agent_id]
        # 驱动真实 WriteTool 经 executor 执行(同名 src/out.txt,不同内容)
        tu = ToolUseBlock(
            id=f"call_{agent_id}", name="write_file",
            input={"file_path": "src/out.txt", "content": content},
        )
        results = await executor.execute_batch([tu])
        assert results[0].ok, f"write_file 应成功: {results[0].error_message}"
        # 不 commit(dirty → cleanup 留目录 → run() 后可读文件断言内容不串)

    monkeypatch.setattr(runner, "run_agent_loop", _fake_loop)

    r1 = SubagentRunner(
        defn=defn, prompt="写 src/out.txt", description="A", tool_use_id="call_a",
        model_override="", spawn_depth=1, is_async=True,
        isolation="worktree",
        parent_provider=_ParentProviderStub(cfg.providers["p"]),
        parent_registry=_parent_registry(), parent_gate=None, cfg=cfg, app=None, ctx=ctx,
        project_root=main, root=tmp_path,
    )
    r2 = SubagentRunner(
        defn=defn, prompt="写 src/out.txt", description="B", tool_use_id="call_b",
        model_override="", spawn_depth=1, is_async=True,
        isolation="worktree",
        parent_provider=_ParentProviderStub(cfg.providers["p"]),
        parent_registry=_parent_registry(), parent_gate=None, cfg=cfg, app=None, ctx=ctx,
        project_root=main, root=tmp_path,
    )
    contents[r1.agent_id] = "content from A"
    contents[r2.agent_id] = "content from B"

    # 并发跑两个 worktree 子 agent(asyncio.Task 各自 copy context → _AGENT_CWD 隔离)
    rep1, rep2 = await asyncio.gather(r1.run(), r2.run())

    # 两 worktree 目录不同(独立 agent_id)
    wt1 = main / ".birdcode" / "worktrees" / r1.agent_id
    wt2 = main / ".birdcode" / "worktrees" / r2.agent_id
    assert wt1 != wt2
    # dirty(不 commit)→ cleanup 留目录 → 两 worktree 目录仍在
    assert wt1.exists()
    assert wt2.exists()
    # 各自 src/out.txt 内容独立不串(同名文件、不同内容)
    assert (wt1 / "src" / "out.txt").read_text(encoding="utf-8") == "content from A"
    assert (wt2 / "src" / "out.txt").read_text(encoding="utf-8") == "content from B"
    # 主仓 src/out.txt 不存在(两子 agent 的写都没落到主仓)
    assert not (main / "src" / "out.txt").exists()
    # 主仓 src/foo.py 不变
    assert (main / "src" / "foo.py").read_text(encoding="utf-8") == "# original\n"
    # 两子 agent 都 completed
    assert rep1.status == "completed"
    assert rep2.status == "completed"


# ---------------- Test 3: worktree 子 agent + 真实 gate + 相对路径写入 ----------------


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_worktree_subagent_real_gate_relative_write(tmp_path, monkeypatch):
    """C1 回归:worktree 异步子 agent + 真实 UiPermissionGate(父)+ 相对路径 write_file。

    生产场景:父 gate=UiPermissionGate(project_root=主仓, sandbox_root=主仓)。
    build_child_gate→fork_worktree_async(sandbox_root=worktree)→子 gate 的 _sandbox_root=worktree,
    _project_root=主仓(skill/permission 锚)。子 agent 写相对路径 src/new.py:
    - WriteTool._cwd=worktree(fork_for_child 注入)→ 文件应落 worktree/src/new.py。
    - 子 gate L2(check_path)必须把相对路径解析到 worktree(=sandbox_root),否则:
      raw "src/new.py".resolve()→<进程 cwd>/src/new.py(非 worktree/非主仓)→L2 越界拒 → 写失败。

    修复前(RED):L2 用进程 cwd 解析相对路径 → 越界 → write_file 返回 deny → ok=False。
    修复后(GREEN):gate 用 sandbox_root 解析相对路径 → L2 通过 → accept-edits L4 放行
      → 文件落 worktree/src/new.py,主仓不变。
    """
    main = tmp_path / "main"
    _init_repo(main)

    cfg = _cfg()
    ctx = SessionContext(session_id="s3", cwd=str(main), version="", git_branch=None)
    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")

    monkeypatch.setattr(runner, "build_provider", _fake_build_provider)

    # 真实父 gate:project_root=主仓、sandbox_root=主仓、accept-edits(write_file L4 自动放行,
    # 绕过 L5;fork_worktree_async 的 _approve 兜底也恒 approve)。子 gate 由 build_child_gate
    # 经 fork_worktree_async(sandbox_root=worktree) 派生——_sandbox_root=worktree。
    async def _get_mode() -> Mode:
        return "accept-edits"

    async def _approve(_tool: str, _summary: str, _path: str | None) -> ModalResult:
        return "approve"

    parent_gate = UiPermissionGate(
        get_mode=_get_mode,
        request_permission=_approve,
        project_root=main,
        sandbox_root=main,
    )

    captured: dict[str, str | None] = {"wt_path": None, "err": None, "content": None}

    async def _fake_loop(**kwargs):
        executor = kwargs["executor"]
        wt_cwd = _AGENT_CWD.get()
        assert wt_cwd is not None, "_AGENT_CWD 应在 run_agent_loop 前设为 worktree dir"
        tu = ToolUseBlock(
            id="call_rel", name="write_file",
            input={"file_path": "src/new.py", "content": "# relative path\n"},
        )
        results = await executor.execute_batch([tu])
        if not results[0].ok:
            captured["err"] = results[0].error_message
            return
        wt_path = Path(wt_cwd) / "src" / "new.py"
        # DURING mock(cleanup 前验证):文件落在 worktree 内、内容正确
        assert wt_path.exists(), "文件应写在 worktree 内(WriteTool._cwd=worktree 解析相对路径)"
        captured["content"] = wt_path.read_text(encoding="utf-8")
        captured["wt_path"] = str(wt_path)
        # commit → clean → cleanup 能删 worktree 目录
        subprocess.run(["git", "add", "."], cwd=wt_cwd, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "rel"], cwd=wt_cwd, check=True, capture_output=True)

    monkeypatch.setattr(runner, "run_agent_loop", _fake_loop)

    r = SubagentRunner(
        defn=defn, prompt="写 src/new.py(相对路径)", description="d", tool_use_id="call_rel",
        model_override="", spawn_depth=1, is_async=True,
        isolation="worktree",
        parent_provider=_ParentProviderStub(cfg.providers["p"]),
        parent_registry=_parent_registry(), parent_gate=parent_gate, cfg=cfg, app=None, ctx=ctx,
        project_root=main, root=tmp_path,
    )
    report = await r.run()

    # 关键断言:write_file 在真实 gate 下成功(L2 未误拒相对路径)
    assert captured["err"] is None, f"相对路径 write_file 被 gate 误拒(L2?): {captured['err']}"
    assert captured["wt_path"] is not None, "write_file 应成功并落盘到 worktree"
    # 文件落在 worktree 内
    assert ".birdcode" in captured["wt_path"]
    assert "worktrees" in captured["wt_path"]
    assert captured["content"] == "# relative path\n"
    # 主仓未被写入
    assert not (main / "src" / "new.py").exists()
    assert (main / "src" / "foo.py").read_text(encoding="utf-8") == "# original\n"
    assert report.status == "completed"
