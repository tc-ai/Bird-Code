# tests/tools/test_resume_agent_tool.py
"""ResumeAgentTool 单测:mock resume_subagent,验证 async→ack 文本 / sync→inline 报告。"""
from pathlib import Path

import pytest

from birdcode.agents.resume import ResumeDeps, ResumeResult
from birdcode.tools.resume_agent_tool import ResumeAgentTool


def _build_resume_tool(deps: ResumeDeps | None = None) -> ResumeAgentTool:
    """构造 ResumeAgentTool;deps 默认全 None(resume_subagent 被 mock,不会真用到)。"""
    if deps is None:
        deps = ResumeDeps(
            manager=None,  # type: ignore[arg-type]
            root=Path("."),
            session_id="test-sid",
            project_root=Path("."),
            worktree_name=None,
            agent_registry=None,
            parent_provider=None,
            parent_registry=None,
            parent_gate=None,
            cfg=None,
            app=None,
            ctx=None,
        )
    return ResumeAgentTool(deps=deps)


@pytest.mark.asyncio
async def test_resume_agent_tool_async_returns_ack(monkeypatch):
    async def fake_resume(*, agent_id, direction, deps):
        return ResumeResult(outcome="async_launched", text=f"ack:{agent_id}:{direction}")

    monkeypatch.setattr(
        "birdcode.tools.resume_agent_tool.resume_subagent", fake_resume
    )
    tool = _build_resume_tool()
    out = await tool.execute(agent_id="sub-1", direction="继续,改成 X")
    assert "ack:sub-1" in out and "改成 X" in out


@pytest.mark.asyncio
async def test_resume_agent_tool_sync_returns_inline_report(monkeypatch):
    async def fake_resume(*, agent_id, direction, deps):
        return ResumeResult(outcome="sync_done", text="REPORT")

    monkeypatch.setattr(
        "birdcode.tools.resume_agent_tool.resume_subagent", fake_resume
    )
    tool = _build_resume_tool()
    out = await tool.execute(agent_id="sub-sync", direction="继续")
    assert out == "REPORT"


def test_resume_agent_tool_metadata():
    """契约:klass 级别 kind/parallel_safe/name/description 正确(单工具,非 per-defn)。"""
    tool = _build_resume_tool()
    assert tool.name == "resume_agent"
    assert tool.kind == "write"
    assert tool.parallel_safe is False
    # 递归防护标记:派生子 agent(复用 id)→ build_child_registry 据此排除,
    # 否则子 agent 调 resume_agent 会绕 MAX_SPAWN_DEPTH 无界递归。
    assert ResumeAgentTool.is_agent_tool is True
    assert tool.description  # 非空
    # parameters 是 ResumeAgentInput(类属性,非每实例)
    from birdcode.tools.resume_agent_tool import ResumeAgentInput

    assert ResumeAgentTool.parameters is ResumeAgentInput
    # name 是实例属性(每个 ResumeAgentTool 实例都同名,但不是 per-defn)
    assert isinstance(tool.name, str)


def test_rebind_updates_ctx_session_id_and_provider():
    """review #4:rebind 把 /clear 的新 ctx/session_id、/profile 的新 provider 传导到 deps。

    修 #4 前:ResumeAgentTool 标了 is_agent_tool 却无 rebind → app._rebind_agent_tool 遍历调
    tool.rebind 抛 AttributeError → 被 clear_conversation 的 except 吞 → 跳过其后的
    manager.rebind / team.reset;且 /clear 后 deps 停留旧 session_id → 续跑找不到新 session 的 meta。
    """
    from birdcode.session.models import SessionContext

    tool = _build_resume_tool()
    assert tool._deps.session_id == "test-sid"  # noqa: SLF001 - 初始值
    new_ctx = SessionContext(session_id="new-sid", cwd=".", version="", git_branch=None)
    sentinel = object()
    tool.rebind(ctx=new_ctx, provider=sentinel)  # type: ignore[arg-type]
    assert tool._deps.session_id == "new-sid"  # noqa: SLF001
    assert tool._deps.ctx is new_ctx  # noqa: SLF001
    assert tool._deps.parent_provider is sentinel  # noqa: SLF001
