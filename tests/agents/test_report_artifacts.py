# tests/agents/test_report_artifacts.py
from __future__ import annotations

from birdcode.agents.definition import AgentDefinition, AgentSource
from birdcode.agents.report import build_task_notification
from birdcode.agents.runner import SubagentReport


def _defn() -> AgentDefinition:
    return AgentDefinition(
        name="general-purpose", description="d", system_prompt="",
        disallowed_tools=(), source=AgentSource.BUILTIN,
        kind="write", parallel_safe=False,
    )


def _report(**kw) -> SubagentReport:
    base = dict(
        is_completed=True, text="done", status="completed",
        duration_ms=10, tokens=5, tool_use_count=1,
    )
    base.update(kw)
    return SubagentReport(**base)


def test_artifacts_uncommitted_worktree():
    msg = build_task_notification(
        _report(artifacts=["sort.py", "foo.txt"], committed=False), _defn(),
        agent_id="sub-abc", prompt="p", is_worktree=True,
    )
    assert "Artifacts (2, 未提交): sort.py, foo.txt" in msg


def test_artifacts_committed_worktree_mentions_branch():
    msg = build_task_notification(
        _report(artifacts=["a.py"], committed=True), _defn(),
        agent_id="sub-abc", prompt="p", is_worktree=True,
    )
    assert "Artifacts (1, 已提交到 worktree-sub-abc): a.py" in msg


def test_artifacts_empty_worktree_shows_none():
    msg = build_task_notification(
        _report(), _defn(), agent_id="sub-abc", prompt="p", is_worktree=True,
    )
    assert "Artifacts: 无" in msg


def test_non_worktree_omits_artifacts_line():
    """非 worktree 模式 → 不显示 Artifacts 行(即使 artifacts 空)。"""
    msg = build_task_notification(
        _report(), _defn(), agent_id="sub-abc", prompt="p", is_worktree=False,
    )
    assert "Artifacts" not in msg


def test_non_worktree_default_omits_artifacts_line():
    """is_worktree 默认 False → 同样省略。"""
    msg = build_task_notification(
        _report(), _defn(), agent_id="sub-abc", prompt="p",
    )
    assert "Artifacts" not in msg
