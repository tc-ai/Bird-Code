# tests/agents/test_report.py
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.report import (
    build_agent_tool_use_result,
    build_task_notification,
)
from birdcode.agents.runner import SubagentReport


def _defn():
    return AgentDefinition(name="explore", description="只读搜索", system_prompt="SP")


def _report(is_completed, text):
    return SubagentReport(
        is_completed=is_completed, text=text, status="completed",
        duration_ms=1234, tokens=1500, tool_use_count=3,
    )


def test_build_notification_completed_echo():
    r = _report(True, "结果正文")
    n = build_task_notification(r, _defn(), agent_id="sub-1", prompt="找 X")
    assert "<task-notification>" in n
    assert "只读搜索" in n  # description echo
    assert "Completed: yes" in n
    assert "结果正文" in n


def test_build_notification_not_completed_has_checklist():
    r = _report(False, "## 待办清单\n- 接手 Y")
    n = build_task_notification(r, _defn(), agent_id="sub-2", prompt="p")
    assert "Completed: no" in n
    assert "待办清单" in n


def test_tool_use_result_completed():
    r = _report(True, "结果")
    d = build_agent_tool_use_result(r, _defn(), agent_id="sub-3", prompt="p", is_async=False)
    assert d["agentId"] == "sub-3"
    assert d["agentType"] == "explore"
    assert d["status"] == "completed"
    assert d["isAsync"] is False
    assert d["isCompleted"] is True
    assert d["content"] == "结果"
    assert d["totalTokens"] == 1500


def test_tool_use_result_not_completed():
    r = _report(False, "## 待办清单\n- X")
    d = build_agent_tool_use_result(r, _defn(), agent_id="sub-4", prompt="p", is_async=False)
    assert d["isCompleted"] is False
