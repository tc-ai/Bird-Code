# tests/session/test_agent_tool_use_result.py
from birdcode.session.models import AgentToolUseResult


def test_default_is_completed_true():
    r = AgentToolUseResult(agent_id="a1", tool_use_id="c1", status="completed")
    assert r.is_completed is True


def test_can_set_false():
    r = AgentToolUseResult(agent_id="a1", tool_use_id="c1", status="completed", is_completed=False)
    assert r.is_completed is False


def test_roundtrip_alias():
    r = AgentToolUseResult(agent_id="a1", tool_use_id="c1", status="completed", is_completed=False)
    d = r.model_dump(by_alias=True, exclude_none=True)
    assert d["isCompleted"] is False
    back = AgentToolUseResult.model_validate(d)
    assert back.is_completed is False
