# tests/session/test_models.py
from birdcode.session.models import (
    AssistantLine,
    PersistInfo,
    SessionContext,
    SessionSummary,
    UserLine,
)


def test_user_line_roundtrip_camel_case_alias():
    """jsonl 用 CC 的 camelCase 字段名(parentUuid/sessionId/gitBranch/isSidechain)。"""
    line = UserLine(
        uuid="u1",
        parent_uuid=None,
        session_id="s1",
        timestamp="2026-07-05T00:00:00Z",
        cwd="C:/proj",
        version="0.1.0",
        git_branch="main",
        message={"role": "user", "content": [{"type": "text", "text": "hi"}]},
    )
    dumped = line.model_dump(by_alias=True, exclude_none=True)
    # parent_uuid=None 被 exclude_none=True 排除(parentUuid 键不出现)
    assert "parentUuid" not in dumped
    assert dumped["sessionId"] == "s1"
    assert dumped["gitBranch"] == "main"
    assert dumped["isSidechain"] is False
    # 反序列化(camelCase → snake_case by alias)
    again = UserLine.model_validate(dumped)
    assert again.session_id == "s1" and again.parent_uuid is None


def test_user_line_parent_uuid_optional():
    line = UserLine(
        uuid="u1", session_id="s1", timestamp="t", message={"role": "user", "content": []},
    )
    assert line.parent_uuid is None
    assert line.is_sidechain is False


def test_assistant_line_message_dict():
    line = AssistantLine(
        uuid="a1",
        parent_uuid="u1",
        session_id="s1",
        timestamp="t",
        message={
            "role": "assistant", "content": [],
            "usage": {"input_tokens": 5}, "stop_reason": "end_turn",
        },
    )
    assert line.message["usage"]["input_tokens"] == 5


def test_extra_fields_allowed_for_forward_compat():
    """未来加字段不破坏旧读取(前向兼容)。"""
    line = UserLine.model_validate(
        {
            "type": "user", "uuid": "u1", "sessionId": "s1", "timestamp": "t",
            "message": {"role": "user", "content": []},
            "futureField": "unknown",  # 未知字段
        }
    )
    assert line.uuid == "u1"


def test_session_context_dataclass():
    ctx = SessionContext(session_id="s1", cwd="C:/proj", version="0.1.0", git_branch="main")
    assert ctx.session_id == "s1" and ctx.git_branch == "main"


def test_persist_info_and_summary_shapes():
    pi = PersistInfo(llm_content="placeholder", path="/p/f.txt", size=100)
    assert pi.size == 100
    # SessionSummary 带 sidecar 摘要(first_user/turn_count 来自 <sid>.meta,不再扫 jsonl)。
    s = SessionSummary(
        session_id="sid", mtime=1.0, first_user_summary="hi", turn_count=2,
    )
    assert s.first_user_summary == "hi"
    assert s.turn_count == 2


def test_agent_tool_use_result_roundtrip_camel_case():
    from birdcode.session.models import AgentToolUseResult

    r = AgentToolUseResult(
        agent_id="a0685194cbfe9829e",
        agent_type="general-purpose",
        status="completed",
        is_async=False,
        prompt="review T7",
        resolved_model="claude-sonnet-5",
        content="done",
        total_duration_ms=1234,
        total_tokens=567,
    )
    dumped = r.model_dump(by_alias=True, exclude_none=True)
    assert dumped["agentId"] == "a0685194cbfe9829e"
    assert dumped["isAsync"] is False
    assert dumped["resolvedModel"] == "claude-sonnet-5"
    assert dumped["totalDurationMs"] == 1234
    # 反序列化(camelCase → snake_case)
    back = AgentToolUseResult.model_validate(dumped)
    assert back.agent_id == "a0685194cbfe9829e" and back.status == "completed"


def test_agent_tool_use_result_async_shape():
    from birdcode.session.models import AgentToolUseResult

    r = AgentToolUseResult(
        agent_id="a1", status="launched", is_async=True,
        output_file="/tmp/x.txt", can_read_output_file=True,
    )
    dumped = r.model_dump(by_alias=True, exclude_none=True)
    assert dumped["status"] == "launched"
    assert dumped["isAsync"] is True
    assert dumped["outputFile"] == "/tmp/x.txt"
    assert dumped["canReadOutputFile"] is True
    assert "content" not in dumped  # None 被 exclude_none 排除


def test_queue_operation_line_roundtrip():
    from birdcode.session.models import QueueOperationLine

    line = QueueOperationLine(
        session_id="s1", timestamp="t",
        operation="enqueue", agent_id="a1", tool_use_id="call_x",
        status="completed",
    )
    dumped = line.model_dump(by_alias=True, exclude_none=False)
    assert dumped["type"] == "queue-operation"
    assert dumped["operation"] == "enqueue"
    assert dumped["agentId"] == "a1"
    assert dumped["toolUseId"] == "call_x"
    back = QueueOperationLine.model_validate(dumped)
    assert back.operation == "enqueue" and back.agent_id == "a1"


def test_subagent_meta_roundtrip():
    from birdcode.session.models import SubagentMeta

    m = SubagentMeta(
        agent_id="a1", agent_type="general-purpose", description="Spec review T7",
        tool_use_id="call_x", spawn_depth=1, is_async=True, model="sonnet",
        resolved_model="claude-sonnet-5", status="completed",
        spawned_at="2026-07-08T00:00:00Z", completed_at="2026-07-08T00:01:00Z",
        total_duration_ms=1234, total_tokens=567,
    )
    dumped = m.model_dump(by_alias=True, exclude_none=True)
    assert dumped["agentId"] == "a1"
    assert dumped["isAsync"] is True
    assert dumped["spawnDepth"] == 1
    back = SubagentMeta.model_validate(dumped)
    assert back.status == "completed" and back.tool_use_id == "call_x"
