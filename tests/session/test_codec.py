# tests/session/test_codec.py
from birdcode.blocks import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message
from birdcode.session import codec
from birdcode.session.models import SessionContext


def _ctx():
    return SessionContext(session_id="s1", cwd="C:/p", version="0.1.0", git_branch="main")


def test_block_roundtrip_text():
    d = codec.block_to_dict(TextBlock(text="hi"))
    assert d == {"type": "text", "text": "hi"}
    assert isinstance(codec.block_from_dict(d), TextBlock)


def test_block_roundtrip_thinking_signature_preserved():
    """signature 必须原样保留(丢了下轮 Anthropic API 报 400)。"""
    d = codec.block_to_dict(ThinkingBlock(text="hm", signature="sig-123"))
    back = codec.block_from_dict(d)
    assert isinstance(back, ThinkingBlock)
    assert back.signature == "sig-123"


def test_block_roundtrip_tool_use_input_preserved():
    block = ToolUseBlock(id="c1", name="read_file", input={"file_path": "x.py", "n": 3})
    d = codec.block_to_dict(block)
    back = codec.block_from_dict(d)
    assert isinstance(back, ToolUseBlock)
    assert back.id == "c1" and back.input == {"file_path": "x.py", "n": 3}


def test_block_roundtrip_tool_result_is_error_and_placeholder_content():
    """占位文本(超阈时)作为 content 原样往返。"""
    placeholder = "<persisted-output>\nOutput too large...\n"
    d = codec.block_to_dict(ToolResultBlock(tool_use_id="c1", content=placeholder, is_error=False))
    back = codec.block_from_dict(d)
    assert isinstance(back, ToolResultBlock)
    assert back.tool_use_id == "c1" and back.content == placeholder and back.is_error is False


def test_message_roundtrip_user():
    msg = Message(role="user", content=[TextBlock(text="hello")])
    d = codec.message_to_dict(msg)
    assert d == {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    back = codec.message_from_dict(d)
    assert back is not None
    assert back.role == "user" and back.content[0].text == "hello"


def test_encode_user_camel_case_fields():
    msg = Message(role="user", content=[TextBlock(text="hi")])
    line = codec.encode_user(
        msg, uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="2026-07-05T00:00:00Z",
    )
    assert line["type"] == "user"
    assert line["parentUuid"] is None
    assert line["sessionId"] == "s1"
    assert line["gitBranch"] == "main"
    assert line["isSidechain"] is False
    assert line["message"]["content"][0]["text"] == "hi"


def test_encode_assistant_includes_usage_and_stop_reason():
    from birdcode.agent.provider import TokenUsage

    msg = Message(role="assistant", content=[TextBlock(text="ok")])
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    line = codec.encode_assistant(
        msg, uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t",
        usage=usage, stop_reason="end_turn",
    )
    assert line["type"] == "assistant"
    assert line["message"]["usage"]["input_tokens"] == 10
    assert line["message"]["stop_reason"] == "end_turn"


def test_decode_lines_to_turns_single_turn():
    """user(提问) → assistant → user(tool_result) → assistant(终) 归一个 Turn。"""
    lines = [
        codec.encode_user(Message(role="user", content=[TextBlock(text="问")]),
                           uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1"),
        codec.encode_assistant(
            Message(
                role="assistant",
                content=[ToolUseBlock(id="c1", name="echo", input={"text": "问"})],
            ),
            uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2",
        ),
        codec.encode_user(
            Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="问")]),
            uuid="u2", parent_uuid="a1", ctx=_ctx(), timestamp="t3",
        ),
        codec.encode_assistant(Message(role="assistant", content=[TextBlock(text="答")]),
                               uuid="a2", parent_uuid="u2", ctx=_ctx(), timestamp="t4"),
    ]
    turns = codec.decode_lines_to_turns(lines)
    assert len(turns) == 1
    assert len(turns[0].messages) == 4
    assert turns[0].messages[0].content[0].text == "问"


def test_decode_lines_to_turns_multiple_turns_boundary():
    """第二个含 TextBlock 的 user 行 = 新 Turn 边界。"""
    lines = [
        codec.encode_user(Message(role="user", content=[TextBlock(text="第一问")]),
                           uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1"),
        codec.encode_assistant(Message(role="assistant", content=[TextBlock(text="第一答")]),
                               uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2"),
        codec.encode_user(Message(role="user", content=[TextBlock(text="第二问")]),
                           uuid="u2", parent_uuid="a1", ctx=_ctx(), timestamp="t3"),
        codec.encode_assistant(Message(role="assistant", content=[TextBlock(text="第二答")]),
                               uuid="a2", parent_uuid="u2", ctx=_ctx(), timestamp="t4"),
    ]
    turns = codec.decode_lines_to_turns(lines)
    assert len(turns) == 2
    assert turns[0].messages[0].content[0].text == "第一问"
    assert turns[1].messages[0].content[0].text == "第二问"


def test_decode_lines_to_turns_sets_usage_from_last_assistant():
    from birdcode.agent.provider import TokenUsage

    lines = [
        codec.encode_user(Message(role="user", content=[TextBlock(text="问")]),
                           uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1"),
        codec.encode_assistant(Message(role="assistant", content=[TextBlock(text="答")]),
                               uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2",
                               usage=TokenUsage(input_tokens=7), stop_reason="end_turn"),
    ]
    turns = codec.decode_lines_to_turns(lines)
    assert turns[0].usage is not None and turns[0].usage.input_tokens == 7


def test_full_message_roundtrip_through_line():
    """完整往返:Message → encode → dict → decode → Message 无损(命根子)。"""
    original = Message(
        role="assistant",
        content=[
            ThinkingBlock(text="思考", signature="sig"),
            ToolUseBlock(id="c1", name="read_file", input={"file_path": "x.py"}),
            TextBlock(text="结果"),
        ],
    )
    line = codec.encode_assistant(original, uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t")
    from birdcode.session.models import AssistantLine

    parsed = AssistantLine.model_validate(line)
    back = codec.message_from_dict(parsed.message)
    assert back is not None
    assert back.role == "assistant"
    assert isinstance(back.content[0], ThinkingBlock) and back.content[0].signature == "sig"
    second = back.content[1]
    assert isinstance(second, ToolUseBlock) and second.input == {"file_path": "x.py"}
    assert isinstance(back.content[2], TextBlock) and back.content[2].text == "结果"


# ---- F6:未知 block type 容错 ----


def test_block_from_dict_unknown_type_returns_none():
    """F6:未知 type 返回 None(不抛),供 message_from_dict 过滤。"""
    assert codec.block_from_dict({"type": "unknown", "foo": "bar"}) is None


def test_message_from_dict_filters_none_blocks():
    """F6:message_from_dict 过滤掉 None(未知 block),保留已识别块。"""
    d = {
        "role": "user",
        "content": [
            {"type": "text", "text": "保留"},
            {"type": "unknown", "foo": "bar"},
            {"type": "text", "text": "也保留"},
        ],
    }
    msg = codec.message_from_dict(d)
    assert msg is not None
    assert msg.role == "user"
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextBlock) and msg.content[0].text == "保留"
    assert isinstance(msg.content[1], TextBlock) and msg.content[1].text == "也保留"


def test_block_to_dict_unknown_block_returns_text_placeholder():
    """F6:未知 ContentBlock → 占位 TextBlock(保护 live agent loop 不崩)。"""
    from dataclasses import dataclass

    @dataclass
    class FakeBlock:
        x: int = 1

    placeholder = codec.block_to_dict(FakeBlock())  # type: ignore[arg-type]
    assert placeholder["type"] == "text"
    assert "FakeBlock" in placeholder["text"]


# ---- F9:TokenUsage schema drift ----


def test_decode_lines_to_turns_tolerates_usage_extra_fields():
    """F9:usage dict 含未来额外字段 → model_validate 忽略(不抛 ValidationError)。"""
    lines = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="问")]),
            uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[TextBlock(text="答")]),
            uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2",
        ),
    ]
    # 注入未来字段(schema drift)
    lines[1]["message"]["usage"] = {
        "input_tokens": 10,
        "output_tokens": 5,
        "future_field": "未识别字段",
    }
    turns = codec.decode_lines_to_turns(lines)  # 不抛
    assert turns and turns[0].usage is not None and turns[0].usage.input_tokens == 10


# ---- F3:末尾 orphan user 提问丢弃 ----


def test_decode_completes_trailing_orphan_user_question():
    """末尾 orphan user(提问无 assistant 跟随)→ 补合成 assistant 完成它(不丢弃)。

    模拟进程在 user 提问落盘后、assistant 回复落盘前崩溃。旧 F3「丢弃」在 append-only
    下不安全:后续 append 会把 orphan 埋到中段,下轮 API 仍 400;改「补 assistant」可
    持久化(再 resume 稳定),且主线以 assistant 收尾。
    """
    lines = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="第一问")]),
            uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[TextBlock(text="第一答")]),
            uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2",
        ),
        # orphan user 提问(无 assistant 跟随)
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="第二问")]),
            uuid="u2", parent_uuid="a1", ctx=_ctx(), timestamp="t3",
        ),
    ]
    turns = codec.decode_lines_to_turns(lines)
    assert len(turns) == 2  # orphan Turn 保留并补全(非丢弃)
    assert turns[0].messages[0].content[0].text == "第一问"
    # 末尾 Turn = [第二问, 合成 assistant]
    assert turns[1].messages[0].content[0].text == "第二问"
    assert turns[1].messages[-1].role == "assistant"


def test_decode_completes_trailing_tool_result_user():
    """末尾 user(tool_result)(工具回填后崩溃、无后续 assistant)→ 补合成 assistant。

    tool_result user 保留(不丢);repair 统一为「末尾为 user → 补 assistant」收尾。
    """
    lines = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="问")]),
            uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[ToolUseBlock(id="c1", name="echo", input={})]),
            uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2",
        ),
        codec.encode_user(
            Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="ok")]),
            uuid="u2", parent_uuid="a1", ctx=_ctx(), timestamp="t3",
        ),
    ]
    turns = codec.decode_lines_to_turns(lines)
    assert len(turns) == 1
    msgs = turns[0].messages
    # tool_result user 保留(倒数第二),其后补合成 assistant
    assert isinstance(msgs[-2].content[0], ToolResultBlock)
    assert msgs[-1].role == "assistant"


# ---- F2:末尾悬空 tool_use 修复 ----


def test_decode_repairs_dangling_tool_use_with_error_result():
    """F2:末尾 assistant 含 tool_use、无配对 tool_result → 补 error tool_result user message。

    模拟进程在工具执行中崩溃:assistant(tool_use) 已落盘、tool_result 未落盘。
    """
    lines = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="问")]),
            uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[ToolUseBlock(id="c1", name="echo", input={})]),
            uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2",
        ),
        # 没有 tool_result 行(进程在工具执行中崩溃)
    ]
    turns = codec.decode_lines_to_turns(lines)
    assert len(turns) == 1
    msgs = turns[0].messages
    # 倒数第二:补的 error tool_result(user)
    tr_msg = msgs[-2]
    assert tr_msg.role == "user"
    tr = tr_msg.content[0]
    assert isinstance(tr, ToolResultBlock)
    assert tr.tool_use_id == "c1" and tr.is_error is True
    # 末尾:合成 assistant(主线以 assistant 收尾,#1)
    assert msgs[-1].role == "assistant"


def test_decode_no_repair_when_tool_use_already_paired():
    """F2 不应误补:turn 内 tool_use 已有配对 tool_result 时不补。"""
    lines = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="问")]),
            uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[ToolUseBlock(id="c1", name="echo", input={})]),
            uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2",
        ),
        codec.encode_user(
            Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="ok")]),
            uuid="u2", parent_uuid="a1", ctx=_ctx(), timestamp="t3",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[TextBlock(text="答")]),
            uuid="a2", parent_uuid="u2", ctx=_ctx(), timestamp="t4",
        ),
    ]
    turns = codec.decode_lines_to_turns(lines)
    assert len(turns) == 1
    # 末尾仍是 assistant(text),未被 F2 改动
    last_msg = turns[0].messages[-1]
    assert last_msg.role == "assistant"
    assert isinstance(last_msg.content[0], TextBlock)
    assert len(turns[0].messages) == 4


# ---- #1: repair 后主线恒以 assistant 收尾 ----


def test_repair_trailing_edge_ends_with_assistant():
    """#1:无论哪种中断尾部,repair 后末尾 Turn 最后一条必为 assistant。

    避免下轮 submit 展平出「连续两条 user」→ Anthropic 400 角色必须交替。
    """
    # case a:悬空 tool_use(进程在工具执行中崩溃)
    lines_a = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="q")]),
            uuid="u", parent_uuid=None, ctx=_ctx(), timestamp="t",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[ToolUseBlock(id="c", name="echo", input={})]),
            uuid="a", parent_uuid="u", ctx=_ctx(), timestamp="t",
        ),
    ]
    turns_a = codec.decode_lines_to_turns(lines_a)
    assert turns_a[-1].messages[-1].role == "assistant"

    # case b:orphan user 提问(进程在提问落盘后、回复前崩溃)
    lines_b = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="q")]),
            uuid="u", parent_uuid=None, ctx=_ctx(), timestamp="t",
        ),
    ]
    turns_b = codec.decode_lines_to_turns(lines_b)
    assert turns_b[-1].messages[-1].role == "assistant"

    # case c:正常完整 Turn(末尾已是 assistant)→ 不追加
    lines_c = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="q")]),
            uuid="u", parent_uuid=None, ctx=_ctx(), timestamp="t",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[TextBlock(text="a")]),
            uuid="a", parent_uuid="u", ctx=_ctx(), timestamp="t",
        ),
    ]
    turns_c = codec.decode_lines_to_turns(lines_c)
    assert turns_c[-1].messages[-1].role == "assistant"
    assert len(turns_c[0].messages) == 2  # 无合成追加


# ---- #3: 残缺行不杀 resume ----


def test_decode_skips_malformed_message_line():
    """#3:message 缺 text/role/content(类型错)的行被跳过,不杀整条 resume。"""
    lines = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="问")]),
            uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t1",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[TextBlock(text="答")]),
            uuid="a1", parent_uuid="u1", ctx=_ctx(), timestamp="t2",
        ),
        # 残缺:assistant 行的 text block 缺 text 字段
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text"}]}},
        # 残缺:user 行缺 role
        {"type": "user", "message": {"content": [{"type": "text", "text": "x"}]}},
        # 残缺:content 非列表
        {"type": "user", "message": {"role": "user", "content": "not-a-list"}},
    ]
    turns = codec.decode_lines_to_turns(lines)  # 不抛
    assert len(turns) == 1
    assert turns[0].messages[0].content[0].text == "问"
    assert turns[0].messages[-1].role == "assistant"


# ---- synthetic 标记往返(占位/收尾合成行可识别)----


def test_synthetic_flag_roundtrip():
    """合成消息(占位 tool_result / 收尾 assistant)的 synthetic 标记持久化往返:
    encode 写 synthetic=True、decode 回填到 Message.synthetic;普通消息默认 False。
    """
    syn = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="c1", content="中断补全", is_error=True)],
        synthetic=True,
    )
    line = codec.encode_user(syn, uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t")
    assert line["synthetic"] is True
    # decode 回填(decode_lines 纯解码,不触发 repair)
    turns = codec.decode_lines([line])
    assert turns[0].messages[0].synthetic is True

    # 普通消息 synthetic=False
    normal = codec.encode_user(
        Message(role="user", content=[TextBlock(text="hi")]),
        uuid="u2", parent_uuid=None, ctx=_ctx(), timestamp="t",
    )
    assert normal["synthetic"] is False


def test_encode_queue_operation_fields():
    from birdcode.session.models import SessionContext

    ctx = SessionContext(session_id="s1", cwd="C:/p", version="0.1.0", git_branch="main")
    line = codec.encode_queue_operation(
        ctx=ctx, timestamp="t",
        operation="enqueue", agent_id="a1", tool_use_id="call_x",
        status="completed",
    )
    assert line["type"] == "queue-operation"
    assert line["operation"] == "enqueue"
    assert line["agentId"] == "a1"
    assert line["toolUseId"] == "call_x"
    assert line["status"] == "completed"
    assert line["sessionId"] == "s1"


def test_encode_user_tool_use_result_dict_by_id():
    """sync/async 结果行:顶层 toolUseResult = dict 按 tool_use_id 索引(兼容批处理)。"""
    msg = Message(role="user", content=[ToolResultBlock(tool_use_id="call_x", content="ok")])
    line = codec.encode_user(
        msg, uuid="u1", parent_uuid="a1", ctx=_ctx(), timestamp="t",
        tool_use_results={"call_x": {"agentId": "a1", "status": "completed"}},
        source_tool_assistant_uuid="a1",
    )
    assert line["toolUseResult"] == {"call_x": {"agentId": "a1", "status": "completed"}}
    assert line["sourceToolAssistantUUID"] == "a1"


def test_encode_user_task_notification_and_agent_id():
    """§4.4② 注入行:isTaskNotification=true + 顶层 agentId(供 resume 匹配 enqueue)。"""
    msg = Message(role="user", content=[TextBlock(text="<task-notification>x</task-notification>")])
    line = codec.encode_user(
        msg, uuid="u1", parent_uuid="q1", ctx=_ctx(), timestamp="t",
        is_task_notification=True, agent_id="a1",
    )
    assert line["isTaskNotification"] is True
    assert line["agentId"] == "a1"


def test_encode_user_omits_new_fields_when_default():
    """普通 user 行不带新字段(无 null 噪声),向后兼容。"""
    msg = Message(role="user", content=[TextBlock(text="hi")])
    line = codec.encode_user(msg, uuid="u1", parent_uuid=None, ctx=_ctx(), timestamp="t")
    for k in ("toolUseResult", "sourceToolAssistantUUID", "isTaskNotification", "agentId"):
        assert k not in line


# ---- Task 8: find_pending_notifications ----


def test_find_pending_notifications():
    """enqueue 无匹配注入行 → pending;有匹配 → 已派发(不返回)。"""
    rows = [
        {"type": "queue-operation", "operation": "enqueue", "agentId": "a1", "toolUseId": "c1"},
        {"type": "queue-operation", "operation": "enqueue", "agentId": "a2", "toolUseId": "c2"},
        # a2 已被注入 → 不 pending;a1 无注入 → pending
        {"type": "user", "isTaskNotification": True, "agentId": "a2"},
    ]
    pending = codec.find_pending_notifications(rows)
    assert len(pending) == 1
    assert pending[0]["agentId"] == "a1"


def test_find_pending_notifications_all_injected_returns_empty():
    rows = [
        {"type": "queue-operation", "operation": "enqueue", "agentId": "a1"},
        {"type": "user", "isTaskNotification": True, "agentId": "a1"},
    ]
    assert codec.find_pending_notifications(rows) == []


def test_find_pending_notifications_ignores_non_enqueue():
    """dequeue/remove 不算 pending。"""
    rows = [
        {"type": "queue-operation", "operation": "dequeue", "agentId": "a1"},
        {"type": "queue-operation", "operation": "remove", "agentId": "a2"},
    ]
    assert codec.find_pending_notifications(rows) == []


def test_find_pending_notifications_filters_missing_agent_id():
    """CR #6b: 缺/None agentId 的坏 enqueue 行被过滤——返回项必带 truthy agentId,
    调用方 obj['agentId'] 安全;docstring 承诺「坏行安全跳过」。
    """
    rows = [
        {"type": "queue-operation", "operation": "enqueue", "agentId": "a1", "toolUseId": "c1"},
        {"type": "queue-operation", "operation": "enqueue", "toolUseId": "c2"},  # 缺 agentId
        {"type": "queue-operation", "operation": "enqueue", "agentId": None, "toolUseId": "c3"},
    ]
    pending = codec.find_pending_notifications(rows)
    assert len(pending) == 1
    assert pending[0]["agentId"] == "a1"


# ---- Phase2 F6a: find_pending dequeue-aware ----


def test_find_pending_excludes_dequeued():
    """enqueue 已被 dequeue 标记消费 → 不算 pending(投递补全不应再重构)。"""
    from birdcode.session.codec import find_pending_notifications

    rows = [
        {"type": "queue-operation", "operation": "enqueue", "agentId": "sub-1",
         "toolUseId": "t1", "status": "completed", "content": "..."},
        {"type": "queue-operation", "operation": "dequeue", "agentId": "sub-1",
         "toolUseId": "t1", "status": "completed", "content": ""},
    ]
    assert find_pending_notifications(rows) == []  # enqueue 有 dequeue → resolved


def test_find_pending_excludes_removed():
    """enqueue 已被 remove(取消)标记 → 不算 pending(F6a:remove 并入 resolved)。"""
    from birdcode.session.codec import find_pending_notifications

    rows = [
        {"type": "queue-operation", "operation": "enqueue", "agentId": "sub-2",
         "toolUseId": "t2", "status": "completed", "content": "..."},
        {"type": "queue-operation", "operation": "remove", "agentId": "sub-2",
         "toolUseId": "t2", "status": "cancelled", "content": ""},
    ]
    assert find_pending_notifications(rows) == []


def test_find_pending_dequeued_keeps_unmatched_enqueue():
    """F6a 回归:有 dequeue 但匹配的是别的 agentId → 本 enqueue 仍 pending。"""
    from birdcode.session.codec import find_pending_notifications

    rows = [
        {"type": "queue-operation", "operation": "enqueue", "agentId": "sub-1",
         "toolUseId": "t1", "status": "completed", "content": "..."},
        {"type": "queue-operation", "operation": "dequeue", "agentId": "sub-9",
         "toolUseId": "t9", "status": "completed", "content": ""},
    ]
    pending = find_pending_notifications(rows)
    assert len(pending) == 1
    assert pending[0]["agentId"] == "sub-1"


# ---- Phase2 F8: find_unresponded_notifications ----


def test_find_unresponded_notifications():
    """已投递 isTaskNotification user 行、无匹配 dequeue → 待响应(带 status 来自 enqueue)。"""
    from birdcode.session.codec import find_unresponded_notifications

    rows = [
        {"type": "queue-operation", "operation": "enqueue", "agentId": "sub-1",
         "toolUseId": "t1", "status": "completed", "content": "<task-notification>...A..."},
        {"type": "user", "isTaskNotification": True, "agentId": "sub-1",
         "message": {"role": "user", "content": [
             {"type": "text", "text": "<task-notification>...A..."},
         ]}},
    ]
    out = find_unresponded_notifications(rows)
    assert len(out) == 1
    assert out[0].agent_id == "sub-1"
    assert out[0].status == "completed"  # 取自 enqueue
    assert out[0].message.role == "user"


def test_find_unresponded_excludes_dequeued():
    """有匹配 dequeue → 已响应,不返回。"""
    from birdcode.session.codec import find_unresponded_notifications

    rows = [
        {"type": "queue-operation", "operation": "enqueue", "agentId": "sub-1",
         "toolUseId": "t1", "status": "completed", "content": "..."},
        {"type": "user", "isTaskNotification": True, "agentId": "sub-1",
         "message": {"role": "user", "content": [{"type": "text", "text": "..."}]}},
        {"type": "queue-operation", "operation": "dequeue", "agentId": "sub-1",
         "toolUseId": "t1", "status": "completed", "content": ""},
    ]
    assert find_unresponded_notifications(rows) == []


# ---- #10: is_task_notification 回填到 Message(resume 重放据此跳过)----


def test_line_to_message_backfills_is_task_notification():
    """#10:行级 isTaskNotification 回填到 Message.is_task_notification。

    resume 重放(_replay_history)据此跳过,不把 <task-notification> XML 当用户气泡渲染。
    """
    from birdcode.session.codec import _line_to_message

    row = {
        "type": "user",
        "isTaskNotification": True,
        "agentId": "a1",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "<task-notification>x</task-notification>"}],
        },
    }
    msg = _line_to_message(row)
    assert msg is not None
    assert msg.is_task_notification is True
    # 普通行(无 isTaskNotification)不带标记
    plain = _line_to_message(
        {"type": "user",
         "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}}
    )
    assert plain is not None
    assert plain.is_task_notification is False


def test_message_to_dict_omits_is_task_notification():
    """#10:is_task_notification 不进 message_to_dict(绝不发给 LLM,仅 UI/重放内部用)。"""
    from birdcode.blocks import TextBlock
    from birdcode.conversation import Message
    from birdcode.session.codec import message_to_dict

    msg = Message(role="user", content=[TextBlock(text="x")], is_task_notification=True)
    d = message_to_dict(msg)
    assert d == {"role": "user", "content": [{"type": "text", "text": "x"}]}
