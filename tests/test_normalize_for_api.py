# tests/test_normalize_for_api.py
"""normalize_messages_for_api:flatten 层安全网。

保证发往 API 的 payload 角色严格交替 + 无孤儿 tool_result。这正是旧 review #1 漏网的
测试层——原测试止于 decoded turns,没人驱动 provider/flattening 层验证实际 payload
的交替约束(MockProvider 不强制交替,故连续 user 400 得以漏网上线)。
"""

from birdcode.agent.base_llm import normalize_messages_for_api
from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message


def _roles(msgs: list[Message]) -> list[str]:
    return [m.role for m in msgs]


def test_merges_consecutive_same_role():
    """连续两条 user → 合并(避免 Anthropic 400 角色必须交替,旧 #1 病根)。"""
    msgs = [
        Message(role="user", content=[TextBlock(text="a")]),
        Message(role="user", content=[TextBlock(text="b")]),
        Message(role="assistant", content=[TextBlock(text="c")]),
    ]
    out = normalize_messages_for_api(msgs)
    assert _roles(out) == ["user", "assistant"]
    # 合并后内容拼接、顺序保留
    assert out[0].content[0].text == "a"
    assert out[0].content[1].text == "b"


def test_strips_orphan_tool_result():
    """孤儿 tool_result(无配对 tool_use——其 assistant 行被 decode 跳过)→ 剥离。

    避免「tool_result 无 tool_use」→ 400(旧 #7 病根,旧 review 判「极窄」被证伪)。
    """
    msgs = [
        Message(role="user", content=[TextBlock(text="q")]),
        # user(tool_result X)但全表无 asst(tool_use X)——孤儿
        Message(role="user", content=[ToolResultBlock(tool_use_id="X", content="orphan")]),
        Message(role="assistant", content=[TextBlock(text="a")]),
    ]
    out = normalize_messages_for_api(msgs)
    # 孤儿行被剥空 → 整条跳过;余 user(q)+assistant(a),合并后交替
    assert _roles(out) == ["user", "assistant"]
    assert not any(isinstance(b, ToolResultBlock) for m in out for b in m.content)


def test_strips_orphan_tool_use():
    """中段孤儿 tool_use(assistant 有 tool_use、全表无配对 tool_result)→ 剥离 tool_use 块,
    保留同条 assistant 的 text。

    repair 只看末尾 Turn,够不到中段;normalize 在 flatten 层兜底。剥离后只剩 text,不致
    「tool_use 无 tool_result」→ 400。来源:decode 跳过损坏的 tool_result 行,其前 tool_use
    沦为中段孤儿。
    """
    msgs = [
        Message(role="user", content=[TextBlock(text="q")]),
        # tool_use X 全表无 tool_result——中段孤儿
        Message(
            role="assistant",
            content=[TextBlock(text="思考"), ToolUseBlock(id="X", name="echo", input={})],
        ),
        Message(role="assistant", content=[TextBlock(text="结论")]),
    ]
    out = normalize_messages_for_api(msgs)
    # 两条 assistant 合并;text 保留、tool_use 被剥
    assert _roles(out) == ["user", "assistant"]
    texts = [b.text for b in out[1].content if isinstance(b, TextBlock)]
    assert texts == ["思考", "结论"]
    assert not any(isinstance(b, ToolUseBlock) for m in out for b in m.content)


def test_orphan_tool_use_only_block_drops_message():
    """assistant 整条只有孤儿 tool_use → 剥空后整条丢弃(不留空 content 消息,否则 400)。"""
    msgs = [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", content=[ToolUseBlock(id="X", name="echo", input={})]),
        Message(role="assistant", content=[TextBlock(text="结论")]),
    ]
    out = normalize_messages_for_api(msgs)
    # 中间空 assistant 被丢;user(q) + assistant(结论)
    assert _roles(out) == ["user", "assistant"]
    assert not any(isinstance(b, ToolUseBlock) for m in out for b in m.content)


def test_normal_payload_unchanged():
    """合法交替 + 配对 → normalize 不改结构(幂等)。"""
    msgs = [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="echo", input={})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="ok")]),
        Message(role="assistant", content=[TextBlock(text="a")]),
    ]
    out = normalize_messages_for_api(msgs)
    assert _roles(out) == ["user", "assistant", "user", "assistant"]
    # tool_use ↔ tool_result 配对保留
    assert any(isinstance(b, ToolUseBlock) for b in out[1].content)
    assert any(isinstance(b, ToolResultBlock) for b in out[2].content)


def test_repaired_history_flattens_to_alternating():
    """端到端缺口填补:decode(repaired turns) → flatten(如 base_llm)→ normalize
    → 角色严格交替。模拟「崩溃留悬空 tool_use → resume 修复 → 下轮新提问」全链。
    """
    from birdcode.session import codec
    from birdcode.session.models import SessionContext

    ctx = SessionContext(session_id="s", cwd="C:/p", version="0.1", git_branch=None)
    lines = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="问")]),
            uuid="u",
            parent_uuid=None,
            ctx=ctx,
            timestamp="t",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[ToolUseBlock(id="c", name="echo", input={})]),
            uuid="a",
            parent_uuid="u",
            ctx=ctx,
            timestamp="t",
        ),
    ]
    turns = codec.decode_lines_to_turns(lines)  # repair 补 tool_result + 合成 assistant
    # 模拟 base_llm.stream 展平 prior + 下轮新提问(current)
    flat_in = [m for t in turns for m in t.messages] + [
        Message(role="user", content=[TextBlock(text="下一轮提问")])
    ]
    out = normalize_messages_for_api(flat_in)
    # 严格交替
    assert out[0].role == "user"
    for i in range(1, len(out)):
        assert out[i].role != out[i - 1].role, f"位置 {i} 角色未交替:{_roles(out)}"
