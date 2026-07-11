# tests/test_mock_provider.py
import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.agent.provider import Done, TextDelta, ToolCallStart
from birdcode.blocks import TextBlock, ToolResultBlock
from birdcode.conversation import Message


@pytest.mark.asyncio
async def test_mock_first_phase_emits_tool_use():
    """第一阶段：回显 + 发起 echo 工具调用，Done(stop_reason=tool_use)。"""
    p = MockProvider(delay=0.0)
    user_msg = [Message(role="user", content=[TextBlock("hello world")])]
    events = [e async for e in p.stream(user_msg, history=[])]
    types = [e.type for e in events]
    assert "text" in types
    assert "tool_start" in types
    assert isinstance(events[-1], Done)
    assert events[-1].stop_reason == "tool_use"
    assert isinstance(events[-2], ToolCallStart)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "回显" in text
    # 第一阶段不再自带 tool_result —— 由 agent_loop+executor 回填
    assert "tool_result" not in types


@pytest.mark.asyncio
async def test_mock_second_phase_finalizes_after_tool_result():
    """第二阶段（末条消息是 tool_result）：流式回显用户原文 + Done(end_turn)。"""
    p = MockProvider(delay=0.0)
    messages = [
        Message(role="user", content=[TextBlock("hello world")]),
        Message(role="assistant", content=[]),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="mock-1", content="echoed 11 chars")],
        ),
    ]
    events = [e async for e in p.stream(messages, history=[])]
    types = [e.type for e in events]
    assert "tool_start" not in types  # 不再发起工具调用
    assert isinstance(events[-1], Done)
    assert events[-1].stop_reason is None  # end_turn（默认）
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "hello" in text or "world" in text


@pytest.mark.asyncio
async def test_mock_done_has_usage():
    p = MockProvider(delay=0.0)
    events = [
        e async for e in p.stream([Message(role="user", content=[TextBlock("hi")])], history=[])
    ]
    done = events[-1]
    assert isinstance(done, Done) and done.usage is not None
    assert done.usage.input_tokens > 0
