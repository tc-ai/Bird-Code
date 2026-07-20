# tests/agent/test_agent_loop.py
"""agent_loop._esc_interrupt_error_message:ESC 中断 tool_result 兜底文案分支。"""

from __future__ import annotations

from birdcode.agent.agent_loop import _esc_interrupt_error_message
from birdcode.blocks import ToolUseBlock


def test_esc_interrupt_msg_agent_tool_carries_agent_id():
    """agent tool(tu.agent_id 非空)→ 兜底带 agent_id + 续跑提示(模型驱动 resume_agent)。"""
    tu = ToolUseBlock(id="t1", name="explore", input={}, agent_id="sub-A")
    msg = _esc_interrupt_error_message(tu)
    assert "sub-A" in msg and "resume_agent" in msg and "用户同意" in msg


def test_esc_interrupt_msg_normal_tool_keeps_verify_message():
    """普通工具(tu.agent_id None)→ 保持「核查状态」文案(bash 非幂等,防盲目重试)。"""
    tu = ToolUseBlock(id="t1", name="bash", input={})
    msg = _esc_interrupt_error_message(tu)
    assert "用户中断(Esc)" in msg and "核查" in msg
    assert "resume_agent" not in msg
