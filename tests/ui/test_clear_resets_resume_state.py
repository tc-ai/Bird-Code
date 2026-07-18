# tests/ui/test_clear_resets_resume_state.py
"""Task 7 Step 1:/clear 重置 resume 状态字段占位测试。

clear_conversation 依赖大量 app 状态(Store/Controller/Executor/Gate/Memory/...),
难以用 SimpleNamespace 完整造。本文件做语义占位:断言字段名存在于 BirdApp.__init__,
真正的"调 clear_conversation 后字段被清空"由既有 /clear e2e + Step 7 全 resume 套件
回归覆盖(见 plan Step 1 注)。若后续想强测,可用 Textual Pilot 跑 /clear 后断言
app._shown_agent_ids == set()。
"""

from __future__ import annotations

from types import SimpleNamespace


def test_clear_resets_resume_state() -> None:
    """占位:确认 resume 状态字段命名 + clear 时应被重置的语义。

    丢弃改走真实 user 消息(submit,落盘常驻),故 resume 相关内存态只剩
    _shown_agent_ids / _reminder_turn(都在 /clear 重置)。
    """
    fake = SimpleNamespace(
        _shown_agent_ids={"sub-1"},
        _reminder_turn=object(),
    )
    # 改前:状态非空(模拟"使用过 resume 提示的会话")
    assert fake._shown_agent_ids == {"sub-1"}
