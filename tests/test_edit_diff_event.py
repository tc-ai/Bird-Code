# tests/test_edit_diff_event.py
import json

import pytest

from birdcode.agent.agent_loop import run_agent_loop
from birdcode.agent.provider import (
    Done,
    EditDiff,
    TokenUsage,
    ToolCallStart,
)
from birdcode.blocks import TextBlock
from birdcode.conversation import Message, Turn
from birdcode.permission.verdict import Verdict
from birdcode.tools.executor import ToolExecutor
from birdcode.tools.registry import ToolRegistry
from birdcode.tools.write_tool import WriteTool


class _AllowGate:
    async def check(self, tool, args):  # type: ignore[no-untyped-def]
        return Verdict("allow", "", "")


def _executor(tmp_path):
    r = ToolRegistry()
    r.register(WriteTool())
    return ToolExecutor(r, permission_gate=_AllowGate())


class _ScriptProvider:
    def __init__(self, scripts):
        self._scripts = list(scripts)

    async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
        for e in self._scripts.pop(0):
            yield e


@pytest.mark.asyncio
async def test_edit_diff_emitted_on_write_success(tmp_path):
    target = tmp_path / "f.txt"
    p = _ScriptProvider(
        [
            [
                ToolCallStart(
                    id="t1",
                    name="write_file",
                    args=json.dumps({"file_path": str(target), "content": "hello\n"}),
                ),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [Done(usage=TokenUsage())],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="write it")])])

    events: list = []

    async def emit(ev):
        events.append(ev)

    await run_agent_loop(provider=p, turn=turn, history=[], executor=_executor(tmp_path), emit=emit)
    diffs = [e for e in events if isinstance(e, EditDiff)]
    assert len(diffs) == 1
    assert diffs[0].id == "t1"
    assert diffs[0].new == "hello\n"  # 写入的新内容
    assert diffs[0].old == ""  # 新文件,old 为空


@pytest.mark.asyncio
async def test_no_edit_diff_when_write_blocked(tmp_path):
    """gate deny → 不 emit EditDiff(没改动)。"""
    target = tmp_path / "f.txt"

    class _DenyGate:
        async def check(self, tool, args):  # type: ignore[no-untyped-def]
            return Verdict("deny", "", "")

    r = ToolRegistry()
    r.register(WriteTool())
    ex = ToolExecutor(r, permission_gate=_DenyGate())
    p = _ScriptProvider(
        [
            [
                ToolCallStart(
                    id="t1",
                    name="write_file",
                    args=json.dumps({"file_path": str(target), "content": "x"}),
                ),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [Done(usage=TokenUsage())],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])
    events: list = []

    async def emit(ev):
        events.append(ev)

    await run_agent_loop(provider=p, turn=turn, history=[], executor=ex, emit=emit)
    assert not any(isinstance(e, EditDiff) for e in events)
