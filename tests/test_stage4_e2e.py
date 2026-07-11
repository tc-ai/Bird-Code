# tests/test_stage4_e2e.py
"""stage4 端到端验收(spec §10 硬验收聚合)。"""

import json

import pytest

from birdcode.agent.agent_loop import run_agent_loop
from birdcode.agent.provider import (
    Done,
    EditDiff,
    TokenUsage,
    ToolCallStart,
)
from birdcode.blocks import TextBlock, ToolResultBlock
from birdcode.conversation import Message, Turn
from birdcode.permission.gate import UiPermissionGate
from birdcode.tools.executor import ToolExecutor
from birdcode.tools.registry import ToolRegistry
from birdcode.tools.write_tool import WriteTool


class _ScriptProvider:
    def __init__(self, scripts):
        self._scripts = list(scripts)

    async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
        for e in self._scripts.pop(0):
            yield e


def _gate(tmp_path, *, mode, modal_result):
    """构造 UiPermissionGate:project_root=tmp_path(沙箱限定其内)。"""

    async def get_mode():
        return mode

    async def request_permission(tool_name, summary, path):  # type: ignore[no-untyped-def]
        return modal_result

    return UiPermissionGate(
        get_mode=get_mode,
        request_permission=request_permission,
        project_root=tmp_path,
    )


def _executor(tmp_path, *, mode, modal_result):
    r = ToolRegistry()
    r.register(WriteTool())
    gate = _gate(tmp_path, mode=mode, modal_result=modal_result)
    return ToolExecutor(r, permission_gate=gate)


async def _run(provider, executor):
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="go")])])
    events: list = []

    async def emit(ev):
        events.append(ev)

    await run_agent_loop(provider=provider, turn=turn, history=[], executor=executor, emit=emit)
    return turn, events


@pytest.mark.asyncio
async def test_plan_mode_blocks_write(tmp_path):
    """plan 模式 → Write 被拒,文件未创建,tool_result is_error=True。"""
    target = tmp_path / "f.txt"
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
    ex = _executor(tmp_path, mode="plan", modal_result="approve")  # plan 优先,不弹
    turn, events = await _run(p, ex)
    assert not target.exists()  # 被拒,未写
    tr = next(b for b in turn.messages[2].content if isinstance(b, ToolResultBlock))
    assert tr.is_error is True
    assert "拒绝" in tr.content
    assert not any(isinstance(e, EditDiff) for e in events)


@pytest.mark.asyncio
async def test_accept_edits_write_auto_approved_and_diff_emitted(tmp_path):
    """accept-edits + Write → 自动通过,文件写入,EditDiff emit。"""
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
    ex = _executor(tmp_path, mode="accept-edits", modal_result="reject")  # 不应弹
    turn, events = await _run(p, ex)
    assert target.read_text(encoding="utf-8") == "hello\n"
    diffs = [e for e in events if isinstance(e, EditDiff)]
    assert len(diffs) == 1 and diffs[0].id == "t1"


@pytest.mark.asyncio
async def test_pydantic_validation_failure_still_toolresult_no_exception(tmp_path):
    """缺 content → Pydantic 校验失败 → ToolResult(ok=False),不抛异常(stage1 硬验收继承)。"""
    p = _ScriptProvider(
        [
            [
                ToolCallStart(
                    id="t1",
                    name="write_file",
                    args=json.dumps({"file_path": str(tmp_path / "f.txt")}),  # 缺 content
                ),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [Done(usage=TokenUsage())],
        ]
    )
    ex = _executor(tmp_path, mode="bypass", modal_result="approve")
    turn, events = await _run(p, ex)
    tr = next(b for b in turn.messages[2].content if isinstance(b, ToolResultBlock))
    assert tr.is_error is True
    assert "校验" in tr.content
