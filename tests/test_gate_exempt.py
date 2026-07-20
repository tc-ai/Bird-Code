# tests/test_gate_exempt.py
"""S2/#3:gate_exempt 工具跳过权限 gate(team 工具全内存操作,无 FS/命令副作用,安全)。"""

from pydantic import BaseModel

from birdcode.blocks import ToolUseBlock
from birdcode.permission.verdict import Verdict
from birdcode.tools.base import Tool
from birdcode.tools.executor import ToolExecutor
from birdcode.tools.registry import ToolRegistry


class _NoArgs(BaseModel):
    pass


class _RejectGate:
    """恒拒 gate(模拟 fork_async L5 对非 bash write 的 reject)。"""

    async def check(self, tool, args):  # noqa: ARG002
        return Verdict("deny", "测试恒拒", "L5")


class _ExemptTool(Tool):
    name = "exempt_tool"
    description = "d"
    parameters = _NoArgs
    kind = "write"
    parallel_safe = False
    gate_exempt = True

    async def execute(self):  # noqa: ARG002
        return "exempt-ok"


class _NormalTool(Tool):
    name = "normal_tool"
    description = "d"
    parameters = _NoArgs
    kind = "write"
    parallel_safe = False

    async def execute(self):  # noqa: ARG002
        return "normal-ok"


async def test_gate_exempt_tool_bypasses_permission_gate():
    """gate_exempt=True → executor 跳过 gate.check(恒拒 gate 下仍执行);普通工具仍被拦。"""
    reg = ToolRegistry()
    reg.register(_ExemptTool())
    reg.register(_NormalTool())
    ex = ToolExecutor(reg, permission_gate=_RejectGate())

    r1 = await ex.execute_batch([ToolUseBlock(id="t1", name="exempt_tool", input={})])
    assert r1[0].ok, "gate_exempt 工具应跳过 gate"
    assert r1[0].output == "exempt-ok"

    r2 = await ex.execute_batch([ToolUseBlock(id="t2", name="normal_tool", input={})])
    assert not r2[0].ok, "普通 write 工具应被恒拒 gate 拦"
