# tests/tools/test_tool_output.py
from pydantic import BaseModel

from birdcode.blocks import ToolUseBlock
from birdcode.tools.base import Tool, ToolOutput
from birdcode.tools.executor import ToolExecutor
from birdcode.tools.registry import ToolRegistry


class _OutInput(BaseModel):
    pass


class _StructuredTool(Tool):
    name = "structured_dummy"
    description = "returns ToolOutput"
    parameters = _OutInput
    kind = "read"

    async def execute(self):  # type: ignore[override]
        return ToolOutput(text="hi", tool_use_result={"agentId": "a1", "status": "completed"})


class _PlainTool(Tool):
    name = "plain_dummy"
    description = "returns str"
    parameters = _OutInput
    kind = "read"

    async def execute(self):  # type: ignore[override]
        return "plain"


async def test_tooloutput_populates_tool_use_result():
    reg = ToolRegistry()
    reg.register(_StructuredTool())
    ex = ToolExecutor(reg)
    res = await ex._exec_one(ToolUseBlock(id="c1", name="structured_dummy", input={}))
    assert res.ok is True
    assert res.llm_content == "hi"
    assert res.tool_use_result == {"agentId": "a1", "status": "completed"}


async def test_str_return_leaves_tool_use_result_none():
    reg = ToolRegistry()
    reg.register(_PlainTool())
    ex = ToolExecutor(reg)
    res = await ex._exec_one(ToolUseBlock(id="c2", name="plain_dummy", input={}))
    assert res.ok is True
    assert res.llm_content == "plain"
    assert res.tool_use_result is None
