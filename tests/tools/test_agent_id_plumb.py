# tests/tools/test_executor.py
"""ToolExecutor.annotate_tool_uses:落盘前给 agent tool_use 注入 agent_id(就地改 tu)。"""

from __future__ import annotations

from pydantic import BaseModel

from birdcode.blocks import ToolUseBlock
from birdcode.tools.base import Tool
from birdcode.tools.executor import ToolExecutor
from birdcode.tools.registry import ToolRegistry


class _AgentToolStub(Tool):
    """is_agent_tool=True(模拟 _AgentTool);input 无 agent_id → annotate 生成 sub-xxx。"""

    name = "explore"
    is_agent_tool = True

    async def execute(self, **args: object) -> str:
        return ""


class _ResumeAgentToolStub(Tool):
    """resume_agent:is_agent_tool=True,input 含 agent_id(被续跑的)→ annotate 复制。"""

    name = "resume_agent"
    is_agent_tool = True

    async def execute(self, **args: object) -> str:
        return ""


class _NormalToolStub(Tool):
    """普通工具:is_agent_tool=False → annotate 不注入。"""

    name = "read_file"
    is_agent_tool = False

    async def execute(self, **args: object) -> str:
        return ""


def test_annotate_generates_agent_id_for_agent_tool():
    reg = ToolRegistry()
    reg.register(_AgentToolStub())
    tu = ToolUseBlock(id="t1", name="explore", input={"prompt": "x"})
    ToolExecutor(reg).annotate_tool_uses([tu])
    assert tu.agent_id is not None and tu.agent_id.startswith("sub-")


def test_annotate_copies_agent_id_for_resume_agent():
    reg = ToolRegistry()
    reg.register(_ResumeAgentToolStub())
    tu = ToolUseBlock(
        id="t1", name="resume_agent", input={"agent_id": "sub-old123", "direction": "go"}
    )
    ToolExecutor(reg).annotate_tool_uses([tu])
    assert tu.agent_id == "sub-old123"  # 复制被续跑的,不新生成


def test_annotate_skips_normal_tools():
    reg = ToolRegistry()
    reg.register(_NormalToolStub())
    tu = ToolUseBlock(id="t1", name="read_file", input={})
    ToolExecutor(reg).annotate_tool_uses([tu])
    assert tu.agent_id is None


def test_annotate_skips_unknown_tool():
    tu = ToolUseBlock(id="t1", name="nonexistent", input={})
    ToolExecutor(ToolRegistry()).annotate_tool_uses([tu])
    assert tu.agent_id is None


class _AgentInput(BaseModel):
    prompt: str = ""


class _CapturingAgentTool(Tool):
    """is_agent_tool=True + 捕获 execute 参数(验证 _exec_one 透传 agent_id/tool_use_id)。"""

    name = "explore"
    is_agent_tool = True
    parameters = _AgentInput
    captured: dict[str, object] = {}

    async def execute(self, **args: object) -> str:
        _CapturingAgentTool.captured = dict(args)
        return "ok"


async def test_exec_one_plumbs_agent_id_and_tool_use_id_to_agent_tool():
    """_exec_one 对 is_agent_tool 工具透传 tu.agent_id + tu.id 到 execute(补占位 plumb)。"""
    reg = ToolRegistry()
    reg.register(_CapturingAgentTool())
    tu = ToolUseBlock(id="toolu_real", name="explore", input={"prompt": "x"}, agent_id="sub-inj")
    await ToolExecutor(reg)._exec_one(tu)
    assert _CapturingAgentTool.captured.get("agent_id") == "sub-inj"
    assert _CapturingAgentTool.captured.get("tool_use_id") == "toolu_real"
