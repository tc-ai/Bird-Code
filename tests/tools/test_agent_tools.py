# tests/tools/test_agent_tools.py
"""_AgentTool(每 defn 一 tool)+ register_agent_tools 工厂单测。"""

import pytest

from birdcode.agents.definition import AgentDefinition
from birdcode.agents.manager import LaunchAck
from birdcode.agents.registry import AgentRegistry
from birdcode.agents.runner import SubagentReport
from birdcode.tools.agent_tools import _AgentTool, register_agent_tools
from birdcode.tools.registry import ToolRegistry


def _defn(**over):
    base = dict(name="general-purpose", description="通用", system_prompt="SP")
    base.update(over)
    return AgentDefinition(**base)


class _CapturingRunner:
    """替代 SubagentRunner:记录构造参数,run() 返回固定 report。"""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _CapturingRunner.last = self
        self.agent_id = "sub-fixed"
        self.sidechain_path = "/tmp/side-sub-fixed.jsonl"  # 异步分支读
        self.isolation = kwargs.get("isolation")  # 忠 SubagentRunner 接口

    async def run(self):
        return SubagentReport(
            is_completed=True,
            text="完成了",
            status="completed",
            duration_ms=10,
            tokens=5,
            tool_use_count=0,
        )


class _FakeSubagentManager:
    def __init__(self):
        self.launched_runner = None

    async def launch_async(self, runner):
        self.launched_runner = runner
        return LaunchAck(agent_id=runner.agent_id, ack_text="已派发异步子 agent,完成后通知。")


@pytest.fixture(autouse=True)
def _patch_runner(monkeypatch):
    from birdcode.tools import agent_tools

    monkeypatch.setattr(agent_tools, "SubagentRunner", _CapturingRunner)


def _tool(defn=None, **over):
    base = dict(
        defn=defn or _defn(),
        cfg=None,
        app=None,
        ctx=None,
        project_root=None,
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
    )
    base.update(over)
    return _AgentTool(**base)


async def test_sync_returns_tooloutput_with_result():
    out = await _tool(_defn(run_in_background=False)).execute(prompt="做 X")
    from birdcode.tools.base import ToolOutput

    assert isinstance(out, ToolOutput)
    assert "完成了" in out.text
    assert out.tool_use_result["isCompleted"] is True
    assert out.tool_use_result["agentType"] == "general-purpose"
    assert _CapturingRunner.last.kwargs["spawn_depth"] == 1  # 父 0 → 子 1
    assert _CapturingRunner.last.kwargs["model_override"] == ""  # D5 继承父


async def test_execute_plumbs_agent_id_and_tool_use_id_to_runner():
    """execute 的 agent_id/tool_use_id 透传 runner(替换占位 (sync-agent),补 plumb 缺口)。"""
    await _tool(_defn(run_in_background=False)).execute(
        prompt="做 X",
        agent_id="sub-injected123",
        tool_use_id="toolu_real",
    )
    assert _CapturingRunner.last.kwargs["agent_id"] == "sub-injected123"
    assert _CapturingRunner.last.kwargs["tool_use_id"] == "toolu_real"


async def test_async_without_mgr_returns_error():
    # 异步由 defn.run_in_background 决定(不进 schema、非 execute 参数)
    out = await _tool(_defn(run_in_background=True)).execute(prompt="x")
    assert isinstance(out, str)
    assert "异步" in out or "Phase" in out or "不支持" in out


async def test_async_with_mgr_returns_launch_ack():
    from birdcode.tools.base import ToolOutput

    mgr = _FakeSubagentManager()
    out = await _tool(_defn(run_in_background=True), subagent_mgr=mgr).execute(prompt="做 X")
    assert isinstance(out, ToolOutput)
    assert out.text == "已派发异步子 agent,完成后通知。"  # 回执 = fake ack_text
    tur = out.tool_use_result
    assert tur["status"] == "launched"
    assert tur["isAsync"] is True
    assert tur["agentType"] == "general-purpose"
    assert tur["agentId"] == "sub-fixed"
    assert "sub-fixed" in tur["outputFile"]
    assert mgr.launched_runner.kwargs["is_async"] is True
    # execute 不经 executor 时 tool_use_id 默认 "";真实场景 executor 透传真实 tu.id
    assert mgr.launched_runner.kwargs["tool_use_id"] == ""
    assert mgr.launched_runner.kwargs["model_override"] == ""


async def test_recursion_guard_at_max_depth():
    out = await _tool(spawn_depth=1).execute(prompt="x")
    assert isinstance(out, str)
    assert "派生深度" in out or "递归" in out


def test_tool_attrs_from_defn():
    """name/description/kind/parallel_safe/run_in_background 取自 defn + is_agent_tool 标记。"""
    t = _tool(_defn(name="explore", description="搜索", kind="read", parallel_safe=True))
    assert t.name == "explore"
    assert t.description == "搜索"
    assert t.kind == "read"
    assert t.parallel_safe is True
    assert t.is_agent_tool is True
    # run_in_background 控制字段取自 defn(不进 schema);显式设,不依赖 defn 默认值
    assert _tool(_defn(run_in_background=False))._run_in_background is False  # noqa: SLF001
    assert _tool(_defn(run_in_background=True))._run_in_background is True  # noqa: SLF001


def test_register_agent_tools_one_per_defn():
    """工厂:每个 defn → 一个 tool,kind/ps 按来源;is_agent_tool 标记。"""
    agents = AgentRegistry()
    agents.register(_defn(name="explore", description="d", kind="read", parallel_safe=True))
    agents.register(_defn(name="general-purpose", description="d"))
    reg = ToolRegistry()
    register_agent_tools(
        reg,
        agents,
        cfg=None,
        app=None,
        ctx=None,
        project_root=None,
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
    )
    assert {"explore", "general-purpose"} <= set(reg.names())
    assert reg.get("explore").kind == "read"
    assert reg.get("explore").parallel_safe is True
    assert reg.get("general-purpose").kind == "write"  # defn 默认
    assert reg.get("general-purpose").parallel_safe is False
    assert reg.get("explore").is_agent_tool is True


def test_rebind_updates_ctx_and_provider():
    from birdcode.session.models import SessionContext

    tool = _tool()
    assert tool._ctx is None  # noqa: SLF001
    assert tool._parent_provider is None  # noqa: SLF001

    class _P:
        profile = "prof-x"

    new_ctx = SessionContext(session_id="new-sid", cwd=".", version="", git_branch=None)
    tool.rebind(ctx=new_ctx, provider=_P())
    assert tool._ctx is new_ctx  # noqa: SLF001
    assert tool._parent_provider.profile == "prof-x"  # noqa: SLF001
    tool.rebind()  # None 参数 = 该项不变
    assert tool._ctx is new_ctx  # noqa: SLF001
