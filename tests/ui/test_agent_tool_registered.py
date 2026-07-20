# tests/ui/test_agent_tool_registered.py
from birdcode.agents.builtins import BUILTIN_AGENTS
from birdcode.tools.agent_tools import _AgentTool


def test_agent_tool_importable():
    # 守护:_AgentTool 可构造,name/kind 来自 defn(接线在 app.on_mount,e2e 覆盖全链)
    defn = next(d for d in BUILTIN_AGENTS if d.name == "general-purpose")
    t = _AgentTool(
        defn=defn,
        cfg=None,
        app=None,
        ctx=None,
        project_root=None,
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
    )
    assert t.name == "general-purpose"
    assert t.kind == "write"
    assert t.is_agent_tool is True
