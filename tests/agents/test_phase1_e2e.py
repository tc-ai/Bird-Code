# tests/agents/test_phase1_e2e.py
"""Phase1 端到端:_AgentTool → SubagentRunner → 报告 → ToolOutput,全链(mock provider)。"""
import pytest

from birdcode.agent.provider import Done, TextDelta, TokenUsage
from birdcode.agents.definition import AgentDefinition
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.session.models import SessionContext
from birdcode.tools.agent_tools import _AgentTool
from birdcode.tools.base import ToolOutput


class _MockChildProvider:
    def __init__(self, profile):
        self.profile = profile

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            yield TextDelta(text="已探索完毕,结论:X")
            yield Done(usage=TokenUsage(input_tokens=12, output_tokens=8), stop_reason="end_turn")

        return _gen()


def _cfg():
    return AppConfig(
        providers={"p": ProviderProfile(name="p", protocol="anthropic", model="m",
                                        base_url="http://x", api_key="k")},
        default="p",
    )


@pytest.fixture
def agent_tool(monkeypatch, tmp_path):
    from birdcode.agents import runner
    from birdcode.session import paths

    monkeypatch.setattr(paths, "default_root", lambda: tmp_path)
    monkeypatch.setattr(
        runner, "build_provider",
        lambda profile, app, *, registry=None, system_override=None, mcp_instructions=None: (  # noqa: ARG005
            _MockChildProvider(profile)
        ),
    )
    defn = AgentDefinition(
        name="explore", description="只读探索", system_prompt="SP", run_in_background=False
    )
    return _AgentTool(
        defn=defn, cfg=_cfg(), app=None,
        ctx=SessionContext(session_id="s1", cwd=".", version="", git_branch=None),
        project_root=tmp_path, parent_provider=_MockChildProvider(_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, spawn_depth=0,
    )


async def test_e2e_sync_agent(agent_tool):
    # _AgentTool 内部 SubagentRunner 用 root=paths.default_root();此处仅验证返回 + 结构,
    # 侧链路径用 runner.sidechain_path(已在 Task 9 单测覆盖落盘)。本测试聚焦 tool_result 形状。
    out = await agent_tool.execute(prompt="探索 X 在哪")
    assert isinstance(out, ToolOutput)
    assert "已探索完毕" in out.text
    tur = out.tool_use_result
    assert tur["agentType"] == "explore"
    assert tur["status"] == "completed"
    assert tur["isCompleted"] is True
    assert tur["totalTokens"] == 20  # 12 + 8
    assert tur["content"] == "已探索完毕,结论:X"
