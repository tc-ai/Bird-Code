"""启动接线组合测试(T6):验证 skill 同时注册为 tool 与斜杠命令。

不挂载完整 App——直接用桩 deps 调 build_capability_registry(T3) +
register_agent_tools + register_skill_tools(T4) + register_skill_commands(T5),
断言一个 inline skill 既进 _tool_registry 又进 _command_registry。
"""

from pathlib import Path

from birdcode.agents.builtins import build_capability_registry
from birdcode.agents.registry import AgentRegistry  # noqa: F401  (保 import 路径一致)
from birdcode.tools.agent_tools import register_agent_tools, register_skill_tools
from birdcode.tools.registry import ToolRegistry
from birdcode.ui.commands.registry import CommandRegistry
from birdcode.ui.commands.skill_commands import register_skill_commands


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_skill_appears_as_tool_and_command(tmp_path):
    agent_dir = tmp_path / "agents"
    skill_dir = tmp_path / "skill"
    _write(
        skill_dir / "commit.md",
        "---\nname: commit\ndescription: make commit\nmode: inline\n---\n$ARGUMENTS",
    )
    caps = build_capability_registry(
        tmp_path,
        project_skill_dir=skill_dir,
        _project_agent_dir=agent_dir,
    )
    tools = ToolRegistry()
    common = dict(
        cfg=None,
        app=None,
        ctx=None,
        project_root=None,
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
        progress_cb=None,
        subagent_mgr=None,
    )
    register_agent_tools(tools, caps, **common)
    skill_tools = register_skill_tools(tools, caps, **common)
    cmds = CommandRegistry()
    register_skill_commands(cmds, caps, skill_tools)

    assert tools.get("commit") is not None  # skill 注册成 tool
    assert cmds.resolve("/commit") is not None  # skill 注册成斜杠命令
    assert "commit" in skill_tools
