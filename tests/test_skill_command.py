"""Skill 斜杠命令注册测试(TDD:RED→GREEN)。"""

import pytest

from birdcode.agents.definition import AgentDefinition
from birdcode.agents.registry import AgentRegistry
from birdcode.ui.commands.command import CommandType
from birdcode.ui.commands.registry import CommandConflictError, CommandRegistry
from birdcode.ui.commands.skill_commands import register_skill_commands


class FakeCtx:
    """最小 CommandContext 桩:记录 send_user_message / show_message。"""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.shown: list[str] = []

    async def send_user_message(self, text: str) -> None:
        self.sent.append(text)

    async def show_message(self, text: str, *, kind: str = "info") -> None:
        self.shown.append(text)


class FakeForkTool:
    """桩 _ForkSkillTool:invoke_async 记录 args。"""

    def __init__(self) -> None:
        self.called: list[str] = []

    async def invoke_async(self, args: str) -> str:
        self.called.append(args)
        return "started"


@pytest.mark.asyncio
async def test_inline_skill_handler_sends_rendered_body():
    defn = AgentDefinition(name="commit", description="d", system_prompt="",
                           mode="inline", is_transparent=True, body="do $ARGUMENTS")
    caps = AgentRegistry()
    caps.register(defn)
    cr = CommandRegistry()
    register_skill_commands(cr, caps, {})
    handler = cr.resolve("/commit").handler
    ctx = FakeCtx()
    await handler(ctx, "fix typo")
    assert ctx.sent == ["do fix typo"]


@pytest.mark.asyncio
async def test_fork_skill_handler_invokes_async_and_shows_ack():
    defn = AgentDefinition(name="big", description="d", system_prompt="",
                           mode="fork", is_transparent=True, body="refactor $ARGUMENTS")
    caps = AgentRegistry()
    caps.register(defn)
    fake_tool = FakeForkTool()
    cr = CommandRegistry()
    register_skill_commands(cr, caps, {"big": fake_tool})
    handler = cr.resolve("/big").handler
    ctx = FakeCtx()
    await handler(ctx, "module X")
    assert fake_tool.called == ["module X"]
    assert ctx.shown == ["started"]


def test_register_only_transparent():
    agent_defn = AgentDefinition(name="hidden", description="d", system_prompt="ap",
                                 is_transparent=False)
    caps = AgentRegistry()
    caps.register(agent_defn)
    cr = CommandRegistry()
    register_skill_commands(cr, caps, {})
    assert cr.resolve("/hidden") is None  # agent 不暴露斜杠命令


def test_command_type_is_prompt():
    defn = AgentDefinition(name="s", description="d", system_prompt="",
                           mode="inline", is_transparent=True, body="b")
    caps = AgentRegistry()
    caps.register(defn)
    cr = CommandRegistry()
    register_skill_commands(cr, caps, {})
    assert cr.resolve("/s").type == CommandType.PROMPT


def test_skill_command_name_conflict_raises():
    defn = AgentDefinition(name="review", description="d", system_prompt="",
                           mode="inline", is_transparent=True, body="b")
    caps = AgentRegistry()
    caps.register(defn)
    cr = CommandRegistry()
    # 先注册一个 /review(模拟 builtin)
    from birdcode.ui.commands.command import Command

    async def _stub(ctx, args): ...
    cr.register(Command(name="/review", description="x", usage="/review",
                        type=CommandType.PROMPT, handler=_stub))
    with pytest.raises(CommandConflictError):
        register_skill_commands(cr, caps, {})
