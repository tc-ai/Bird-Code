"""透明 skill 的斜杠命令注册(启动期,复用 CommandRegistry)。

inline → CommandType.PROMPT,handler 渲染 body 后 send_user_message(注入点 B,同 /review)。
fork → handler 调 _ForkSkillTool.invoke_async(始终异步,不阻塞 UI),摘要经 wake 回流。
CommandContext 不暴露 subagent 依赖 → fork handler 关闭 tool 实例(其持全部 deps)。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from birdcode.agents.definition import AgentDefinition, render_body
from birdcode.agents.registry import AgentRegistry
from birdcode.ui.commands.command import Command, CommandContext, CommandType
from birdcode.ui.commands.registry import CommandRegistry


def _inline_skill_handler_factory(
    defn: AgentDefinition,
) -> Callable[[CommandContext, str], Awaitable[None]]:
    async def _handler(ctx: CommandContext, args: str) -> None:
        await ctx.send_user_message(render_body(defn, args))

    return _handler


def _fork_skill_handler_factory(tool: object) -> Callable[[CommandContext, str], Awaitable[None]]:
    async def _handler(ctx: CommandContext, args: str) -> None:
        ack = await tool.invoke_async(args)  # type: ignore[attr-defined]
        await ctx.show_message(ack)

    return _handler


def register_skill_commands(
    cmd_registry: CommandRegistry,
    caps: AgentRegistry,
    skill_tools: Mapping[str, object],
) -> None:
    """对每个透明 skill 注册 /<name> 命令。agent(非透明)不注册。
    skill_tools = register_skill_tools 的返回值;fork handler 关闭对应 tool。

    注意: skill_tools 必须包含每个透明 fork defn 的入口(索引 skill_tools[defn.name])。
    """
    for defn in caps.all():
        if not defn.is_transparent:
            continue
        if defn.mode == "inline":
            handler = _inline_skill_handler_factory(defn)
        else:  # fork
            handler = _fork_skill_handler_factory(skill_tools[defn.name])
        cmd_registry.register(
            Command(
                name=f"/{defn.name}",
                description=defn.description,
                usage=f"/{defn.name} $ARGUMENTS",
                type=CommandType.PROMPT,
                handler=handler,
            )
        )
