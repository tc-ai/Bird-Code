# src/birdcode/agent/system_prompt/__init__.py
"""系统提示拼装出口。

- build_system_text(override, mcp_instructions):模块 + 稳定环境 + MCP instructions。
- build_system_reminder(registry):动态环境 + 延迟工具清单,包成 <system-reminder>。
"""

from __future__ import annotations

from birdcode.agent.system_prompt.env import _resolve_cwd, gather_dynamic_env, gather_stable_env
from birdcode.agent.system_prompt.modules import modules_text

__all__ = ["build_system_text", "build_system_reminder"]


def _mcp_section(instructions: dict[str, str]) -> str:
    if not instructions:
        return ""
    lines = [
        "⑧ MCP Server",
        "以下外部 server 已连接(工具名形如 mcp__<server名>__<tool名>,"
        "需用 tool_search 拉取完整定义):",
    ]
    for name, ins in instructions.items():
        lines.append(f"\n### {name}")
        if ins:
            lines.append(ins)
    return "\n".join(lines)


def build_system_text(
    *,
    override: str | None = None,
    mcp_instructions: dict[str, str] | None = None,
    project_instructions: str | None = None,
    subagent: bool = False,
) -> str:
    """项目指令 + 七个模块 + 稳定环境 + (可选)MCP instructions。

    override 非空时整段 verbatim 返回(逃生口,此时 BirdCode.md 被旁路)。
    顺序:project_instructions(BirdCode.md)→ 模块 → 稳定环境 → MCP。
    高优先级内容靠前,模型注意力权重更高。MCP 会话内稳定、进缓存 block。
    subagent=True:模块走子 agent 精简版(_IDENTITY→_SUBAGENT_IDENTITY、去 _TOOLS)。
    记忆索引**不走这里**——每轮从 <system-reminder> 现读注入(见 build_system_reminder),
    这样提取一旦落盘、后续轮次即可见(代价:记忆块不进前缀缓存)。
    """
    if override:
        return override
    parts: list[str] = []
    if project_instructions:
        parts.append(project_instructions)
    parts.append(modules_text(subagent=subagent))
    parts.append(gather_stable_env())
    mcp = _mcp_section(mcp_instructions or {})
    if mcp:
        parts.append(mcp)
    return "\n\n".join(parts)


async def build_system_reminder(
    *, registry: object | None = None, memory: str = "", subagent: bool = False
) -> str:
    """动态环境 + 延迟工具清单 + 记忆索引,包成 <system-reminder>。

    memory 非空则附末尾。调用方每轮现读磁盘填入(见 base_llm._inject_reminder),
    故记忆更新无需等下会话——提取落盘后,后续轮次即见。

    subagent=True:末尾追加工作目录硬约束(放 reminder 末尾=离生成点最近、注意力最高)。
    cwd 经 _resolve_cwd:worktree 子 agent → 其 worktree dir(_AGENT_CWD);否则 Path.cwd()。
    """
    body = await gather_dynamic_env(registry=registry, memory=memory)
    if not subagent:
        # 主 agent:鼓励用子 agent 分担独立子任务(并行、隔离工作目录)。
        body += "\n任务复杂？可以尝试使用子 agent 工具来帮你完成任务。"
    if subagent:
        cwd = _resolve_cwd(None)
        body += (
            "\n你是个子agent，务必在工作目录下进行你的工作，"
            "严禁编辑/修改非工作目录下的任何文件，可以读取。"
            f"你的工作目录是 {cwd}"
        )
    return f"<system-reminder>\n{body}\n</system-reminder>"
