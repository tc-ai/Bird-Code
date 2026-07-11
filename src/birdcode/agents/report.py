# src/birdcode/agents/report.py
"""子 agent 报告 / <task-notification> / toolUseResult 构造。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from birdcode.agents.definition import AgentDefinition
from birdcode.agents.runner import SubagentReport
from birdcode.session.models import AgentToolUseResult


@dataclass
class SubagentProgress:
    """同步子 agent 进度(经 progress_cb 发给主 UI;不进 ProviderEvent 联合)。"""

    agent_id: str
    description: str
    elapsed_ms: int
    input_tokens: int  # 减去缓存的输入(Anthropic input_tokens 累加,已扣 cache_read/creation)
    tool_use_count: int
    phase: str  # "running"(完成由 ToolResult 自然替换卡片)


def build_task_notification(
    report: SubagentReport, defn: AgentDefinition, *, agent_id: str, prompt: str,
    is_worktree: bool = False,
) -> str:
    """<task-notification> 文本(同步 tool_result 与异步注入共用同一构造)。

    is_worktree=True 时附 Artifacts 行(产物清单 + 是否提交);False(非 worktree)省略。
    """
    completed = "yes" if report.is_completed else "no"
    stats = (
        f"Duration: {report.duration_ms}ms | Tokens: {report.tokens} | "
        f"Tools: {report.tool_use_count}"
    )
    lines = [
        "<task-notification>",
        f"Agent: {defn.description} (type={defn.name}, id={agent_id})",
        f"Status: {report.status}",
        f"Completed: {completed}",
        stats,
    ]
    if is_worktree:
        artifacts_line = _format_artifacts(report, agent_id)
        if artifacts_line:
            lines.append(artifacts_line)
    lines.append("")
    lines.append(report.text)
    lines.append("</task-notification>")
    return "\n".join(lines)


def _format_artifacts(report: SubagentReport, agent_id: str) -> str:
    """Artifacts 行(worktree 模式)。无产物 → 'Artifacts: 无';有则带 (N, 提交状态) 标签。"""
    if not report.artifacts:
        return "Artifacts: 无"
    tag = f"已提交到 worktree-{agent_id}" if report.committed else "未提交"
    return f"Artifacts ({len(report.artifacts)}, {tag}): {', '.join(report.artifacts)}"


def build_agent_tool_use_result(
    report: SubagentReport, defn: AgentDefinition, *,
    agent_id: str, prompt: str, is_async: bool,
) -> dict[str, Any]:
    """构造 AgentToolUseResult(供 ToolOutput.tool_use_result;不进 message.content/不发 LLM)。

    用 alias(camelCase)构造——与 runner.py SubagentMeta(...) 一致,满足 pydantic mypy
    插件(model_config populate_by_name=True 在运行时同时接受两种写法)。
    """
    return AgentToolUseResult(
        agentId=agent_id, agentType=defn.name, status=report.status, isAsync=is_async,
        prompt=prompt, resolvedModel=None, content=report.text,
        isCompleted=report.is_completed, outputFile=None, canReadOutputFile=False,
        totalDurationMs=report.duration_ms, totalTokens=report.tokens,
        totalToolUseCount=report.tool_use_count, usage=None, toolStats=None,
    ).model_dump(by_alias=True, exclude_none=True)


def build_launch_tool_use_result(
    defn: AgentDefinition, *, agent_id: str, prompt: str, output_file: str,
) -> dict[str, Any]:
    """异步启动态 AgentToolUseResult:status=launched、content=None、output_file=侧链路径。

    子 agent 异步派发后立即回执给主 agent 的 toolUseResult(is_async=True):content=None
    (尚无结果)、output_file 指向侧链 jsonl(canReadOutputFile=False——侧链在跑,不可读)。
    完成统计全 None(尚未跑完)。与 build_agent_tool_use_result 同样用 alias 构造。
    """
    return AgentToolUseResult(
        agentId=agent_id, agentType=defn.name, status="launched", isAsync=True,
        prompt=prompt, resolvedModel=None, content=None, isCompleted=True,
        outputFile=output_file, canReadOutputFile=False, totalDurationMs=None,
        totalTokens=None, totalToolUseCount=None, usage=None, toolStats=None,
    ).model_dump(by_alias=True, exclude_none=True)
