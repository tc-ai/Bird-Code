# src/birdcode/ui/pure.py
"""UI 纯逻辑（无 Textual 依赖，可单测）：续行/斜杠判定、路径缩写。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from birdcode.agent.context import ContextUsage


def should_continue(line: str) -> bool:
    """行尾反斜杠 → 回车应换行而非提交。"""
    return line.endswith("\\")


@dataclass(frozen=True)
class ParsedCommand:
    name: str  # 已小写、含 "/"，如 "/help"
    args: str  # 首空格后的原文（大小写保留）；无参数则 ""


def parse_command(text: str) -> ParsedCommand | None:
    """解析斜杠输入：返回 (name, args) 或 None（非斜杠 / 空 / 裸 "/"）。

    name = 首空格前的 token（含 "/"、转小写→大小写不敏感）。
    args = 首空格后的原文（保留大小写与多空格）；无空格则 ""。
    """
    s = text.strip()
    if not s.startswith("/"):
        return None
    rest = s[1:]  # 去掉 "/"
    if not rest.strip():
        return None  # 裸 "/" 或 "/ "
    parts = rest.split(maxsplit=1)
    name = "/" + parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return ParsedCommand(name=name, args=args)


def shorten_path(path: str) -> str:
    """状态栏路径缩写：home 下取末两段，否则取末两段。"""
    p = Path(path)
    try:
        rel = p.relative_to(Path.home())
        rel_parts = [s for s in rel.parts if s and s != "."]
        if not rel_parts:
            return "~"
        return "~/" + "/".join(rel_parts[-2:])
    except ValueError:
        parts = [s for s in p.parts if s and s != "."]
        return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else str(p))


def enter_action(line: str) -> str:
    """回车语义：行尾反斜杠→续行，否则提交。"""
    return "continue" if should_continue(line) else "submit"


def _fmt_tokens(n: int) -> str:
    """token → 人类可读：≥1000 显示 '1.7k'（1 位小数），否则整数。"""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def format_context_report(usage: ContextUsage, model: str) -> str:
    """渲染 /context 报告（纯文本 + Unicode 块字符，仿 CC /context）。

    顶部：模型 + 已用/窗口 + 占比。中部：进度条 + autocompact 阈值标记（▲）。
    分类目：System prompt / Tools（含 MCP 子项）/ Messages / Used / Free。
    脚注：autocompact 触发点 + 上次 API 实测 input（若有，作估算的参照）。
    """
    w = usage.window or 1

    def pct(x: int) -> str:
        return f"{x / w * 100:.1f}%"

    bar_width = 24
    filled = max(0, min(bar_width, round(usage.used / w * bar_width)))
    bar = "█" * filled + "░" * (bar_width - filled)
    # 阈值标记对齐到 bar 内列（首列是 "["，故前留 1 空格对齐）。
    thr_pos = max(0, min(bar_width, round(usage.autocompact_threshold / w * bar_width)))
    marker = " " + " " * thr_pos + "▲ autocompact"

    lines: list[str] = [
        f"Context Usage · {model} · {_fmt_tokens(usage.used)} / {_fmt_tokens(w)} "
        f"({pct(usage.used)})",
        "",
        f"[{bar}]",
        marker,
        "",
    ]

    def row(label: str, tokens: int, indent: str = "") -> None:
        lines.append(f"{indent}{label:<13}{_fmt_tokens(tokens):>8}  ({pct(tokens)})")

    row("System prompt", usage.system_prompt)
    row(f"Tools ({usage.tool_count})", usage.tools)
    if usage.mcp_tool_count:
        row(f"└ MCP ({usage.mcp_tool_count})", usage.mcp_tools, indent="  ")
    row("Messages", usage.messages)
    lines.append("─" * 36)
    row("Used", usage.used)
    row("Free", usage.free)
    lines.append("")
    lines.append(
        f"Autocompact 触发于 {_fmt_tokens(usage.autocompact_threshold)} "
        f"({pct(usage.autocompact_threshold)})"
    )
    if usage.last_measured is not None:
        lines.append(f"(估算；上次 API 实测 input={_fmt_tokens(usage.last_measured)})")
    else:
        lines.append("(估算；尚无 API 实测)")
    return "\n".join(lines)
