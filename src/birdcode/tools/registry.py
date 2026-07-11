# src/birdcode/tools/registry.py
"""name → Tool 注册表;并把已注册工具导出为各家 API 的 tools 参数。"""

from __future__ import annotations

from birdcode.tools.base import Tool
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.tools.registry")  # 进 debug.log


def _partition(
    tools: dict[str, Tool], discovered_order: list[str]
) -> tuple[list[Tool], list[Tool]]:
    """三分流(仅非 hidden):稳定(非延迟) / 已发现延迟(按发现序) / 延迟未发现(跳过)。

    已发现段按【发现顺序】(discovered_order)排,而非注册序——保证列表单调 append:
    新发现的工具总在末尾,断点前旧前缀字节不变 → cache read 命中,仅新增项 creation。
    若按注册序,新发现可能插中间、错乱前缀 → 后续项全部 miss。
    """
    stable = [t for t in tools.values() if not t.hidden and not t.should_defer()]
    disc = [
        tools[name]
        for name in discovered_order
        if name in tools and not tools[name].hidden and tools[name].should_defer()
    ]
    return stable, disc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # 已发现延迟工具:list 保【发现顺序】(append-only,断点前前缀稳定 → cache 命中);
        # 并行 set 做 O(1) 成员检查(tool_search 循环调 is_discovered,纯 list 是 O(T×D))。
        # 纯内存、会话级:每次启动 default_registry() 新建即空,不落盘、不跨会话残留。
        self._discovered: list[str] = []
        self._discovered_set: set[str] = set()

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool.name 不能为空")
        if tool.name in self._tools:
            log.warning(
                "工具 %s 重复注册,覆盖旧实例(%s)",
                tool.name,
                type(self._tools[tool.name]).__name__,
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def mark_discovered(self, name: str) -> None:
        """标记延迟工具为已发现——下一轮进 schema 列表。单调:不提供 un-discover。

        按【发现顺序】append(去重),非 set——保证列表前缀稳定(见 _discovered 注释)。
        仅当 name 是已注册工具时才记录;未知/重复静默忽略。
        """
        if name in self._tools and name not in self._discovered_set:
            self._discovered.append(name)
            self._discovered_set.add(name)

    def is_discovered(self, name: str) -> bool:
        return name in self._discovered_set

    def to_anthropic_tools(self, *, cached: bool = False) -> list[dict[str, object]]:
        """延迟未发现工具不进列表;已发现的按发现序 append 在稳定段后。cached=True 时
        断点打在【整个列表最后一个】——已发现 MCP 工具一并纳入缓存(只增不减,新发现 =
        末尾 append,旧前缀含旧已发现 cache read 命中、仅新增项 creation)。"""
        stable, disc = _partition(self._tools, self._discovered)
        tools: list[dict[str, object]] = [
            {"name": t.name, "description": t.description, "input_schema": t.schema()}
            for t in (*stable, *disc)
        ]
        if cached and tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools

    def to_openai_tools(self) -> list[dict[str, object]]:
        stable, disc = _partition(self._tools, self._discovered)
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema(),
                },
            }
            for t in (*stable, *disc)
        ]

    def deferred_reminder(self) -> str:
        """延迟未发现工具的「名字 — 一行截断描述」清单,供动态 system-reminder。
        会话内随发现状态变(已发现的离开)——位于缓存前缀之外,不破缓存。"""
        lines: list[str] = []
        for t in self._tools.values():
            if t.hidden or not t.should_defer() or t.name in self._discovered_set:
                continue
            desc = (t.description or "").splitlines()[0].strip() if t.description else ""
            lines.append(f"- {t.name} — {desc}" if desc else f"- {t.name}")
        if not lines:
            return ""
        return (
            "以下延迟工具可用(完整 schema 未加载,需要时用 tool_search 拉取):\n" + "\n".join(lines)
        )
