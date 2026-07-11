"""ToolSearchTool:按需发现延迟工具。

常驻可见(非延迟、非 hidden),kind=read、parallel_safe=False
(read-default 放行免 HITL;串行避免 registry 写竞态)。
- select:<name> 精确拉取;关键词在所有「延迟且未发现」工具的
  name+description 里子串搜(大小写不敏感)。
- 命中即 mark_discovered + 回吐完整 JSON schema 文本
  (模型本轮可读,下轮进 tools 列表)。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from birdcode.tools.base import Tool

if TYPE_CHECKING:
    from birdcode.tools.registry import ToolRegistry


class _ToolSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="select:mcp__<server名>__<tool名> 精确拉取;或关键词(如 prometheus)模糊搜索",
    )


class ToolSearchTool(Tool):
    name = "tool_search"
    description = (
        "按需发现并加载延迟工具的完整 schema。传 select:mcp__<server名>__<tool名> 精确拉取,"
        "或传关键词模糊搜索名称与描述。"
    )
    kind = "read"
    parallel_safe = False
    parameters = _ToolSearchInput
    # 持有【父 registry】引用:子 agent 调 select/mark_discovered 会改父 registry 的延迟工具
    # 发现状态,且子 agent 的工具表在派生时就固定、本不该动态发现 MCP 工具。
    # 标 session_scoped → build_child_registry 排除(子 agent 不持有 tool_search)。
    session_scoped = True

    def __init__(self, *, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, query: str) -> str:  # type: ignore[override]
        if query.startswith("select:"):
            name = query[len("select:"):].strip()
            t = self._registry.get(name)
            if t is not None and t.should_defer():
                self._registry.mark_discovered(name)
                return _format_schema(t)
            return f"未找到延迟工具: {name}"
        q = query.lower()
        hits = []
        for n in self._registry.names():
            t = self._registry.get(n)
            if t is None or not t.should_defer() or self._registry.is_discovered(n):
                continue
            if q in n.lower() or q in (t.description or "").lower():
                hits.append(t)
        for t in hits:
            self._registry.mark_discovered(t.name)
        if not hits:
            return "无匹配延迟工具"
        return "\n\n".join(_format_schema(t) for t in hits)


def _format_schema(t: Tool) -> str:
    return f"工具 {t.name} 已加载,完整 schema:\n{json.dumps(t.schema(), ensure_ascii=False)}"
