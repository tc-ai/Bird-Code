"""McpToolAdapter:把单个 MCP 工具桥接成 BirdCode Tool。

name = mcp__<server>__<tool>(强制命名空间)。kind=write、parallel_safe=False、should_defer=True。
input_schema 取 server 返回的原始 JSON schema。execute 透传给 McpManager.call_tool;
McpServerDownError 原样上冒(executor 捕获转 ok=False)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from birdcode.tools.base import Tool

if TYPE_CHECKING:
    from birdcode.mcp.client import McpManager


class McpToolAdapter(Tool):
    kind = "write"
    parallel_safe = False
    # 延迟工具(should_defer=True),唯一加载路径是 tool_search。子 agent 的 registry 不含
    # tool_search(session_scoped 排除)→ 把本工具复制进子表只会让 deferred_reminder 宣称
    # 「可用 tool_search 拉 MCP 工具」,而子 agent 既无 tool_search、deferred MCP 工具也永不
    # 进其工具列表 → 空耗一轮调用不存在的工具。标 session_scoped → build_child_registry 排除。
    session_scoped = True

    def __init__(
        self,
        server: str,
        tool: str,
        description: str,
        input_schema: dict[str, Any] | None,
        manager: McpManager,
    ) -> None:
        self.name = f"mcp__{server}__{tool}"
        self._server = server
        self._tool = tool
        self._manager = manager
        self.description = description or ""
        # None(server 未声明)→ 宽松对象 schema;但 server 明确给 {}(JSON Schema 语义=接受任意)
        # 须原样保留——`or` 会把 falsy 的 {} 也替换掉,收窄了 server 契约。
        default_schema: dict[str, Any] = {"type": "object", "properties": {}}
        self.input_schema = input_schema if input_schema is not None else default_schema

    def should_defer(self) -> bool:
        return True

    async def execute(self, **args: object) -> str:
        return await self._manager.call_tool(self._server, self._tool, dict(args))
