"""MCP 工具执行期错误。

ToolError 是所有「工具执行失败」的基类——ToolExecutor._exec_one 捕获它转 ToolResult(ok=False),
绝不抛 Traceback。普通工具亦可 raise ToolError 走同一干净通路。
"""

from __future__ import annotations


class ToolError(Exception):
    """工具执行期失败。executor 据此转 ToolResult(ok=False, error_message=str(self))。"""


class McpServerDownError(ToolError):
    """MCP server 断连 / 不可用。携带结构化 <error>...<hint> 文案回填模型。"""

    def __init__(self, server: str, tool: str, reason: str) -> None:
        self.server = server
        self.tool = tool
        self.reason = reason
        super().__init__(self._format())

    def _format(self) -> str:
        return (
            "<error>MCP 工具不可用:"
            f"mcp__{self.server}__{self.tool} 所属 server \"{self.server}\" {self.reason},"
            "本次调用未执行。</error>\n"
            "<hint>该工具在本会话内仍保留在列表中,但本次无法执行(断连后不自动重连)。"
            "请改用其它可用工具,或重启会话后重试;无需修改计划。</hint>"
        )
