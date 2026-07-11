import pytest

from birdcode.mcp.adapter import McpToolAdapter
from birdcode.mcp.errors import McpServerDownError


class _FakeManager:
    def __init__(self, result="ok-result", raises=None):
        self._result = result
        self._raises = raises
        self.calls: list[tuple[str, str, dict]] = []

    async def call_tool(self, server, tool, args):
        self.calls.append((server, tool, args))
        if self._raises:
            raise self._raises
        return self._result


def test_adapter_name_namespace():
    a = McpToolAdapter("grafana", "query_prometheus", "desc", {"type": "object"}, _FakeManager())
    assert a.name == "mcp__grafana__query_prometheus"


def test_adapter_is_write_not_parallel_deferred():
    a = McpToolAdapter("s", "t", "d", {"type": "object"}, _FakeManager())
    assert a.kind == "write"
    assert a.parallel_safe is False
    assert a.should_defer() is True


def test_adapter_input_schema_set():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    a = McpToolAdapter("s", "t", "d", schema, _FakeManager())
    assert a.input_schema == schema


async def test_adapter_execute_forwards_to_manager():
    mgr = _FakeManager(result="42")
    a = McpToolAdapter("grafana", "query", "d", {"type": "object"}, mgr)
    out = await a.execute(query="up")
    assert out == "42"
    assert mgr.calls == [("grafana", "query", {"query": "up"})]


async def test_adapter_execute_passes_server_down_error():
    mgr = _FakeManager(raises=McpServerDownError("grafana", "query", "连接已断开"))
    a = McpToolAdapter("grafana", "query", "d", {"type": "object"}, mgr)
    with pytest.raises(McpServerDownError):
        await a.execute()


def test_adapter_preserves_empty_schema_not_replaced():
    """回归(#12):server 明确给 input_schema={}(JSON Schema 语义=接受任意)须原样保留,
    不可被 `or` 当 falsy 替换成 {type:object}——否则收窄了 server 契约。"""
    assert McpToolAdapter("s", "t", "d", {}, _FakeManager()).input_schema == {}
    # None 才用宽松默认
    assert McpToolAdapter("s", "t", "d", None, _FakeManager()).input_schema == {
        "type": "object",
        "properties": {},
    }
