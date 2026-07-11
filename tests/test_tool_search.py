from birdcode.mcp.tool_search import ToolSearchTool
from birdcode.tools.base import Tool
from birdcode.tools.registry import ToolRegistry


class _Deferred(Tool):
    def __init__(self, name, desc):
        self.name = name
        self.description = desc
        self.input_schema = {"type": "object"}

    def should_defer(self) -> bool:
        return True

    async def execute(self, **args):  # type: ignore[override]
        return ""


def _reg_with(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


async def test_select_mode_marks_discovered_and_returns_schema():
    reg = _reg_with(_Deferred("mcp__grafana__query_prometheus", "查询"))
    ts = ToolSearchTool(registry=reg)
    out = await ts.execute(query="select:mcp__grafana__query_prometheus")
    assert reg.is_discovered("mcp__grafana__query_prometheus")
    assert "query_prometheus" in out


async def test_select_unknown_returns_message_no_mark():
    reg = _reg_with()
    ts = ToolSearchTool(registry=reg)
    out = await ts.execute(query="select:mcp__nope__x")
    assert "未找到" in out
    assert not reg.is_discovered("mcp__nope__x")


async def test_keyword_mode_searches_name_and_desc():
    reg = _reg_with(
        _Deferred("mcp__grafana__query_prometheus", "查询 Prometheus 指标"),
        _Deferred("mcp__slack__send", "发送 Slack 消息"),
    )
    ts = ToolSearchTool(registry=reg)
    out = await ts.execute(query="prometheus")
    assert reg.is_discovered("mcp__grafana__query_prometheus")
    assert not reg.is_discovered("mcp__slack__send")
    assert "query_prometheus" in out


async def test_keyword_no_match():
    reg = _reg_with(_Deferred("mcp__grafana__x", "g"))
    ts = ToolSearchTool(registry=reg)
    out = await ts.execute(query="zzz")
    assert "无匹配" in out


async def test_search_skips_already_discovered():
    reg = _reg_with(_Deferred("mcp__g__q", "prometheus"))
    reg.mark_discovered("mcp__g__q")
    ts = ToolSearchTool(registry=reg)
    out = await ts.execute(query="prometheus")
    assert "无匹配" in out
