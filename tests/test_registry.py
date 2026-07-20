from pydantic import BaseModel

from birdcode.tools.base import Tool
from birdcode.tools.glob_tool import GlobTool
from birdcode.tools.read_tool import ReadTool
from birdcode.tools.registry import ToolRegistry


def test_cached_marks_last_tool_only():
    reg = ToolRegistry()
    reg.register(GlobTool())
    reg.register(ReadTool())
    tools = reg.to_anthropic_tools(cached=True)
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tools[0]


def test_default_no_cache_control():
    reg = ToolRegistry()
    reg.register(ReadTool())
    assert "cache_control" not in reg.to_anthropic_tools()[0]


def test_cached_empty_registry_safe():
    assert ToolRegistry().to_anthropic_tools(cached=True) == []


def test_openai_tools_unaffected():
    reg = ToolRegistry()
    reg.register(ReadTool())
    t = reg.to_openai_tools()[0]
    assert t["type"] == "function"
    assert "cache_control" not in t["function"]


# ---- Task 6: input_schema verbatim emission ----


class _Input(BaseModel):
    x: int


class _PydanticTool(Tool):
    name = "py_tool"
    description = "d"
    parameters = _Input

    async def execute(self, **args):  # type: ignore[override]
        return ""


class _RawSchemaTool(Tool):
    name = "raw_tool"
    description = "d"
    input_schema = {"type": "object", "properties": {"y": {"type": "string"}}}

    async def execute(self, **args):  # type: ignore[override]
        return ""


def test_pydantic_tool_emits_model_json_schema():
    reg = ToolRegistry()
    reg.register(_PydanticTool())
    schema = reg.to_anthropic_tools()[0]["input_schema"]
    assert schema == _Input.model_json_schema()


def test_raw_schema_tool_emits_input_schema_verbatim():
    reg = ToolRegistry()
    reg.register(_RawSchemaTool())
    schema = reg.to_anthropic_tools()[0]["input_schema"]
    assert schema == {"type": "object", "properties": {"y": {"type": "string"}}}


def test_raw_schema_openai_emits_parameters():
    reg = ToolRegistry()
    reg.register(_RawSchemaTool())
    fn = reg.to_openai_tools()[0]["function"]
    assert fn["parameters"] == {"type": "object", "properties": {"y": {"type": "string"}}}


# ---- Task 7: deferred partition + discovery + cache breakpoint ----


class _DeferredRawTool(Tool):
    name = "raw_tool"
    description = "d"
    input_schema = {"type": "object"}

    def should_defer(self) -> bool:
        return True

    async def execute(self, **args):  # type: ignore[override]
        return ""


def test_deferred_tool_hidden_until_discovered():
    reg = ToolRegistry()
    reg.register(_PydanticTool())  # stable
    reg.register(_DeferredRawTool())  # deferred
    names = [t["name"] for t in reg.to_anthropic_tools()]
    assert "py_tool" in names
    assert "raw_tool" not in names  # deferred undiscovered → excluded
    reg.mark_discovered("raw_tool")
    names2 = [t["name"] for t in reg.to_anthropic_tools()]
    assert "raw_tool" in names2
    assert reg.is_discovered("raw_tool") is True


def test_cache_breakpoint_on_last_including_discovered():
    """已发现 MCP 工具纳入缓存:断点打在【整个列表最后一个】(含已发现),非 stable 末尾。
    发现=末尾 append,旧前缀(含旧已发现)cache read 命中、仅新增项 creation。"""
    reg = ToolRegistry()
    reg.register(_PydanticTool())  # stable
    reg.register(_DeferredRawTool())
    reg.mark_discovered("raw_tool")
    tools = reg.to_anthropic_tools(cached=True)
    assert tools[0]["name"] == "py_tool"
    assert "cache_control" not in tools[0]  # 断点不在 stable
    assert tools[1]["name"] == "raw_tool"
    assert tools[1]["cache_control"] == {"type": "ephemeral"}  # 断点在末尾(含已发现)


def test_mark_discovered_unknown_name_is_noop():
    reg = ToolRegistry()
    reg.mark_discovered("does_not_exist")
    assert reg.is_discovered("does_not_exist") is False


def test_discovered_order_is_append_only_and_stable():
    """已发现工具按【发现顺序】append(非注册序/非 set):后发现的排末尾,先发现的位置
    不变 → 断点前前缀稳定,旧已发现 cache read 命中、仅新增项 creation。"""

    class _DA(_DeferredRawTool):
        name = "mcp__s__a"

    class _DB(_DeferredRawTool):
        name = "mcp__s__b"

    class _DC(_DeferredRawTool):
        name = "mcp__s__c"

    reg = ToolRegistry()
    reg.register(_PydanticTool())  # stable
    reg.register(_DA())
    reg.register(_DB())
    reg.register(_DC())
    # 故意不按注册序发现:c 先、a 后(b 暂不发现)
    reg.mark_discovered("mcp__s__c")
    reg.mark_discovered("mcp__s__a")
    assert [t["name"] for t in reg.to_anthropic_tools()] == ["py_tool", "mcp__s__c", "mcp__s__a"]
    # 再发现 b:append 末尾,c/a 位置不变(前缀稳定)
    reg.mark_discovered("mcp__s__b")
    assert [t["name"] for t in reg.to_anthropic_tools()] == [
        "py_tool",
        "mcp__s__c",
        "mcp__s__a",
        "mcp__s__b",
    ]
    # 重复 mark 同一工具:no-op(不重复进列表)
    reg.mark_discovered("mcp__s__a")
    assert [t["name"] for t in reg.to_anthropic_tools()].count("mcp__s__a") == 1


# ---- Task 8: deferred_reminder ----


def test_deferred_reminder_lists_undiscovered_only():
    reg = ToolRegistry()
    reg.register(_PydanticTool())  # stable, not in reminder

    class _Multi(_DeferredRawTool):
        name = "multi_tool"
        description = "查询 Prometheus 指标\n第二行不要"

    reg.register(_Multi())
    reminder = reg.deferred_reminder()
    assert "multi_tool" in reminder
    assert "查询 Prometheus 指标" in reminder
    assert "第二行不要" not in reminder  # truncated to first line
    assert "py_tool" not in reminder
    reg.mark_discovered("multi_tool")
    assert "multi_tool" not in reg.deferred_reminder()


def test_deferred_reminder_empty_when_none():
    reg = ToolRegistry()
    reg.register(_PydanticTool())
    assert reg.deferred_reminder() == ""
