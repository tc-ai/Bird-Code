# 复用 test_mcp_manager 的 fake session 套路(pytest prepend 模式把 tests/ 加进 sys.path,
# 故用裸模块名导入;无 __init__.py,不能用 tests.xxx 形式)
from test_mcp_manager import _FakeResult, _FakeSession, _FakeTool, _patch_sdk

from birdcode.blocks import ToolUseBlock
from birdcode.config.schema import McpStdioServer
from birdcode.mcp.client import McpManager
from birdcode.mcp.tool_search import ToolSearchTool
from birdcode.tools.executor import ToolExecutor
from birdcode.tools.registry import ToolRegistry


async def test_e2e_deferred_then_discovered_then_called(monkeypatch):
    """延迟工具:首轮不在 schema 列表 → tool_search 发现 → 下轮进列表 → 调用成功。"""
    sess = _FakeSession(tools=[_FakeTool("query", "查询指标")], call_result=_FakeResult("metric=1"))
    _patch_sdk(monkeypatch, lambda: sess)
    reg = ToolRegistry()
    mgr = McpManager({"g": McpStdioServer(type="stdio", command="x")}, reg)
    await mgr.startup()
    reg.register(ToolSearchTool(registry=reg))
    ex = ToolExecutor(reg)

    # 1) 延迟工具不在 schema 列表、在 reminder
    tools = reg.to_anthropic_tools(cached=True)
    assert "mcp__g__query" not in [t["name"] for t in tools]
    assert "mcp__g__query" in reg.deferred_reminder()

    # 2) 模型调 tool_search 发现它
    r = await ex.execute_batch(
        [ToolUseBlock(id="s1", name="tool_search", input={"query": "select:mcp__g__query"})]
    )
    assert r[0].ok and "query" in r[0].output

    # 3) 下轮它进 schema 列表、离开 reminder
    tools2 = reg.to_anthropic_tools(cached=True)
    assert "mcp__g__query" in [t["name"] for t in tools2]
    assert "mcp__g__query" not in reg.deferred_reminder()

    # 4) 模型直接调它(无 gate → 直接执行,转发到 fake session)
    r2 = await ex.execute_batch([ToolUseBlock(id="c1", name="mcp__g__query", input={"q": "up"})])
    assert r2[0].ok and r2[0].output == "metric=1"

    await mgr.shutdown()


async def test_e2e_server_down_returns_structured_error(monkeypatch):
    """server 断连:调用 → McpServerDownError → ToolResult(ok=False, 结构化文案);列表不变。"""
    sess = _FakeSession(tools=[_FakeTool("query")], call_raises=ConnectionError("reset"))
    _patch_sdk(monkeypatch, lambda: sess)
    reg = ToolRegistry()
    mgr = McpManager({"g": McpStdioServer(type="stdio", command="x")}, reg)
    await mgr.startup()
    reg.register(ToolSearchTool(registry=reg))
    reg.mark_discovered("mcp__g__query")  # 已发现,直接可调
    ex = ToolExecutor(reg)

    before = [t["name"] for t in reg.to_anthropic_tools()]
    r = await ex.execute_batch([ToolUseBlock(id="c1", name="mcp__g__query", input={})])
    assert r[0].ok is False
    assert "<error>" in r[0].error_message and "mcp__g__query" in r[0].error_message
    # tools 列表不变(D8:不删除)
    assert [t["name"] for t in reg.to_anthropic_tools()] == before

    await mgr.shutdown()
