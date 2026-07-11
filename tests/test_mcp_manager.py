import asyncio
import io

import pytest

import birdcode.mcp.client as mclient
from birdcode.config.schema import McpHttpServer, McpStdioServer
from birdcode.mcp.errors import McpServerDownError, ToolError
from birdcode.tools.registry import ToolRegistry


class _FakeTool:
    def __init__(self, name, description="d", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object"}


class _FakeToolList:
    def __init__(self, tools):
        self.tools = tools


class _FakeText:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    def __init__(self, text, is_error=False):
        self.content = [_FakeText(text)]
        self.isError = is_error


class _FakeInitResult:
    """模拟 InitializeResult:instructions 字段在其上(ClientSession 对象本身不持有)。"""

    def __init__(self, instructions):
        self.instructions = instructions


class _FakeSession:
    def __init__(self, tools, instructions="", call_result=None, call_raises=None):
        self._tools = tools
        self._instructions = instructions
        self._call_result = call_result
        self._call_raises = call_raises
        self.initialized = False

    async def initialize(self):
        self.initialized = True
        return _FakeInitResult(self._instructions)  # instructions 在返回值上,非 session 属性(#1)

    async def list_tools(self):
        return _FakeToolList(self._tools)

    async def call_tool(self, name, args):
        if self._call_raises:
            raise self._call_raises
        return self._call_result or _FakeResult("ok")

    async def send_ping(self):
        return None


class _FakeTransportCM:
    def __init__(self):
        self.exited = False  # #6:断言超时回滚时 transport 被 __aexit__ 关闭

    async def __aenter__(self):
        return (object(), object())

    async def __aexit__(self, *exc):
        self.exited = True
        return False


def _patch_sdk(monkeypatch, session_factory, captured=None, transports=None):
    class _ClientSessionCM:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return session_factory()

        async def __aexit__(self, *exc):
            return False

    def _make_transport():
        cm = _FakeTransportCM()
        if transports is not None:
            transports.append(cm)  # #6:让测试断言超时回滚时 transport 被关闭
        return cm

    monkeypatch.setattr(mclient, "stdio_client", lambda params, errlog=None: _make_transport())
    monkeypatch.setattr(mclient, "_open_errlog", lambda name, cfg=None: io.StringIO())

    def _http(url, *, http_client=None, terminate_on_close=True):
        if captured is not None and http_client is not None:
            captured["url"] = url
            captured["headers"] = dict(http_client.headers)
        return _make_transport()

    monkeypatch.setattr(mclient, "streamable_http_client", _http)
    monkeypatch.setattr(mclient, "ClientSession", _ClientSessionCM)


async def test_startup_connects_and_registers_deferred_tools(monkeypatch):
    sess = _FakeSession(
        tools=[_FakeTool("query", "查询"), _FakeTool("list", "列出")], instructions="use me"
    )
    _patch_sdk(monkeypatch, lambda: sess)
    reg = ToolRegistry()
    mgr = mclient.McpManager({"grafana": McpStdioServer(type="stdio", command="x")}, reg)
    await mgr.startup()
    assert sess.initialized
    assert "mcp__grafana__query" in reg.names()
    assert "mcp__grafana__list" in reg.names()
    assert reg.get("mcp__grafana__query").should_defer() is True
    assert mgr.instructions == {"grafana": "use me"}


async def test_startup_isolates_failing_server(monkeypatch):
    sessions = iter([None, _FakeSession(tools=[_FakeTool("ok")])])

    def factory():
        nxt = next(sessions)
        if nxt is None:
            raise RuntimeError("boom")
        return nxt

    _patch_sdk(monkeypatch, factory)
    reg = ToolRegistry()
    mgr = mclient.McpManager(
        {
            "bad": McpStdioServer(type="stdio", command="x"),
            "good": McpHttpServer(type="streamable_http", url="http://x"),
        },
        reg,
    )
    await mgr.startup()
    assert "mcp__good__ok" in reg.names()
    assert not any(n.startswith("mcp__bad") for n in reg.names())


async def test_startup_http_server_passes_headers(monkeypatch):
    sess = _FakeSession(tools=[_FakeTool("q")], instructions="")
    captured: dict = {}
    _patch_sdk(monkeypatch, lambda: sess, captured=captured)
    reg = ToolRegistry()
    mgr = mclient.McpManager(
        {
            "g": McpHttpServer(
                type="streamable_http", url="https://x/mcp", headers={"Authorization": "Bearer t"}
            )
        },
        reg,
    )
    await mgr.startup()
    assert captured["url"] == "https://x/mcp"
    # httpx 规范化头名为小写(HTTP 头大小写不敏感);出现 user-agent 证明是真 httpx.AsyncClient。
    assert captured["headers"]["authorization"] == "Bearer t"
    assert "user-agent" in captured["headers"]
    assert "mcp__g__q" in reg.names()


async def test_call_tool_returns_text(monkeypatch):
    sess = _FakeSession(tools=[], call_result=_FakeResult("hello", is_error=False))
    _patch_sdk(monkeypatch, lambda: sess)
    mgr = mclient.McpManager({"g": McpStdioServer(type="stdio", command="x")}, ToolRegistry())
    await mgr.startup()
    out = await mgr.call_tool("g", "t", {"a": 1})
    assert out == "hello"


async def test_call_tool_server_down_when_unknown(monkeypatch):
    _patch_sdk(monkeypatch, lambda: _FakeSession(tools=[]))
    mgr = mclient.McpManager({}, ToolRegistry())
    await mgr.startup()
    with pytest.raises(McpServerDownError):
        await mgr.call_tool("nope", "t", {})


async def test_call_tool_mid_call_failure_marks_dead(monkeypatch):
    sess = _FakeSession(tools=[], call_raises=ConnectionError("reset"))
    _patch_sdk(monkeypatch, lambda: sess)
    mgr = mclient.McpManager({"g": McpStdioServer(type="stdio", command="x")}, ToolRegistry())
    await mgr.startup()
    with pytest.raises(McpServerDownError):
        await mgr.call_tool("g", "t", {})
    with pytest.raises(McpServerDownError):
        await mgr.call_tool("g", "t", {})


async def test_call_tool_is_error_raises_tool_error(monkeypatch):
    sess = _FakeSession(tools=[], call_result=_FakeResult("bad args", is_error=True))
    _patch_sdk(monkeypatch, lambda: sess)
    mgr = mclient.McpManager({"g": McpStdioServer(type="stdio", command="x")}, ToolRegistry())
    await mgr.startup()
    with pytest.raises(ToolError):
        await mgr.call_tool("g", "t", {})


async def test_handshake_timeout_isolates_hanging_server(monkeypatch):
    """超时须覆盖 initialize/list_tools——握手挂起的坏 server 被隔离、不挂死 startup。"""
    monkeypatch.setattr(mclient, "_CONNECT_TIMEOUT", 0.05)

    class _HangingSession(_FakeSession):
        async def initialize(self):
            await asyncio.sleep(10)  # 模拟接受连接但握手永不响应

    _patch_sdk(monkeypatch, lambda: _HangingSession(tools=[_FakeTool("q")]))
    reg = ToolRegistry()
    mgr = mclient.McpManager({"hang": McpStdioServer(type="stdio", command="x")}, reg)
    await asyncio.wait_for(mgr.startup(), timeout=5.0)  # 若超时未覆盖握手,会卡 ~10s
    assert "mcp__hang__q" not in reg.names()  # 挂起 server 被隔离,工具未注册


async def test_handshake_timeout_cleans_up_transport(monkeypatch):
    """#6:超时后 per-server AsyncExitStack 回滚——已 enter 的 transport 必须被 __aexit__
    关闭,不能泄漏到 shutdown(否则与「已跳过(隔离)」日志自相矛盾)。"""
    monkeypatch.setattr(mclient, "_CONNECT_TIMEOUT", 0.05)
    transports: list = []

    class _HangingSession(_FakeSession):
        async def initialize(self):
            await asyncio.sleep(10)  # 接受连接但握手永不响应

    _patch_sdk(monkeypatch, lambda: _HangingSession(tools=[_FakeTool("q")]), transports=transports)
    mgr = mclient.McpManager({"hang": McpStdioServer(type="stdio", command="x")}, ToolRegistry())
    await asyncio.wait_for(mgr.startup(), timeout=5.0)
    assert transports, "transport 应已创建"
    assert all(t.exited for t in transports), "超时后 transport 必须被回滚关闭(不再泄漏)"


async def test_shutdown_closes_transports(monkeypatch):
    """#2:shutdown 让各 server lifecycle task 自行退出(stop event)→ transport 在该 task
    内关闭(anyio cancel_scope 同 task exit,不再 RuntimeError)。旧实现 transport 进
    AsyncExitStack、shutdown 跨 task aclose → 「cancel scope in different task」+ 子进程泄漏。"""
    sess = _FakeSession(tools=[_FakeTool("q")])
    transports: list = []
    _patch_sdk(monkeypatch, lambda: sess, transports=transports)
    mgr = mclient.McpManager({"s": McpStdioServer(type="stdio", command="x")}, ToolRegistry())
    await mgr.startup()
    assert transports and not any(t.exited for t in transports)  # 运行中:transport 开着
    await mgr.shutdown()
    assert all(t.exited for t in transports)  # shutdown 后全关(同 task exit,不跨 task)


async def test_call_tool_reconnects_after_server_death(monkeypatch):
    """C:server 死后(session 被 pop)再 call_tool → 自动重连(重启 lifecycle)→ 调用成功。"""
    sess = _FakeSession(tools=[_FakeTool("q")], call_result=_FakeResult("ok"))
    _patch_sdk(monkeypatch, lambda: sess)
    mgr = mclient.McpManager({"g": McpStdioServer(type="stdio", command="x")}, ToolRegistry())
    await mgr.startup()
    mgr._sessions.pop("g", None)  # 模拟 server 死(心跳/上次调用失败标记死)
    assert mgr._sessions.get("g") is None
    out = await mgr.call_tool("g", "q", {})  # 应触发重连
    assert out == "ok"


async def test_call_tool_timeout_raises_server_down(monkeypatch):
    """B:session.call_tool 超过 _CALL_TIMEOUT → McpServerDownError(不挂死 agent_loop)。"""
    monkeypatch.setattr(mclient, "_CALL_TIMEOUT", 0.05)

    class _SlowCallSession(_FakeSession):
        async def call_tool(self, name, args):
            await asyncio.sleep(10)

    _patch_sdk(
        monkeypatch,
        lambda: _SlowCallSession(tools=[_FakeTool("q")], call_result=_FakeResult("ok")),
    )
    mgr = mclient.McpManager({"g": McpStdioServer(type="stdio", command="x")}, ToolRegistry())
    await mgr.startup()
    with pytest.raises(McpServerDownError):
        await mgr.call_tool("g", "q", {})
    assert mgr._sessions.get("g") is None  # 超时后标记死(pop)→ 下次重连


async def test_heartbeat_failure_marks_server_dead(monkeypatch):
    """心跳 ping 失败 → 标记死(pop session)。"""
    monkeypatch.setattr(mclient, "_PING_TIMEOUT", 30.0)  # ping 不靠超时,靠 raises

    class _DeadPingSession(_FakeSession):
        async def send_ping(self):
            raise ConnectionError("ping failed")

    _patch_sdk(monkeypatch, lambda: _DeadPingSession(tools=[_FakeTool("q")]))
    mgr = mclient.McpManager(
        {"g": McpStdioServer(type="stdio", command="x")},
        ToolRegistry(),
        heartbeat_interval=0.01,
    )
    await mgr.startup()
    await asyncio.sleep(0.1)  # 等心跳触发(interval 0.01s)
    assert mgr._sessions.get("g") is None  # 心跳失败标记死


async def test_connect_phase_single_budget_timeout(monkeypatch):
    """#1:单一 asyncio.timeout 预算(open+session+init+list)—— open 快(0.1s)+ init 挂(0.1s)
    + timeout=0.15s:总 0.2s > 0.15 → 超时隔离。

    旧双 wait_for 实现给每阶段独立 0.15s 预算 → open(0.1)+init(0.1)都 < 0.15 → 成功注册工具
    (测试会失败);现 asyncio.timeout 单预算 → 总 0.2 > 0.15 超时 → 无工具(测试通过)。锁定 #1。"""
    monkeypatch.setattr(mclient, "_CONNECT_TIMEOUT", 0.15)

    class _QuickOpenTransport:
        async def __aenter__(self):
            await asyncio.sleep(0.1)  # open 0.1s(< 单阶段预算,但占单总预算)
            return (object(), object())

        async def __aexit__(self, *exc):
            return False

    class _InitHangSession(_FakeSession):
        async def initialize(self):
            await asyncio.sleep(0.1)  # init 0.1s(open+init 总 0.2s > 0.15 单预算)
            return _FakeInitResult(self._instructions)

    _patch_sdk(monkeypatch, lambda: _InitHangSession(tools=[_FakeTool("q")]))
    monkeypatch.setattr(mclient, "stdio_client", lambda params, errlog=None: _QuickOpenTransport())
    reg = ToolRegistry()
    mgr = mclient.McpManager({"hang": McpStdioServer(type="stdio", command="x")}, reg)
    await asyncio.wait_for(mgr.startup(), timeout=5.0)
    assert not any(n.startswith("mcp__hang") for n in reg.names())  # 单预算超时 → 无工具
    await mgr.shutdown()


async def test_open_phase_timeout_isolates_hanging_transport(monkeypatch):
    """#5:transport open 永久挂起 → asyncio.timeout 超时 → 隔离(open 阶段有限)。"""
    monkeypatch.setattr(mclient, "_CONNECT_TIMEOUT", 0.05)

    class _HangTransport:
        async def __aenter__(self):
            await asyncio.sleep(10)  # transport open 挂起(模拟 stdio 子进程不响应)
            return (object(), object())

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(mclient, "stdio_client", lambda params, errlog=None: _HangTransport())
    monkeypatch.setattr(mclient, "_open_errlog", lambda name, cfg=None: io.StringIO())
    reg = ToolRegistry()
    mgr = mclient.McpManager({"hang": McpStdioServer(type="stdio", command="x")}, reg)
    await asyncio.wait_for(mgr.startup(), timeout=5.0)
    assert not any(n.startswith("mcp__hang") for n in reg.names())
    await mgr.shutdown()


async def test_call_tool_rejected_during_shutdown(monkeypatch):
    """#3:shutdown 后 call_tool → McpServerDownError(不残留 60s wait_for call_tool)。"""
    sess = _FakeSession(tools=[_FakeTool("q")], call_result=_FakeResult("ok"))
    _patch_sdk(monkeypatch, lambda: sess)
    mgr = mclient.McpManager({"g": McpStdioServer(type="stdio", command="x")}, ToolRegistry())
    await mgr.startup()
    await mgr.shutdown()  # _shutting_down=True + _servers={}
    with pytest.raises(McpServerDownError, match="正在关闭"):  # 锁定 _shutting_down 路径(#5)
        await mgr.call_tool("g", "q", {})
