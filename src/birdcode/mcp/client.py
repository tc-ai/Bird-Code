"""McpManager:per-server lifecycle task + 懒重连(C)+ 周期 ping(心跳)+ call_tool 超时(B)。

每个 server 一个 `_server_lifecycle` task:连完(open+initialize+list_tools 单一超时)
→ 进心跳循环(周期 `send_ping`,失败标记死→返回→finally pop session→下次 call 触发重连)。
`call_tool`:session 在则用(并 `wait_for` 兜底 B);不在(死过)→ 懒重连(C:重启 lifecycle
+ 等连上)→ 再调。`shutdown`:stop event 通知各 task 自行退出(同 task 关 transport,anyio
cancel_scope 不跨 task),**不 shield**(允许外部 cancel 传到 lifecycle task),**扩
except** 防单 task 异常(如 errlog.close OSError)中止清理 for,**gather 带超时**防
某 task `__aexit__` 不响应取消(Windows stdio)致 on_unmount 挂死。

- transport 的 async with 必须在 lifecycle task 内 enter+exit。
- errlog 仅 stdio、在 try 内。
- httpx client 显式大 read timeout(300s,A),防慢 URL 触发 `ReadTimeout`→task_group 崩→
  永久隔离。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from birdcode.config.schema import McpHttpServer, McpServerConfig, McpStdioServer
from birdcode.mcp.adapter import McpToolAdapter
from birdcode.mcp.errors import McpServerDownError, ToolError
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    from birdcode.tools.registry import ToolRegistry

log = get_logger("birdcode.mcp.client")  # 经 file logger 写 ~/.birdcode/debug.log

_CONNECT_TIMEOUT = 30.0  # open+initialize+list_tools 总超时(单一 wait_for)
_CALL_TIMEOUT = 60.0  # 单次 call_tool 墙钟兜底(B,防 session.call_tool 挂)
_PING_TIMEOUT = 10.0  # 心跳 ping 超时
_SHUTDOWN_GRACE = 5.0  # shutdown 等 server task 自行退出的宽限
# httpx timeout(A):read 放大到 300s,防慢 URL 抓取触发 ReadTimeout→task_group 一崩全崩→
# 整个 server 永久隔离(曾发生于 fetch 抓 aliyun 文章)。
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)


class McpManager:
    def __init__(
        self,
        servers: dict[str, McpServerConfig],
        registry: ToolRegistry,
        *,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self._servers = servers
        self._initial_servers = dict(servers)  # 快照:shutdown 清 _servers 后,startup 可恢复
        self._registry = registry
        self._sessions: dict[str, ClientSession] = {}
        self._instructions: dict[str, str] = {}
        self._server_tasks: dict[str, asyncio.Task[None]] = {}
        self._stop: dict[str, asyncio.Event] = {}
        self._ready: dict[str, asyncio.Event] = {}
        self._heartbeat_interval = heartbeat_interval
        self._shutting_down = False  # shutdown 期间拒重连:防 spawn 新 task 其 stop 永不被 set

    async def startup(self) -> None:
        self._shutting_down = False  # 重置
        self._servers = dict(self._initial_servers)  # 恢复:协调 shutdown 清空
        if not self._servers:
            return
        for name, cfg in self._servers.items():
            self._spawn_lifecycle(name, cfg)
        # 等「连接阶段」结束(ready 在连接完成/失败/超时 set),带超时,不让慢 server 挂死调用方。
        await asyncio.gather(*(self._wait_ready(n) for n in self._servers), return_exceptions=True)

    def _spawn_lifecycle(self, name: str, cfg: McpServerConfig) -> None:
        self._stop[name] = asyncio.Event()
        self._ready[name] = asyncio.Event()
        self._server_tasks[name] = asyncio.create_task(
            self._server_lifecycle(name, cfg), name=f"mcp:{name}"
        )

    async def _wait_ready(self, name: str) -> None:
        try:
            await asyncio.wait_for(
                self._ready[name].wait(), timeout=_CONNECT_TIMEOUT + _SHUTDOWN_GRACE
            )
        except TimeoutError:
            log.warning("MCP server %s 连接阶段超时(%ss)", name, _CONNECT_TIMEOUT)

    async def _server_lifecycle(self, name: str, cfg: McpServerConfig) -> None:
        """单个 server 的完整生命周期:连完 → 心跳循环 → 死则退出(可被重连)。

        连接阶段(open+session+init+list)在**单一 asyncio.timeout(_CONNECT_TIMEOUT)** 内
        (30s 总预算,不再两段 wait_for 致 60s>35s 窗口;ClientSession.__aenter__ 也覆盖
        )。heartbeat 在 timeout 外(长期,不被 30s 限)——故 transport/session 用手动 cm 持有
        跨 heartbeat(anyio cancel_scope 同 task:enter/exit 都在本 task)。finally 只 pop 自己的
        session;关闭失败提为 warning(子进程泄漏信号);errlog 在嵌套 finally 总关
        (即使 cm 关闭抛 CancelledError 也不跳过)。
        """
        stop = self._stop[name]
        ready = self._ready[name]
        errlog: TextIO | None = None
        transport_cm: AbstractAsyncContextManager[tuple[Any, Any]] | None = None
        session_cm: AbstractAsyncContextManager[ClientSession] | None = None
        my_session: ClientSession | None = None
        transport_entered = False
        session_entered = False
        try:
            errlog = _open_errlog(name, cfg)
            transport_cm = _open_transport(cfg, errlog)
            async with asyncio.timeout(_CONNECT_TIMEOUT):
                read, write = await transport_cm.__aenter__()
                transport_entered = True
                session_cm = ClientSession(read, write)
                my_session = await session_cm.__aenter__()
                session_entered = True
                init, tools_resp = await self._init_and_list(my_session)
            # 连接成功(timeout 块正常退出,未超时)
            self._instructions[name] = init.instructions or ""
            for t in tools_resp.tools:
                self._registry.register(
                    McpToolAdapter(name, t.name, t.description or "", t.inputSchema, self)
                )
            self._sessions[name] = my_session
            log.info("MCP server %s 已连接,注册 %d 个工具", name, len(tools_resp.tools))
            ready.set()
            await self._heartbeat_loop(name, my_session, stop)
        except Exception:
            log.exception("MCP server %s 连接失败/断开,已跳过(下次调用将尝试重连)", name)
        finally:
            ready.set()  # 失败/超时也 set,让 startup/reconnect 的 waiter 不卡死
            # 只 pop 自己的 session(防重连时旧 lifecycle 的 finally 误 pop 新 session)
            if my_session is None or self._sessions.get(name) is my_session:
                self._sessions.pop(name, None)
            # 关 session + transport(同 task)。per-cm try/except BaseException(含 CancelledError,
            # 不让 session 的异常中断 transport 关闭 → stdio 子进程泄漏);entered 防未 enter 的 cm
            # __aexit__;明确标签替代无用的 type(cm).__name__。
            for label, cm, entered in (
                ("session", session_cm, session_entered),
                ("transport", transport_cm, transport_entered),
            ):
                if entered and cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                    except BaseException:  # noqa: BLE001 - 含 CancelledError:保证两个 cm 都关
                        log.warning(
                            "MCP server %s %s 关闭失败/被取消(可能子进程/连接泄漏)",
                            name,
                            label,
                            exc_info=True,
                        )
            if errlog is not None:
                try:
                    errlog.close()
                except OSError:
                    log.debug("MCP server %s errlog 关闭失败", name, exc_info=True)

    async def _init_and_list(self, session: ClientSession) -> tuple[Any, Any]:
        """initialize + list_tools(由 _server_lifecycle 套单一超时)。"""
        init = await session.initialize()
        tools_resp = await session.list_tools()
        return init, tools_resp

    async def _heartbeat_loop(self, name: str, session: ClientSession, stop: asyncio.Event) -> None:
        """周期 ping;失败标记死(返回→finally pop session→下次 call 触发重连)。

        心跳关(<=0)则直接等 stop。每轮先 `wait_for(stop.wait(), interval)`——stop 到则正常
        退出(shutdown),否则(超时)发一次 ping。ping 超时/异常 → 标记掉线退出。
        """
        if self._heartbeat_interval <= 0:
            await stop.wait()
            return
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._heartbeat_interval)
                return  # stop set → 正常退出
            except TimeoutError:
                pass  # 心跳间隔到
            try:
                await asyncio.wait_for(session.send_ping(), timeout=_PING_TIMEOUT)
            except Exception:  # noqa: BLE001 - ping 失败=server 不健康
                log.warning("MCP server %s 心跳失败,标记掉线(下次调用将重连)", name)
                return

    async def call_tool(self, server: str, tool: str, args: dict[str, Any]) -> str:
        if self._shutting_down:
            raise McpServerDownError(server, tool, "正在关闭,不接受调用")
        session = self._sessions.get(server)
        if session is None:
            # 懒重连(C):server 死过(心跳/上次调用失败/启动失败)→ 重启 lifecycle + 等连上。
            cfg = self._servers.get(server)
            if cfg is None:
                raise McpServerDownError(server, tool, "未知 server")
            session = await self._reconnect(server, cfg)
            if session is None:
                raise McpServerDownError(server, tool, "重连失败")
        try:
            # B:墙钟兜底,防 session.call_tool 永挂(server 慢/不回响应)卡死 agent_loop。
            result = await asyncio.wait_for(session.call_tool(tool, args), timeout=_CALL_TIMEOUT)
        except TimeoutError:
            self._sessions.pop(server, None)  # 标记死→下次重连
            raise McpServerDownError(server, tool, f"调用超时(>{int(_CALL_TIMEOUT)}s)") from None
        except Exception as exc:  # noqa: BLE001 - 传输错/中途断连:标记掉线→下次重连
            self._sessions.pop(server, None)
            raise McpServerDownError(server, tool, f"调用中途连接断开: {exc}") from exc
        return _extract_text(result, server, tool)

    async def _reconnect(self, name: str, cfg: McpServerConfig) -> ClientSession | None:
        """懒重连(C):重启该 server 的 lifecycle task + 等连上。返回 session 或 None。

        旧 task 若仍挂着(理论上已死,但防御)先 cancel;重连失败则清 instructions(
        避免模型反复看到死 server 的 instructions 而反复试,代价是该轮缓存 miss——罕见)。
        """
        if self._shutting_down:
            return None  # shutdown 中拒重连:防新 task 的 stop 永不被 set → 孤儿
        log.info("MCP server %s 断连,尝试重连...", name)
        old = self._server_tasks.get(name)
        if old is not None and not old.done():
            old.cancel()
        self._spawn_lifecycle(name, cfg)
        try:
            await asyncio.wait_for(
                self._ready[name].wait(), timeout=_CONNECT_TIMEOUT + _SHUTDOWN_GRACE
            )
        except TimeoutError:
            log.warning("MCP server %s 重连超时", name)
        session = self._sessions.get(name)
        if session is None:
            self._instructions.pop(name, None)  # 重连失败:清 instructions
        return session

    @property
    def instructions(self) -> dict[str, str]:
        return dict(self._instructions)

    async def shutdown(self) -> None:
        self._shutting_down = True  # 先设:拒重连:防 spawn 新 task 其 stop 永不被 set → 孤儿
        # 通知各 lifecycle task 自行退出:stop.wait() 返回 → 退出 async with → 同 task 关 transport。
        for ev in list(self._stop.values()):  # list:防 _reconnect 并发改 dict
            ev.set()
        for task in list(self._server_tasks.values()):  # list
            if not task.done():
                try:
                    # 删 shield:允许 shutdown 被外部 cancel 时取消传到 lifecycle task。
                    # except 仅 Exception(不含 CancelledError——shutdown 自身被外部 cancel 时
                    # 让其传播中止,而非吞掉继续跑 2×宽限)。单 task 异常(errlog.close OSError)
                    # 由 Exception 接,不中止清理 for。
                    await asyncio.wait_for(task, timeout=_SHUTDOWN_GRACE)
                except Exception:
                    if not task.done():
                        task.cancel()
        # gather 带超时:防某 task __aexit__ 不响应取消(Windows stdio 子进程)致 on_unmount 挂死。
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._server_tasks.values(), return_exceptions=True),
                timeout=_SHUTDOWN_GRACE,
            )
        except TimeoutError:
            pending = [n for n, t in self._server_tasks.items() if not t.done()]
            log.warning("MCP shutdown:%d 个 server task 未在宽限内退出: %s", len(pending), pending)
        self._sessions.clear()
        self._servers = {}  # 清:shutdown 后拒重连(old adapter 调 → McpServerDownError,不僵尸)


def _open_errlog(name: str, cfg: McpServerConfig) -> TextIO | None:
    """stdio server 的 stderr 重定向目标(仅 stdio);http 不需要、返回 None。

    Textual 接管 console 后 `sys.stderr.fileno()` 失效,mcp 的 `stdio_client` 默认把 server
    stderr 重定向到 `sys.stderr` → Windows `Bad file descriptor`。故 stdio server 开一个日志
    文件(`~/.birdcode/mcp-<name>.log`)收 stderr;开失败回退 devnull。caller 负责 close。
    """
    if not isinstance(cfg, McpStdioServer):
        return None
    try:
        path = Path.home() / ".birdcode" / f"mcp-{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "w", encoding="utf-8")  # noqa: SIM115
    except OSError:
        log.warning("MCP server %s 的 stderr 日志文件创建失败,改用 devnull", name, exc_info=True)
        return open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


@asynccontextmanager
async def _open_transport(
    cfg: McpServerConfig, errlog: TextIO | None
) -> AsyncIterator[tuple[Any, Any]]:
    """打开 transport(stdio / streamable_http),yield (read, write)。

    包成单个 async cm:stdio 的 anyio cancel_scope 绑定调用方 task,必须同 task exit。
    http:显式大 read timeout(`_HTTP_TIMEOUT`,300s,A)防慢 URL 触发 ReadTimeout→task_group 崩。
    """
    if isinstance(cfg, McpStdioServer):
        if errlog is None:  # 不用 assert(-O 下剥离);_open_errlog 对 stdio 应返回非 None
            raise RuntimeError("stdio errlog 未初始化")
        params = StdioServerParameters(
            command=cfg.command, args=list(cfg.args), env={**os.environ, **cfg.env}
        )
        async with stdio_client(params, errlog=errlog) as transport:
            yield transport[0], transport[1]  # 兼容 SDK 返回 2 或 3 元组
    elif isinstance(cfg, McpHttpServer):
        async with httpx.AsyncClient(headers=dict(cfg.headers), timeout=_HTTP_TIMEOUT) as client:
            async with streamable_http_client(cfg.url, http_client=client) as transport:
                yield transport[0], transport[1]
    else:  # pragma: no cover - 判别联合已保证
        raise TypeError(f"未知 MCP server 类型: {cfg!r}")


def _extract_text(result: Any, server: str, tool: str) -> str:
    """拼 CallToolResult 的 text content;isError → ToolError(走 ok=False 通路)。"""
    parts = [getattr(c, "text", "") for c in getattr(result, "content", [])]
    text = "".join(p for p in parts if p)
    if getattr(result, "isError", False):
        raise ToolError(text or f"MCP 工具 mcp__{server}__{tool} 返回错误")
    return text
