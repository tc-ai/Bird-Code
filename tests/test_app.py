# tests/test_app.py
import asyncio
import gc

import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.ui.app import BirdApp
from birdcode.ui.widgets.input_area import InputArea


@pytest.mark.asyncio
async def test_compose_has_input_and_status():
    async with BirdApp(MockProvider(delay=0.0)).run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#input") is not None
        assert pilot.app.query_one("#status") is not None


@pytest.mark.asyncio
async def test_one_turn_streams_and_finalizes():
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("hello world"))
        await pilot.pause(delay=2.0)
        assert app.controller.busy is False
        assert len(app.controller.history) == 1
        assert app.query(".turn")  # 有已提交的轮次
        assert len(app.query("Markdown")) >= 1  # 助手 Markdown 已挂载


@pytest.mark.asyncio
async def test_transcript_contains_turns():
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("hello world"))
        await pilot.pause(delay=2.0)
        assert "hello world" in app.transcript()


@pytest.mark.asyncio
async def test_escape_interrupts():
    app = BirdApp(MockProvider(delay=0.5))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("to be interrupted"))
        await pilot.pause(delay=0.05)
        assert app.controller.busy is True
        await pilot.press("escape")
        await pilot.pause(delay=1.0)
        assert app.controller.busy is False


@pytest.mark.asyncio
async def test_ctrl_c_does_not_interrupt():
    # Ctrl+C 不再中断——放行给 Textual 内建复制（TextArea.action_copy →
    # screen.copy_text → help_quit），故生成期间 busy 保持 True。
    # 用慢流（delay=5.0）保证窗口期内绝不会自然结束，排除「自己跑完」的干扰。
    app = BirdApp(MockProvider(delay=5.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("streaming..."))
        await pilot.pause(delay=0.1)
        assert app.controller.busy is True
        await pilot.press("ctrl+c")
        await pilot.pause(delay=0.2)
        assert app.controller.busy is True  # ctrl+c 没有中断它
        app.controller.interrupt()  # 收尾，停掉后台流
        await pilot.pause(delay=0.3)
        assert app.controller.busy is False


@pytest.mark.asyncio
async def test_ctrl_q_quits():
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+q")
        await pilot.pause()
        assert app._exit is True


@pytest.mark.asyncio
async def test_view_pinned_to_bottom_after_turn():
    # 修复：提交并流式完成后，视图应钉在底部（输入框始终可见，不被顶出屏外）。
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("hello world"))
        await pilot.pause(delay=2.0)
        scroll = app.query_one("#scroll")
        max_y = max(0, scroll.virtual_size.height - scroll.size.height)
        assert scroll.scroll_y >= max_y - 1


@pytest.mark.asyncio
async def test_input_retains_focus_after_turn():
    # 诊断：一整轮结束后，焦点必须仍在输入框——否则终端光标（IME 锚点）会停在
    # 助手输出上，导致中文首字光标错位。
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        assert app.focused is ia
        await app._on_submitted(InputArea.Submitted("hello world"))
        await pilot.pause(delay=2.0)
        assert app.focused is ia, f"focus drifted to {app.focused!r}"


@pytest.mark.asyncio
async def test_composer_is_outside_scroll():
    # 输入区固定在底部、不在滚动区里 → 屏位恒定，终端光标(IME 锚点)不随滚动漂移。
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        scroll = app.query_one("#scroll")
        composer = app.query_one("#composer")
        ia = app.query_one(InputArea)
        # composer 与 scroll 是 Screen 下的兄弟节点
        assert composer in app.screen.children
        assert scroll in app.screen.children
        # 输入框不属于滚动区
        assert scroll not in list(ia.ancestors_with_self)


@pytest.mark.asyncio
async def test_queued_turn_keeps_view_pinned():
    # 满屏 + type-ahead：A 流式中把 B 入队；B 后续执行时视图仍钉底（输入框恒可见）。
    app = BirdApp(MockProvider(delay=0.2))
    async with app.run_test(size=(60, 8)) as pilot:  # 小窗口 → 很快满屏
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("first message here"))
        await pilot.pause(delay=0.2)
        assert app.controller.busy is True
        await app._on_submitted(InputArea.Submitted("second message here"))  # 入队
        await pilot.pause()  # 让 submit(B) 任务跑起来 → 入队（A 仍在流式）
        assert app.controller.queue_size == 1
        await pilot.pause(delay=4.0)  # 等 A、B 都跑完
        assert app.controller.busy is False
        scroll = app.query_one("#scroll")
        max_y = max(0, scroll.virtual_size.height - scroll.size.height)
        assert scroll.scroll_y >= max_y - 1, f"not pinned: {scroll.scroll_y}/{max_y}"
        # 输入框固定在底部，屏位必须落在可视区内
        ia = app.query_one(InputArea)
        assert 0 <= ia.region.y < app.size.height


@pytest.mark.asyncio
async def test_follow_respects_user_scroll_up():
    # Bug 1 回归:用户上滑后,跟随定时器回调 _tick_follow 不再把视图钉回底部。
    # 旧实现 timer 每 80ms 调 _pin_to_bottom(无条件 scroll_end),同步子 agent 运行期间
    # 用户无法上滑读历史——视图被持续拉回底部。
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(60, 8)) as pilot:  # 小窗口→输出很快溢出
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("word " * 120))
        await pilot.pause(delay=2.0)
        scroll = app.query_one("#scroll")
        max_y = max(0, scroll.virtual_size.height - scroll.size.height)
        if max_y <= 0:
            pytest.skip("输出未溢出滚动区,无法验证上滑回归")
        # 模拟用户上滑:把视图滚到顶部。scroll_y 减小 → TranscriptScroll.watch_scroll_y
        # 置 user_following=False(我们的 scroll_end 只增不减,故减小必为用户上滑)。
        scroll.scroll_y = 0
        await pilot.pause()
        assert scroll.scroll_y <= 1
        assert scroll.user_following is False
        # 跑若干次跟随定时器回调(生成期间每 80ms 一次)
        for _ in range(5):
            app._tick_follow()
            await pilot.pause()
        assert scroll.scroll_y <= 1, f"用户上滑被拉回底部: {scroll.scroll_y}/{max_y}"


@pytest.mark.asyncio
async def test_bg_task_exception_is_retrieved(monkeypatch):
    """#4 加固：后台轮次任务的完成回调应取走异常。

    set.discard 只从集合移除、不读 .exception()，故失败的后台任务回收时 asyncio 经
    事件循环异常处理器报 'Task exception was never retrieved'。回调应调用 .exception()
    取走异常（且对 cancelled 任务不抛），并落日志。
    """
    app = BirdApp(MockProvider())

    loop = asyncio.get_running_loop()
    reported: list[dict] = []
    monkeypatch.setattr(loop, "call_exception_handler", reported.append)

    async def boom() -> None:
        raise RuntimeError("bg fail")

    task = asyncio.ensure_future(boom())
    while not task.done():  # 推进到完成，但不 await（不取走异常）= fire-and-forget
        await asyncio.sleep(0)
    app._bg_tasks.add(task)
    app._reap_bg_task(task)  # done-callback 等价：移除引用 + 取走异常
    assert task not in app._bg_tasks
    del task  # 释放最后引用 → Task.__del__（若异常未取走会调 handler）
    gc.collect()

    assert reported == [], "background task exception was never retrieved"


# ---- stage2: 注入链 —— provider 与 executor 共享同一 registry ----
# 注:BirdApp 用 _tool_registry(而非 _registry)避免与 Textual App._registry
# (WeakSet[DOMNode],DOM 节点追踪)命名冲突——覆盖后者会破坏挂载。


def test_birdapp_stores_and_reuses_registry():
    """BirdApp.__init__ 接收 registry 存 self._tool_registry(on_mount 用它构造 executor)。"""
    from birdcode.tools.echo import EchoTool
    from birdcode.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(EchoTool())
    app = BirdApp(MockProvider(), model="mock", registry=reg)
    assert app._tool_registry is reg  # noqa: SLF001


def test_birdapp_registry_defaults_to_none():
    app = BirdApp(MockProvider(), model="mock")
    assert app._tool_registry is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_on_event_tool_result_passes_truncated_to_tool_line():
    """BirdApp._on_event 收到 ToolResult(truncated=True) → 透传给 ToolLine.finish。"""
    from birdcode.agent.provider import ToolCallStart, ToolResult
    from birdcode.ui.widgets.tool_line import ToolLine

    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_event(ToolCallStart(id="t1", name="echo", args="{}"))
        line = app._tools.get("t1")
        assert isinstance(line, ToolLine)
        await app._on_event(ToolResult(id="t1", ok=True, summary="s", truncated=True))
        assert getattr(line, "_truncated", False) is True


# ---- stage15: MCP 接入 on_mount ----


@pytest.mark.asyncio
async def test_on_mount_safe_without_mcp_servers():
    """on_mount 在无 mcp_servers 配置(cfg=None)时不崩溃:manager 已建、instructions 空。

    McpManager({}, reg).startup() 对空 dict 直接 return(no-op),故 on_mount 不会因
    MCP 失败;_mcp_manager 属性必然存在(__init__ 预置 None + on_mount 赋值)。
    tool_search 常驻工具也在此注册(无论有无 MCP server)。
    """
    app = BirdApp(MockProvider(delay=0.0))  # cfg=None → mcp_servers={}
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        assert app._mcp_manager is not None  # noqa: SLF001
        assert app._mcp_manager.instructions == {}  # noqa: SLF001
        assert "tool_search" in app._tool_registry.names()  # noqa: SLF001


def test_mcp_instructions_returns_empty_when_no_manager():
    """BirdApp.mcp_instructions() 在 manager 未建时安全返回空 dict(供 provider 懒读)。"""
    app = BirdApp(MockProvider(), model="mock")
    # __init__ 后 _mcp_manager 为 None(on_mount 尚未跑)
    assert app.mcp_instructions() == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_on_mount_does_not_block_on_mcp_startup(monkeypatch):
    """#10:on_mount 不同步 await MCP startup——UI 先就绪、startup 后台跑。
    用慢 startup 证明 on_mount 在它完成前就返回(_mcp_startup_task 仍 pending)。"""

    class _SlowMgr:
        def __init__(self, *a, **kw) -> None:
            pass

        async def startup(self) -> None:
            await asyncio.sleep(5)  # 远超 pilot.pause,若 on_mount 同步 await 会卡死测试

        async def shutdown(self) -> None:
            pass

        @property
        def instructions(self) -> dict:
            return {}

    monkeypatch.setattr("birdcode.mcp.client.McpManager", _SlowMgr)
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        assert app._mcp_startup_task is not None  # noqa: SLF001
        assert not app._mcp_startup_task.done()  # 慢 startup 仍在跑 → on_mount 未被它阻塞
        assert "tool_search" in app._tool_registry.names()  # noqa: SLF001


@pytest.mark.asyncio
async def test_slash_server_and_tool_run_without_error():
    """/server、/tool 命令执行不崩;无 MCP 时 /server 给空提示、/tool 列内置工具。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("/server"))
        await app._on_submitted(InputArea.Submitted("/tool"))
        await pilot.pause()
        # 两命令都产出 Static 输出挂在滚动区(不崩即通过)
        assert app.query("#scroll Static")


# ---- Task 9: resume 重放 + /clear 开新会话 + /sessions + gate 放行 tool-results ----


def _enc(p):
    from birdcode.session import paths

    return paths.encode_cwd(p)


@pytest.mark.asyncio
async def test_resume_replays_history_into_ui(tmp_path):
    """resume=True:on_mount 调 controller.resume → history 填充 + UI 重放出 Turn widget。"""
    from birdcode.blocks import TextBlock
    from birdcode.conversation import Message
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    ctx = SessionContext(session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="旧问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="旧答")]), is_assistant=True
    )
    store.close()

    store2 = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store2, resume=True)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause(delay=0.5)  # 等 on_mount 异步重放完成
        assert len(app.controller.history) == 1
        assert app.controller.history[0].messages[0].content[0].text == "旧问"
        assert app.query(".turn")  # UI 重放出 Turn


@pytest.mark.asyncio
async def test_new_session_no_replay(tmp_path):
    """resume=False(新会话)→ history 空,UI 无 Turn。"""
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    ctx = SessionContext(session_id="new1", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause(delay=0.3)
        assert len(app.controller.history) == 0


@pytest.mark.asyncio
async def test_clear_opens_new_session_id_keeps_old(tmp_path):
    """/clear → 开新 sessionId(旧文件保留);controller history 清空;controller 持新 store。"""
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    ctx = SessionContext(session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("hello"))
        await pilot.pause(delay=1.0)  # 等 MockProvider 跑完一轮
        old_sid = app._store._ctx.session_id  # noqa: SLF001
        await app._on_submitted(InputArea.Submitted("/clear"))
        await pilot.pause()
        new_sid = app._store._ctx.session_id  # noqa: SLF001
        assert new_sid != old_sid  # 开新 sessionId
        assert len(app.controller.history) == 0
        # controller 持的 store 也应指向新 store
        assert app.controller._store is app._store  # noqa: SLF001
    # 旧文件保留(reopen 不删旧 jsonl)
    assert (tmp_path / _enc(tmp_path) / f"{old_sid}.jsonl").exists()


@pytest.mark.asyncio
async def test_sessions_command_lists_history(tmp_path):
    """/sessions → 列出本项目历史会话(只读 Static),不抛即通过。"""
    from birdcode.blocks import TextBlock
    from birdcode.conversation import Message
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    # 预置 2 个会话
    for sid, text in [("a1", "第一问"), ("b2", "第二问")]:
        ctx = SessionContext(session_id=sid, cwd=str(tmp_path), version="0.1.0", git_branch=None)
        s = SessionStore(ctx, tmp_path, root=tmp_path)
        await s.append(Message(role="user", content=[TextBlock(text=text)]))
        s.close()

    ctx = SessionContext(session_id="c3", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("/sessions"))
        await pilot.pause()
        # #scroll 里应挂上列出文本(至少 banner + 列表 Static)
        scroll = app.query_one("#scroll")
        assert scroll.children


@pytest.mark.asyncio
async def test_sessions_command_handles_no_store():
    """/sessions 在 store=None(未启用持久化)时降级提示,不崩。"""
    app = BirdApp(MockProvider(delay=0.0))  # 不传 store
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("/sessions"))
        await pilot.pause()
        # 滚动区有 Static 输出(提示「未启用」)即通过
        assert app.query("#scroll Static")


@pytest.mark.asyncio
async def test_no_store_path_keeps_old_behavior(tmp_path):
    """BirdApp 不传 store(默认 None)→ on_mount 的 store 分支跳过,行为同旧。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause(delay=0.3)
        # controller 未持 store,resume 返回空,history 空
        assert app._store is None  # noqa: SLF001
        assert len(app.controller.history) == 0
        # 提交一轮仍正常工作
        await app._on_submitted(InputArea.Submitted("hello"))
        await pilot.pause(delay=1.0)
        assert len(app.controller.history) == 1


# ---- F1: /clear 必须重接 executor._output_sink 与 gate 的 extra_roots ----


@pytest.mark.asyncio
async def test_clear_rehooks_executor_sink_and_gate_extra_roots(tmp_path):
    """F1 回归守护:/clear 后 executor._output_sink 指向新 store 的 sink、
    gate._extra_roots 含新 store 的 _tool_results 目录。否则大输出落到旧 session、
    新 session 的 tool-results 在 gate extra_roots 外被 L2 拦(read_file 取不到落盘)。
    """
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    ctx = SessionContext(session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        # on_mount 后 executor/gate 应已存到 self 上(F1 改造)
        assert app._executor is not None  # noqa: SLF001
        assert app._gate is not None  # noqa: SLF001
        old_sink = app._executor._output_sink  # noqa: SLF001
        old_store = app._store  # noqa: SLF001
        assert old_sink is not None
        # /clear 前抓旧 store 的 tool-results 根(用于断言「旧根已移除」)
        old_tr = old_store.tool_results_dir.resolve()
        await app._on_submitted(InputArea.Submitted("/clear"))
        await pilot.pause()
        new_sink = app._executor._output_sink  # noqa: SLF001
        new_store = app._store  # noqa: SLF001
        # 新 store 与旧 store 不同(reopen)
        assert new_store is not old_store
        # 新 sink 必须与旧 sink 不同(指向新 session)
        assert new_sink is not old_sink
        new_tr = new_store.tool_results_dir.resolve()
        roots = app._gate.extra_roots
        # #4:新根在、旧根已移除(replace 非累积)。旧实现 add 只追加 → 旧根残留,
        # 下面 old_tr not in roots 会失败:这正是该回归测试要守住的底线(旧断言同义反复)。
        assert new_tr in roots
        assert old_tr not in roots


@pytest.mark.asyncio
async def test_clear_exception_safe_controller_gets_open_store(tmp_path, monkeypatch):
    """#10:/clear 中 replace_extra_root 抛异常时,controller._store 仍指向新(开)store。

    旧顺序(rehook 在 controller 赋值前):异常跳过 controller._store 赋值 → controller 仍指
    reopen 已关闭的旧 store → 后续 _on_message 写已关句柄 → ValueError 被 store 静默吞 →
    整段 /clear 后对话从 jsonl 丢失(仅下次 --resume 空结果时才暴露)。修复:controller._store
    先于 rehook 赋值,rehook 包 try。
    """
    import json as _json

    from birdcode.blocks import TextBlock
    from birdcode.conversation import Message
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    ctx = SessionContext(session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        new_store_ref = app._store  # noqa: SLF001

        # 让 replace_extra_root 抛异常(模拟 resolve/系统调用/AV 锁失败)
        def _boom(old, new):  # noqa: ANN001
            raise OSError("simulated resolve failure")

        monkeypatch.setattr(app._gate, "replace_extra_root", _boom)  # noqa: SLF001
        await app._on_submitted(InputArea.Submitted("/clear"))  # 不应抛
        await pilot.pause()
        # controller 必须持新(开)store,而非旧已关 store
        assert app.controller._store is app._store is not new_store_ref  # noqa: SLF001
        # 新 store 句柄开 → append 能落盘(关键:旧 bug 下这里会静默丢)
        await app.controller._store.append(  # noqa: SLF001
            Message(role="user", content=[TextBlock(text="clear后能落盘")])
        )
    # 验证落到了新 session 的 jsonl
    jf = app._store._jsonl  # noqa: SLF001
    rows = [_json.loads(ln) for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert any(
        r.get("message", {}).get("content", [{}])[0].get("text") == "clear后能落盘" for r in rows
    ), "clear 后消息未落盘(controller 仍指已关旧 store?)"


# ---- Task 10: /compact 手动压缩命令 ----


def _compact_cfg():
    """最小可用 AppConfig(给 BirdApp 构造 ContextManager 用:有 store + cfg 才启用)。"""
    from birdcode.config.schema import AppConfig

    return AppConfig.model_validate(
        {
            "providers": {
                "p": {"protocol": "anthropic", "model": "m", "base_url": "u", "api_key": "k"}
            },
            "default": "p",
        }
    )


@pytest.mark.asyncio
async def test_compact_without_context_degrades_gracefully():
    """/compact 在 controller._context=None(未启用持久化)时降级提示,不崩。"""
    # 不传 store / cfg → on_mount 不构造 ContextManager → controller._context=None
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        assert app.controller._context is None  # noqa: SLF001
        await app._on_submitted(InputArea.Submitted("/compact"))  # 不应抛
        await pilot.pause()
        # #scroll 里应挂上提示 Static(未启用持久化)
        assert app.query("#scroll Static")


@pytest.mark.asyncio
async def test_compact_calls_compact_now_with_manual_trigger(tmp_path, monkeypatch):
    """/compact → controller._context.compact_now(history, trigger=manual) 被调。"""
    from birdcode.agent.context import CompactionResult, ContextManager
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    captured: dict = {}

    async def fake_compact_now(self, *, history, trigger="manual"):  # noqa: ANN001
        captured["trigger"] = trigger
        captured["history_is_list"] = isinstance(history, list)
        return CompactionResult(
            trigger=trigger,
            pre_tokens=1000,
            post_tokens=500,
            boundary_uuid="",
            summary_uuid="",
            fell_back=False,
        )

    monkeypatch.setattr(ContextManager, "compact_now", fake_compact_now)

    ctx = SessionContext(session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store, cfg=_compact_cfg(), resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        # on_mount 构造了 ContextManager(有 store + cfg)
        assert app.controller._context is not None  # noqa: SLF001
        await app._on_submitted(InputArea.Submitted("/compact"))
        await pilot.pause()
        # compact_now 被调,trigger=manual,history 是 controller.history 那个 list
        assert captured.get("trigger") == "manual"
        assert captured.get("history_is_list") is True
        # UI 有反馈 Static(压缩结果)
        assert app.query("#scroll Static")


@pytest.mark.asyncio
async def test_compact_fallback_label_shown_when_fell_back(tmp_path, monkeypatch):
    """/compact 走硬截断兜底(fell_back=True)→ UI 展示带「硬截断兜底」标注。"""
    from birdcode.agent.context import CompactionResult, ContextManager
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    async def fake_compact_now(self, *, history, trigger="manual"):  # noqa: ANN001
        return CompactionResult(
            trigger="manual",  # F13:摘要熔断也保留调用方 trigger,用 fell_back 标识兜底
            pre_tokens=2000,
            post_tokens=800,
            boundary_uuid="",
            summary_uuid="",
            fell_back=True,
        )

    monkeypatch.setattr(ContextManager, "compact_now", fake_compact_now)

    ctx = SessionContext(session_id="s2", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store, cfg=_compact_cfg(), resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("/compact"))
        await pilot.pause()
        # 收集 #scroll 里所有 Static 文本,应含「硬截断兜底」
        texts = [str(s.content) for s in app.query("#scroll Static")]
        assert any("硬截断兜底" in t for t in texts), texts


@pytest.mark.asyncio
async def test_compact_exception_does_not_crash(tmp_path, monkeypatch):
    """/compact 内 compact_now 抛异常 → catch + log + 友好提示,绝不抛 Traceback。"""
    from birdcode.agent.context import ContextManager
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    async def boom_compact_now(self, *, history, trigger="manual"):  # noqa: ANN001
        raise RuntimeError("compact boom")

    monkeypatch.setattr(ContextManager, "compact_now", boom_compact_now)

    ctx = SessionContext(session_id="s3", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    app = BirdApp(MockProvider(delay=0.0), store=store, cfg=_compact_cfg(), resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("/compact"))  # 不应抛
        await pilot.pause()
        # UI 有失败提示 Static
        texts = [str(s.content) for s in app.query("#scroll Static")]
        assert any("失败" in t for t in texts), texts


@pytest.mark.asyncio
async def test_send_user_message_runs_in_background_and_interruptible():
    """/review 用的 send_user_message 后台跑 submit(不内联阻塞消息泵)→ Esc 可中断。

    回归守护:旧实现 `await controller.submit(text)` 内联在 _on_submitted 里,占住
    消息泵,Esc 无法中断、状态不刷新。修复后走 create_task(+_bg_tasks),与普通输入同路径。
    """
    app = BirdApp(MockProvider(delay=0.3))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app.send_user_message("review me")
        await pilot.pause()  # 让后台 submit 起步、置 busy
        assert app.controller.busy is True  # submit 在后台跑(未阻塞返回)
        assert app._bg_tasks  # noqa: SLF001 - 走了 create_task 路径
        await pilot.press("escape")  # Esc 现在能中断(消息泵未被占住)
        await pilot.pause(delay=1.0)
        assert app.controller.busy is False
        assert not app._bg_tasks  # noqa: SLF001 - 回收


@pytest.mark.asyncio
async def test_bare_slash_not_sent_to_llm():
    """裸 "/" 或 "/ " → 静默忽略,绝不当用户消息发给 LLM(旧行为)。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("/"))
        await app._on_submitted(InputArea.Submitted("   /   "))
        await pilot.pause(delay=0.3)
        assert len(app.controller.history) == 0  # 没有轮次进 LLM
        assert not app._bg_tasks  # noqa: SLF001


@pytest.mark.asyncio
async def test_unknown_command_warns():
    """/未知命令 → 友好警告(钉底),不崩、不发 LLM。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("/nope"))
        await pilot.pause()
        texts = [str(s.content) for s in app.query("#scroll Static")]
        assert any("未知命令" in t for t in texts), texts
        assert len(app.controller.history) == 0
