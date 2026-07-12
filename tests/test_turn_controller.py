# tests/test_turn_controller.py
import asyncio

import pytest

from birdcode.agent.provider import (
    Done,
    Error,
    Interrupted,
    TextDelta,
    TokenUsage,
    TurnStart,
)
from birdcode.agents.mailbox import MailboxMessage
from birdcode.conversation import TurnController


class FakeProvider:
    """Yields a fixed list of events, then completes."""

    def __init__(self, events):
        self.events = list(events)

    async def stream(self, messages, *, history):
        for e in self.events:
            yield e


class GatedProvider:
    """Yields one delta, blocks on `gate` until set, then yields Done."""

    def __init__(self, gate):
        self.gate = gate

    async def stream(self, messages, *, history):
        yield TextDelta(text="...")
        await self.gate.wait()
        yield Done(usage=TokenUsage())


class RaisingProvider:
    """Yields one delta, then raises（模拟真实 provider 的超时/限流/鉴权异常）。"""

    async def stream(self, messages, *, history):
        yield TextDelta(text="partial")
        raise RuntimeError("boom")


async def _collect_events():
    return []


@pytest.mark.asyncio
async def test_submit_runs_one_turn_and_marks_idle():
    received = []
    statuses = []

    async def on_event(ev):
        received.append(ev)

    async def on_status():
        statuses.append(1)

    p = FakeProvider([TextDelta(text="hi"), Done(usage=TokenUsage())])
    ctrl = TurnController(p, on_event=on_event, on_status=on_status)

    await ctrl.submit("hello")
    assert ctrl.busy is False
    # TurnStart（controller 开轮，带回 user_text）+ TextDelta + Done
    assert len(received) == 3
    assert isinstance(received[0], TurnStart) and received[0].user_text == "hello"
    assert isinstance(received[-1], Done)
    assert len(ctrl.history) == 1


@pytest.mark.asyncio
async def test_provider_exception_becomes_error_event():
    """#1：provider 抛异常（而非 yield Error）→ controller 转成 Error 事件，不静默失败。"""
    received = []

    async def on_event(ev):
        received.append(ev)

    async def on_status(): ...

    ctrl = TurnController(RaisingProvider(), on_event=on_event, on_status=on_status)
    await ctrl.submit("x")
    assert ctrl.busy is False  # 没有冒泡杀死任务、没卡住
    assert any(isinstance(e, Error) for e in received)
    assert isinstance(received[-1], Error)


@pytest.mark.asyncio
async def test_abort_clears_queue_and_suppresses_interrupted():
    """#3：abort() 取消当前流 + 清空队列，且不发 Interrupted。"""
    gate = asyncio.Event()
    events = []

    async def on_event(ev):
        events.append(ev)

    async def on_status(): ...

    ctrl = TurnController(GatedProvider(gate), on_event=on_event, on_status=on_status)
    task = asyncio.create_task(ctrl.submit("a"))
    await asyncio.sleep(0)  # 让 a 起来并阻塞在 gate
    await ctrl.submit("b")
    await ctrl.submit("c")
    assert ctrl.queue_size == 2

    ctrl.abort()
    await task
    assert ctrl.busy is False
    assert ctrl.queue_size == 0
    assert not any(isinstance(e, Interrupted) for e in events)  # abort 静默，无中断行


@pytest.mark.asyncio
async def test_abort_does_not_orphan_turn_in_history():
    """#3 加固：abort()（/clear）取消的轮次不应在 history.clear() 之后复活。

    复现 /clear 竞态——abort() 只调度取消，history.clear() 会先于 _process 的 finally
    跑（_close_markdown 在 _md_stream 为 None 时不 await）。若 finally 无条件 append，
    被取消的轮次会以「孤立 user turn（无助手回复、interrupted=False）」复活，污染下一轮
    LLM 上下文（history 即下一轮 stream 的输入）。
    """
    gate = asyncio.Event()

    async def on_event(ev): ...
    async def on_status(): ...

    ctrl = TurnController(GatedProvider(gate), on_event=on_event, on_status=on_status)
    task = asyncio.create_task(ctrl.submit("a"))
    await asyncio.sleep(0)  # a 起来并阻塞在 gate
    ctrl.abort()  # /clear：取消 a 流 + 置 _aborted
    ctrl.history.clear()  # 在 a 的 _process finally 之前跑（真实 /clear 路径）
    await task  # a 的 _process finally 在此后才跑
    assert len(ctrl.history) == 0, "aborted turn 复活成孤立 user turn"


@pytest.mark.asyncio
async def test_submit_while_busy_enqueues():
    gate = asyncio.Event()
    p = GatedProvider(gate)
    events = []

    async def on_event(ev):
        events.append(ev)

    async def on_status():
        pass

    ctrl = TurnController(p, on_event=on_event, on_status=on_status)

    task = asyncio.create_task(ctrl.submit("first"))
    await asyncio.sleep(0)  # let it start and block on gate
    assert ctrl.busy is True

    await ctrl.submit("second")  # busy -> enqueue
    assert ctrl.queue_size == 1

    gate.set()
    await task
    # drain happens after first turn; second turn completes too
    assert ctrl.busy is False
    assert len(ctrl.history) == 2


@pytest.mark.asyncio
async def test_interrupt_marks_turn_and_emits_interrupted():
    gate = asyncio.Event()
    p = GatedProvider(gate)
    events = []

    async def on_event(ev):
        events.append(ev)

    async def on_status():
        pass

    ctrl = TurnController(p, on_event=on_event, on_status=on_status)
    task = asyncio.create_task(ctrl.submit("x"))
    await asyncio.sleep(0)
    assert ctrl.busy is True

    ctrl.interrupt()
    await task

    assert ctrl.busy is False
    assert any(isinstance(e, Interrupted) for e in events)
    assert ctrl.history[-1].interrupted is True


@pytest.mark.asyncio
async def test_empty_submit_ignored():
    p = FakeProvider([])

    async def on_event(ev): ...
    async def on_status(): ...

    ctrl = TurnController(p, on_event=on_event, on_status=on_status)
    await ctrl.submit("   ")
    assert ctrl.busy is False and ctrl.history == []


@pytest.mark.asyncio
async def test_consume_degrades_when_callback_itself_fails():
    """#6 加固：若 on_event 回调自身抛异常，Error 重发不应再经同一路径逃脱 _consume。

    _process 只接 CancelledError，一旦异常从 _consume 冒泡就会穿到后台轮次任务
    （= 未检索异常）。故回调故障时 Error 重发应降级为记录，而非再次抛出。
    """

    async def on_event(ev):
        raise RuntimeError("ui broken")

    async def on_status(): ...

    p = FakeProvider([TextDelta(text="hi"), Done(usage=TokenUsage())])
    ctrl = TurnController(p, on_event=on_event, on_status=on_status)
    await ctrl.submit("x")  # 不应抛出
    assert ctrl.busy is False


@pytest.mark.asyncio
async def test_set_provider_takes_effect_next_turn():
    """set_provider 下一轮生效；当前进行中的轮次仍用旧 provider。"""
    received = []

    async def on_event(ev):
        received.append(ev)

    async def on_status(): ...

    old = FakeProvider([TextDelta(text="old"), Done(usage=TokenUsage())])
    new = FakeProvider([TextDelta(text="new"), Done(usage=TokenUsage())])
    ctrl = TurnController(old, on_event=on_event, on_status=on_status)

    await ctrl.submit("first")  # 用 old
    ctrl.set_provider(new)
    await ctrl.submit("second")  # 用 new

    texts = [e.text for e in received if isinstance(e, TextDelta)]
    assert "old" in texts and "new" in texts


@pytest.mark.asyncio
async def test_tool_roundtrip_end_to_end():
    """mock provider 首轮发起 echo → executor 执行 → tool_result 回传 → 第 2 轮最终回复。"""
    from birdcode.agent.mock_provider import MockProvider
    from birdcode.agent.provider import ToolCallStart, ToolResult
    from birdcode.blocks import ToolResultBlock, ToolUseBlock
    from birdcode.tools.executor import ToolExecutor, default_registry

    received = []

    async def on_event(ev):
        received.append(ev)

    async def on_status(): ...

    ctrl = TurnController(
        MockProvider(delay=0),
        on_event=on_event,
        on_status=on_status,
        executor=ToolExecutor(default_registry()),
    )
    await ctrl.submit("hello world")

    assert len(ctrl.history) == 1
    msgs = ctrl.history[0].messages
    assert len(msgs) == 4  # user → assistant(tool_use) → user(tool_result) → assistant(text)
    assert any(isinstance(b, ToolUseBlock) for b in msgs[1].content)
    tr = next(b for b in msgs[2].content if isinstance(b, ToolResultBlock))
    assert tr.content == "hello world" and tr.is_error is False
    assert any(isinstance(e, ToolCallStart) for e in received)
    assert any(isinstance(e, ToolResult) for e in received)
    # turn.usage restored by the carry-over fix (last round's Done carries usage)
    assert ctrl.history[0].usage is not None


# ---- Task 7: TurnController 持有 SessionStore + resume + 落盘 user/assistant ----


@pytest.mark.asyncio
async def test_controller_appends_user_and_assistant_to_store(tmp_path):
    """controller 每条 Message 完成 → store.append;store 持久化 user + assistant。"""
    import json

    from birdcode.session import paths as _paths
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    ctx = SessionContext(session_id="s1", cwd="C:/p", version="0.1.0", git_branch=None)
    store = SessionStore(ctx, tmp_path, root=tmp_path)

    async def on_status(): ...

    ctrl = TurnController(
        FakeProvider([TextDelta(text="hi"), Done(usage=TokenUsage())]),
        on_event=_noop_event, on_status=on_status, store=store,
    )
    await ctrl.submit("hello")
    store.close()

    # 读回:应有 user(hello) + assistant(hi) 两行
    jf = tmp_path / _paths.encode_cwd(tmp_path) / "s1.jsonl"
    lines = jf.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    types = [r["type"] for r in rows]
    assert "user" in types and "assistant" in types


@pytest.mark.asyncio
async def test_controller_resume_loads_history_and_sets_leaf(tmp_path):
    """resume():load_mainline 填 history;后续 append 接在叶子后(链不重置)。"""
    import json

    from birdcode.blocks import TextBlock
    from birdcode.conversation import Message
    from birdcode.session import paths as _paths
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    ctx = SessionContext(session_id="s1", cwd="C:/p", version="0.1.0", git_branch=None)
    # 预置一个会话
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="旧问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="旧答")]), is_assistant=True
    )
    store.close()

    # 新 controller resume
    store2 = SessionStore(ctx, tmp_path, root=tmp_path)
    ctrl = TurnController(
        FakeProvider([TextDelta(text="新答"), Done(usage=TokenUsage())]),
        on_event=_noop_event, on_status=_noop_status, store=store2,
    )
    turns = await ctrl.resume()
    assert len(turns) == 1 and len(ctrl.history) == 1
    # 新轮 append 接在叶子后
    await ctrl.submit("新问")
    store2.close()
    jf2 = tmp_path / _paths.encode_cwd(tmp_path) / "s1.jsonl"
    rows = [
        json.loads(line)
        for line in jf2.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 旧2行 + 新轮(user+assistant)2行 = 4;且第3行 parentUuid == 第2行 uuid
    assert len(rows) == 4
    assert rows[2]["parentUuid"] == rows[1]["uuid"]


@pytest.mark.asyncio
async def test_controller_without_store_works_as_before():
    """store=None(旧路径/无持久化)→ 行为不变,不 append。"""
    ctrl = TurnController(
        FakeProvider([TextDelta(text="hi"), Done(usage=TokenUsage())]),
        on_event=_noop_event, on_status=_noop_status,
    )
    await ctrl.submit("hello")
    assert len(ctrl.history) == 1


async def _noop_event(ev):  # noqa: ARG001 - 占位回调
    return None


async def _noop_status():
    return None


@pytest.mark.asyncio
async def test_receive_runs_turn_with_sender_prefix():
    """TurnController.receive(MailboxMessage) → 后台起一轮,Turn 含'[来自 sender]:'。

    receive 仿 notify_wake(后台 _drain,不阻塞投递方);teammate 消息走 str→_process,
    与 user submit / _WakeInput 通道互不干扰。
    """
    received = []

    async def on_event(ev):
        received.append(ev)

    async def on_status(): ...

    p = FakeProvider([TextDelta(text="ok"), Done(usage=TokenUsage())])
    ctrl = TurnController(p, on_event=on_event, on_status=on_status)
    ctrl.receive(MailboxMessage(sender="bob", to="lead", content="done"))
    for _ in range(100):  # receive 后台起 drain → 轮询 history 直到轮落地
        if len(ctrl.history) >= 1:
            break
        await asyncio.sleep(0.01)
    assert len(ctrl.history) == 1
    user_text = ctrl.history[0].messages[0].content[0].text
    assert "[来自 bob]" in user_text and "done" in user_text
    assert ctrl.busy is False


@pytest.mark.asyncio
async def test_receive_drain_runs_as_lead_not_teammate():
    """#2 回归:receive 经 teammate task 调 create_task(_drain),新 task 复制投递方的 _AGENT_NAME。

    若 _drain 不钉回 "lead",lead 本轮(由 receive 唤醒)的 provider/工具会把 sender 误归属到
    投递方 teammate。本测试在 _AGENT_NAME="teammate_bob" 的上下文里调 receive,断言 lead 轮
    provider 读到的 _AGENT_NAME 是 "lead"(证明 _drain 内 set("lead") 生效,且只影响 drain task
    的 context 副本,未污染投递方)。
    """
    from birdcode.agents.mailbox import _AGENT_NAME

    captured: list[str] = []

    class _NameProbe:
        async def stream(self, messages, *, history):  # noqa: ARG002
            captured.append(_AGENT_NAME.get())
            yield TextDelta(text="ok")
            yield Done(usage=TokenUsage())

    ctrl = TurnController(_NameProbe(), on_event=_noop_event, on_status=_noop_status)
    token = _AGENT_NAME.set("teammate_bob")  # 模拟 teammate 上下文调用 deliver(=receive)
    try:
        ctrl.receive(MailboxMessage(sender="bob", to="lead", content="done"))
    finally:
        _AGENT_NAME.reset(token)
    for _ in range(100):  # 等 receive 的后台 drain 跑完一轮(provider 记下 _AGENT_NAME)
        if captured:
            break
        await asyncio.sleep(0.01)
    assert captured == ["lead"], "lead 轮的 _AGENT_NAME 应钉为 lead,而非投递方 teammate"


@pytest.mark.asyncio
async def test_receive_does_not_inflate_user_count():
    """#8:receive(peer 消息)用 _PeerInput,不计 _user_count(对比 user submit 计)。

    状态栏 queue_size 只含用户文本;teammate→lead 消息不虚增(忙时若干 teammate 发消息
    不应误显为用户在排队输入)。
    """
    gate = asyncio.Event()
    ctrl = TurnController(GatedProvider(gate), on_event=_noop_event, on_status=_noop_status)
    task = asyncio.create_task(ctrl.submit("user1"))  # 占位 busy
    await asyncio.sleep(0)
    assert ctrl.busy
    assert ctrl.queue_size == 0

    ctrl.receive(MailboxMessage(sender="bob", to="lead", content="hi"))
    assert ctrl.queue_size == 0  # peer 消息不计

    await ctrl.submit("user2")  # busy → 入队 str
    assert ctrl.queue_size == 1  # user 文本计

    gate.set()
    await task


@pytest.mark.asyncio
async def test_begin_shutdown_suppresses_notify_wake_and_receive():
    """#1 回归:begin_shutdown 后 notify_wake/receive 不起后台 drain(免退出期跑 run_agent_loop)。

    d37798c 的 join_all 让退出收尾链(cancel_all→join_all→_supervise→_inject→notify_wake)确定
    触发 notify_wake;空闲态会 create_task(_drain)→_process_wake→run_agent_loop——一个 on_unmount
    既不 await 也不 cancel 的 orphan drain,在退出收尾的 MCP/store await 期跑 LLM 轮 + 工具。
    修后 _shutting_down 门控:notify_wake/receive 直接 return,不起 _drain_task、不置 busy、不发事件。
    """
    received = []

    async def on_event(ev):
        received.append(ev)

    p = FakeProvider([TextDelta(text="x"), Done(usage=TokenUsage())])
    ctrl = TurnController(p, on_event=on_event, on_status=_noop_status)
    ctrl.begin_shutdown()
    assert ctrl._shutting_down is True  # noqa: SLF001

    ctrl.notify_wake()  # 退出态:no-op
    assert ctrl._drain_task is None and ctrl.busy is False  # noqa: SLF001
    ctrl.receive(MailboxMessage(sender="t", to="lead", content="hi"))  # 同样 no-op
    assert ctrl._drain_task is None and ctrl.busy is False  # noqa: SLF001
    assert received == []  # 没跑 turn(无 TurnStart/TextDelta)
