# tests/test_wake.py
"""Task 8:TurnController 唤醒调度 — notify_wake/_process_wake/_consume_wake。

覆盖三条路径(真实 TurnController + 真实 SessionStore + fake provider):
  (a) 空闲唤醒:注入 isTaskNotification user 行 + enqueue(无 dequeue)→ notify_wake()
      → 后台 drain 自动跑一轮 → 产出 assistant 行 + 写 dequeue。
  (b) 正忙排队:submit 一个慢轮时 notify_wake → _WAKE 入队,当前轮后才消费。
  (c) 已消费不再唤醒:dequeue 已存在 → find_unresponded 返回空 → _process_wake no-op。

关键不变量:通知 user 行由 manager._inject 已落盘,_process_wake 绝不重复 append。
"""

from __future__ import annotations

import asyncio

import pytest

from birdcode.agent.provider import Done, TextDelta, TokenUsage
from birdcode.blocks import TextBlock
from birdcode.conversation import Message, TurnController
from birdcode.session.codec import find_unresponded_notifications
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore


class _TextProvider:
    """固定产一条 TextDelta("收到") + Done 的流式 provider。"""

    def __init__(self, profile: object | None = None) -> None:
        self.profile = profile

    def stream(self, messages: object, *, history: object):  # noqa: ARG002
        async def _gen():
            yield TextDelta(text="收到")
            yield Done(usage=TokenUsage(input_tokens=1, output_tokens=1), stop_reason="end_turn")

        return _gen()


class _GatedProvider:
    """yield 一条 delta,阻塞在 gate,放行后 yield Done(模拟慢轮,测排队)。"""

    def __init__(self, gate: asyncio.Event) -> None:
        self.gate = gate

    async def stream(self, messages: object, *, history: object):  # noqa: ARG002
        yield TextDelta(text="...")
        await self.gate.wait()
        yield Done(usage=TokenUsage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


class _RaisingProvider:
    """yield 一条 delta 后抛异常(模拟 provider 超时/限流/鉴权失败 → Error 路径)。"""

    async def stream(self, messages: object, *, history: object):  # noqa: ARG002
        yield TextDelta(text="partial")
        raise RuntimeError("boom")


async def _noop_event(_ev: object) -> None: ...


async def _noop_status() -> None: ...


def _make_store(tmp_path) -> SessionStore:
    ctx = SessionContext(session_id="wake", cwd="C:/p", version="0.1.0", git_branch=None)
    return SessionStore(ctx, tmp_path, root=tmp_path)


async def _inject_notification(
    store: SessionStore,
    *,
    agent_id: str,
    text: str,
    status: str = "completed",
    already_dequeued: bool = False,
) -> None:
    """模拟 SubagentManager._inject:写 enqueue queue-op + isTaskNotification user 行。

    already_dequeued=True 时再补一条 dequeue(模拟已被消费,测 no-op 路径)。
    """
    await store.append_queue_operation(
        operation="enqueue",
        agent_id=agent_id,
        tool_use_id=f"tu-{agent_id}",
        status=status,
    )
    await store.append(
        Message(role="user", content=[TextBlock(text=text)]),
        is_task_notification=True,
        agent_id=agent_id,
    )
    if already_dequeued:
        await store.append_queue_operation(
            operation="dequeue",
            agent_id=agent_id,
            tool_use_id="",
            status=status,
        )


async def _wait_until_idle(ctrl: TurnController, *, timeout: float = 2.0) -> None:
    """notify_wake 是 fire-and-forget(后台 _drain);轮询 busy 直到 False。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while ctrl.busy:
        if loop.time() > deadline:
            raise TimeoutError("TurnController 未在限时内回到空闲")
        await asyncio.sleep(0.005)


def _queue_ops(rows: list[dict]) -> list[dict]:
    return [r for r in rows if isinstance(r, dict) and r.get("type") == "queue-operation"]


# ---- (a) 空闲唤醒:自动跑一轮 + 写 dequeue ----


@pytest.mark.asyncio
async def test_wake_idle_runs_turn_and_writes_dequeue(tmp_path):
    store = _make_store(tmp_path)
    try:
        await _inject_notification(
            store, agent_id="a1", text="<task-notification>子agent完成</task-notification>"
        )

        ctrl = TurnController(
            _TextProvider(),  # type: ignore[arg-type]
            on_event=_noop_event,
            on_status=_noop_status,
            store=store,
        )

        # 唤醒前:一条待响应通知
        assert len(find_unresponded_notifications(store.mainline_rows())) == 1

        ctrl.notify_wake()  # 空闲 → 后台 drain
        await _wait_until_idle(ctrl)

        # 产出 assistant 行,history 多了一个 wake 轮
        assert ctrl.busy is False
        assert len(ctrl.history) == 1
        wake_turn = ctrl.history[-1]
        assert any(m.role == "assistant" for m in wake_turn.messages)
        # 通知 user 行读进了 turn.messages(不重复 append,store 里仍只有一条 a1 通知行)
        notif_rows = [
            r
            for r in store.mainline_rows()
            if isinstance(r, dict)
            and r.get("type") == "user"
            and r.get("isTaskNotification")
            and r.get("agentId") == "a1"
        ]
        assert len(notif_rows) == 1, "通知 user 行不应被 _process_wake 重复落盘"

        # dequeue 已写:find_unresponded 现在返回空
        assert find_unresponded_notifications(store.mainline_rows()) == []
        ops = _queue_ops(store.mainline_rows())
        assert any(o.get("operation") == "dequeue" and o.get("agentId") == "a1" for o in ops)
    finally:
        store.close()


# ---- (b) 正忙时排队:当前轮后才消费 ----


@pytest.mark.asyncio
async def test_wake_while_busy_queues_behind_current_turn(tmp_path):
    store = _make_store(tmp_path)
    try:
        gate = asyncio.Event()
        ctrl = TurnController(
            _GatedProvider(gate),  # type: ignore[arg-type]
            on_event=_noop_event,
            on_status=_noop_status,
            store=store,
        )

        # 起一个慢轮(submit 空闲 → 同步驱动 drain,阻塞在 gate)
        task = asyncio.create_task(ctrl.submit("first"))
        await asyncio.sleep(0)
        assert ctrl.busy is True

        # 注入通知 + 唤醒(正忙 → _WAKE 入队,不新起 drain)
        await _inject_notification(
            store, agent_id="a2", text="<task-notification>a2完成</task-notification>"
        )
        ctrl.notify_wake()
        # _WAKE 已入队(排在 first 之后,放行后由同一 drain 消费)。#11 起 _WAKE 不计入
        # 用户排队计数(queue_size 只数 str),故此处为 0;_WAKE 真被处理由下方
        # history==2 + a2 的 dequeue 共同验证。
        assert ctrl.queue_size == 0, "无用户 type-ahead;_WAKE 不计入 queue_size(#11)"

        # 放行 first → drain 继续 → 消费 _WAKE(submit await 整个 drain,含 wake)
        gate.set()
        await task

        assert ctrl.busy is False
        # first 轮 + wake 轮
        assert len(ctrl.history) == 2
        # wake 已消费:写 dequeue
        assert find_unresponded_notifications(store.mainline_rows()) == []
        ops = _queue_ops(store.mainline_rows())
        assert any(o.get("operation") == "dequeue" and o.get("agentId") == "a2" for o in ops)
    finally:
        store.close()


# ---- (c) 已 dequeue 的不再唤醒(no-op)----


@pytest.mark.asyncio
async def test_wake_with_already_dequeued_is_noop(tmp_path):
    store = _make_store(tmp_path)
    try:
        # 注入通知,但已写 dequeue(模拟已被消费)
        await _inject_notification(
            store,
            agent_id="a3",
            text="<task-notification>a3完成</task-notification>",
            already_dequeued=True,
        )
        assert find_unresponded_notifications(store.mainline_rows()) == []

        ctrl = TurnController(
            _TextProvider(),  # type: ignore[arg-type]
            on_event=_noop_event,
            on_status=_noop_status,
            store=store,
        )

        ctrl.notify_wake()
        await _wait_until_idle(ctrl)

        # 无待响应 → _process_wake no-op:不产 assistant,不污染 history,不多写 dequeue
        assert ctrl.busy is False
        assert len(ctrl.history) == 0
        ops = _queue_ops(store.mainline_rows())
        # 仅 (c) 注入的那一条 dequeue,没有多写
        dequeues = [o for o in ops if o.get("operation") == "dequeue"]
        assert len(dequeues) == 1
    finally:
        store.close()


# ---- (#2) wake 轮 provider 异常:不消费通知、不写 dequeue ----


@pytest.mark.asyncio
async def test_wake_provider_error_keeps_notification_unresponded(tmp_path):
    """#2:wake 轮 provider 抛异常(超时/限流/5xx)→ 不写 dequeue、不污染 history。

    通知保持「待响应」(find_unresponded 仍返回),下轮可重试;而非被永久消费、永不重唤。
    (Esc 中断仍写 dequeue 是已接受的 M3;此处只针对 provider 异常路径。)
    """
    store = _make_store(tmp_path)
    try:
        await _inject_notification(
            store, agent_id="a1", text="<task-notification>x</task-notification>"
        )
        ctrl = TurnController(
            _RaisingProvider(),  # type: ignore[arg-type]
            on_event=_noop_event,
            on_status=_noop_status,
            store=store,
        )

        ctrl.notify_wake()
        await _wait_until_idle(ctrl)

        assert ctrl.busy is False
        # 关键:没写 dequeue → 通知仍待响应(可重试)
        assert find_unresponded_notifications(store.mainline_rows()) != [], (
            "provider 异常不应消费通知"
        )
        ops = _queue_ops(store.mainline_rows())
        assert not any(o.get("operation") == "dequeue" for o in ops), "异常轮不应写 dequeue"
        # 失败轮不入 history(避免下轮上下文出现孤立 user turn)
        assert len(ctrl.history) == 0
    finally:
        store.close()


# ---- (#3) notify_wake 持有 drain task 引用(防 GC + 可观测)----


@pytest.mark.asyncio
async def test_notify_wake_holds_drain_task_reference(tmp_path):
    """#3:notify_wake 空闲时后台起 _drain,必须持有 task 引用(防 asyncio GC 中断执行)。

    引用在 drain 完成后被清理(不泄漏)。用 _GatedProvider 阻塞 wake 轮使 drain task 存活,
    从而能断言引用被持有且 task 未完成。
    """
    store = _make_store(tmp_path)
    try:
        gate = asyncio.Event()
        ctrl = TurnController(
            _GatedProvider(gate),  # type: ignore[arg-type]
            on_event=_noop_event,
            on_status=_noop_status,
            store=store,
        )
        await _inject_notification(
            store, agent_id="a1", text="<task-notification>x</task-notification>"
        )
        ctrl.notify_wake()
        await asyncio.sleep(0)  # 让后台 drain 起来并阻塞在 gate

        # 引用被持有(非 None)、task 仍在运行(阻塞在 gate)
        assert ctrl._drain_task is not None
        assert not ctrl._drain_task.done()

        gate.set()  # 放行 → wake 轮完成 → drain 收尾
        await _wait_until_idle(ctrl)
        await asyncio.sleep(0)  # 让 done-callback 清理引用

        # 完成后引用被清理(不泄漏、不持已死 task)
        assert ctrl._drain_task is None
    finally:
        store.close()
