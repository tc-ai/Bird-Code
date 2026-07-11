# tests/ui/test_resume_notifications.py
"""Task 10:BirdApp._resume_pending_notifications — resume 投递补全 + 重注入未响应唤醒。

聚焦 _resume_pending_notifications 单方法(真实 SessionStore + spy controller),
避开构造完整 BirdApp/Textual App:用未绑定方法 + SimpleNamespace 模拟 self(与
test_subagent_card 直接调 SubagentCard._format_duration 同风格)。

覆盖四条路径:
  (1) 投递补全:enqueue 无 user 行 → 重构 isTaskNotification user 行写回 + 触发 wake
      (补投后无 dequeue → find_unresponded 非空)。
  (2) 已投递未响应:enqueue + user 行(无 dequeue)→ find_pending 空(不重复补投)+ wake。
  (3) 已消费:enqueue + user + dequeue → 既不补投也不 wake。
  (4) 守卫:无 store / 无 controller → 早返回(no-op)。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from birdcode.blocks import TextBlock
from birdcode.conversation import Message
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore
from birdcode.ui.app import BirdApp


def _make_store(tmp_path) -> SessionStore:
    ctx = SessionContext(session_id="resume", cwd="C:/p", version="0.1.0", git_branch=None)
    return SessionStore(ctx, tmp_path, root=tmp_path)


class _SpyController:
    """仅记录 notify_wake 调用次数(无需真实 TurnController/queue/drain)。"""

    def __init__(self) -> None:
        self.wake_calls = 0

    def notify_wake(self) -> None:
        self.wake_calls += 1


def _notification_texts(rows: list[dict]) -> list[str]:
    """提取主线中 isTaskNotification user 行的文本(每行首块 text)。"""
    out: list[str] = []
    for r in rows:
        if r.get("type") != "user" or not r.get("isTaskNotification"):
            continue
        blocks = (r.get("message") or {}).get("content") or []
        if blocks and isinstance(blocks[0], dict):
            out.append(blocks[0].get("text") or "")
    return out


def _write_sidechain(store, agent_id: str, text: str) -> None:
    """写一条最小侧链 jsonl(user + assistant),供 _reconstruct_subagent_notification 重读。"""
    import json

    from birdcode.session import codec, paths

    p = paths.subagent_jsonl_path(store.root, store.ctx.session_id, store.project_root, agent_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    ctx = store.ctx
    rows = [
        codec.encode_user(
            Message(role="user", content=[TextBlock(text="go")]),
            uuid="u1", parent_uuid=None, ctx=ctx, timestamp="t",
        ),
        codec.encode_assistant(
            Message(role="assistant", content=[TextBlock(text=text)]),
            uuid="a1", parent_uuid="u1", ctx=ctx, timestamp="t",
        ),
    ]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.asyncio
async def test_resume_completes_pending_and_wakes(tmp_path):
    """enqueue 无 user 行 → 从侧链重构 user 行写回 + notify_wake(补投后仍无 dequeue)。"""
    store = _make_store(tmp_path)
    _write_sidechain(store, "A1", "子 agent A1 完成")
    await store.append_queue_operation(
        operation="enqueue",
        agent_id="A1",
        tool_use_id="tu-A1",
        status="completed",
    )
    spy = _SpyController()
    fake_self = SimpleNamespace(_store=store, _controller=spy)

    await BirdApp._resume_pending_notifications(fake_self)

    texts = _notification_texts(store.mainline_rows())
    assert len(texts) == 1 and "子 agent A1 完成" in texts[0]  # 侧链重读出的报告正文
    assert spy.wake_calls == 1


@pytest.mark.asyncio
async def test_resume_unresponded_wakes_without_reappend(tmp_path):
    """enqueue + user 行(无 dequeue)→ find_pending 空(不重复补投)+ wake。"""
    store = _make_store(tmp_path)
    await store.append_queue_operation(
        operation="enqueue",
        agent_id="A2",
        tool_use_id="tu-A2",
        status="completed",
    )
    await store.append(
        Message(role="user", content=[TextBlock(text="A2 done")]),
        is_task_notification=True,
        agent_id="A2",
    )
    spy = _SpyController()
    fake_self = SimpleNamespace(_store=store, _controller=spy)

    await BirdApp._resume_pending_notifications(fake_self)

    # 未重复补投:仍只有原 1 条 notification user 行
    assert _notification_texts(store.mainline_rows()) == ["A2 done"]
    assert spy.wake_calls == 1


@pytest.mark.asyncio
async def test_resume_dequeued_noop(tmp_path):
    """enqueue + user + dequeue → 既不补投也不 wake(find_unresponded 为空)。"""
    store = _make_store(tmp_path)
    await store.append_queue_operation(
        operation="enqueue",
        agent_id="A3",
        tool_use_id="tu-A3",
        status="completed",
    )
    await store.append(
        Message(role="user", content=[TextBlock(text="A3 done")]),
        is_task_notification=True,
        agent_id="A3",
    )
    await store.append_queue_operation(
        operation="dequeue",
        agent_id="A3",
        tool_use_id="tu-A3",
        status="completed",
    )
    spy = _SpyController()
    fake_self = SimpleNamespace(_store=store, _controller=spy)

    await BirdApp._resume_pending_notifications(fake_self)

    assert _notification_texts(store.mainline_rows()) == ["A3 done"]  # 未补投
    assert spy.wake_calls == 0


@pytest.mark.asyncio
async def test_resume_no_store_returns_early():
    """无 store → 早返回(不读 controller)。"""
    spy = _SpyController()
    fake_self = SimpleNamespace(_store=None, _controller=spy)
    await BirdApp._resume_pending_notifications(fake_self)  # 不应抛
    assert spy.wake_calls == 0


@pytest.mark.asyncio
async def test_resume_no_controller_returns_early(tmp_path):
    """无 controller → 早返回(即便有 pending enqueue 也不补投/唤醒)。"""
    store = _make_store(tmp_path)
    await store.append_queue_operation(
        operation="enqueue",
        agent_id="A4",
        tool_use_id="tu-A4",
        status="completed",
    )
    fake_self = SimpleNamespace(_store=store, _controller=None)
    await BirdApp._resume_pending_notifications(fake_self)
    assert _notification_texts(store.mainline_rows()) == []  # 早返回,未补投
