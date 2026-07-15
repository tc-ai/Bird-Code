# tests/ui/test_completed_uninjected_autoinject.py
"""Task 8: 完成未注入自动补发(免授权)——落点 A。

_resume_pending_notifications 对每个 pending enqueue 读 meta:
- meta 终态(completed/error/cancelled)→ 调 _reconstruct_subagent_notification 补发
  (免授权:这是 crash 恢复的投递补全,区别于续跑——续跑必须授权)。
- meta 非终态(launched/running/idle)→ **跳过**,不在此自动处理,交续跑路径
  (橙色提示/继续 路由,T11/T12;授权 T13)。meta 缺失/读失败视为终态(回退 enqueue
  行 status,通常是 completed)以维持旧行为,不破坏回归。

聚焦 _resume_pending_notifications 单方法(真实 SessionStore + spy controller),
复用 test_resume_notifications 的 SimpleNamespace 模拟 self 套路。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from birdcode.blocks import TextBlock
from birdcode.conversation import Message
from birdcode.session import codec, paths
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.store import SessionStore
from birdcode.session.subagent_meta import write_subagent_meta
from birdcode.ui.app import BirdApp


def _make_store(tmp_path) -> SessionStore:
    ctx = SessionContext(session_id="task8", cwd="C:/p", version="0.1.0", git_branch=None)
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


def _write_sidechain(store: SessionStore, agent_id: str, text: str) -> None:
    """写一条最小侧链 jsonl(user + assistant),供 _reconstruct_subagent_notification 重读。"""
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


def _write_meta(store: SessionStore, agent_id: str, status: str) -> None:
    p = paths.subagent_meta_path(store.root, store.ctx.session_id, store.project_root, agent_id)
    write_subagent_meta(
        p,
        SubagentMeta(agent_id=agent_id, tool_use_id=f"tu-{agent_id}", status=status),
    )


@pytest.mark.asyncio
async def test_completed_meta_with_enqueue_but_no_notification_autoinjects(tmp_path):
    """meta=completed + enqueue 无配套 user 行 → 自动补发 Status:completed(免授权)。

    落点 A:子 agent 实际已完成(终态)但主会话缺完成通知 → 免授权补发。
    """
    store = _make_store(tmp_path)
    _write_meta(store, "C1", "completed")
    _write_sidechain(store, "C1", "子 agent C1 已完成的报告正文")
    await store.append_queue_operation(
        operation="enqueue",
        agent_id="C1",
        tool_use_id="tu-C1",
        status="completed",
    )
    spy = _SpyController()
    fake_self = SimpleNamespace(_store=store, _controller=spy)

    await BirdApp._resume_pending_notifications(fake_self)

    texts = _notification_texts(store.mainline_rows())
    assert len(texts) == 1  # 补发了一条 isTaskNotification user 行
    assert "Status: completed" in texts[0]
    assert "子 agent C1 已完成的报告正文" in texts[0]
    # 补发后无 dequeue → 落入未响应轴 → wake
    assert spy.wake_calls == 1


@pytest.mark.asyncio
async def test_nonterminal_meta_not_autoinjected_here(tmp_path):
    """meta=running(非终态)+ enqueue 无 user 行 → 不在此自动补发(交续跑路径)。

    非终态(launched/running/idle)说明 crash 打断了运行,子 agent 未真正完成 →
    续跑路径(橙色提示/继续,需用户授权)接管,本处不自动注入。
    """
    store = _make_store(tmp_path)
    _write_meta(store, "N1", "running")
    _write_sidechain(store, "N1", "半截正文")
    await store.append_queue_operation(
        operation="enqueue",
        agent_id="N1",
        tool_use_id="tu-N1",
        status="completed",  # enqueue 行带 completed(旧 caller 缺省),但 meta=running 是权威
    )
    spy = _SpyController()
    fake_self = SimpleNamespace(_store=store, _controller=spy)

    await BirdApp._resume_pending_notifications(fake_self)

    assert _notification_texts(store.mainline_rows()) == []  # 未补发
    assert spy.wake_calls == 0  # 无未响应 → 不 wake
