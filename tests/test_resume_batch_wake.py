# tests/test_resume_batch_wake.py
"""_resume_pending_notifications 扫 subagents meta(非主线 input):is_async + 未 handled
→ 逐个落盘 hint,最后一次 wake(打包,N→1)。

handled = 主线有该 agentId 的 dequeue(正常消费 / resume mark_resumed 标记)或 completed/error
enqueue(完成通知);cancelled 不算(中断≠完成,可能需续跑)。扫 meta.is_async 是 runner 写的
真相,覆盖 worktree 强制异步 + defn-default async(不靠 input.run_in_background)。
"""

from __future__ import annotations

import types

import pytest

from birdcode.session.models import SubagentMeta
from birdcode.session.paths import subagent_meta_path
from birdcode.session.subagent_meta import write_subagent_meta
from birdcode.ui.app import BirdApp


class _RowsStore:
    """轻量 store:mainline_rows 返回构造行(供 handled 扫描);append 记录 hint。"""

    def __init__(
        self,
        rows: list[dict],
        *,
        root: object,
        project_root: object,
        session_id: str = "s1",
        worktree_name: str | None = None,
    ) -> None:
        self._rows = rows
        self.root = root
        self.project_root = project_root
        self.ctx = types.SimpleNamespace(session_id=session_id)
        self.worktree_name = worktree_name
        self.appends: list[tuple[object, dict[str, object]]] = []

    def mainline_rows(self) -> list[dict]:
        return self._rows

    async def append(self, msg: object, **kw: object) -> None:
        self.appends.append((msg, kw))


class _WakeCtrl:
    def __init__(self) -> None:
        self.wakes = 0

    def notify_wake(self) -> None:
        self.wakes += 1


def _bare_app(store: _RowsStore, ctrl: _WakeCtrl) -> BirdApp:
    app = BirdApp.__new__(BirdApp)
    app._store = store  # type: ignore[attr-defined]
    app._controller = ctrl  # type: ignore[attr-defined]
    return app


def _write_meta(store: _RowsStore, aid: str, *, is_async: bool, status: str = "running") -> None:
    write_subagent_meta(
        subagent_meta_path(
            store.root,
            store.ctx.session_id,
            store.project_root,
            aid,
            worktree_name=store.worktree_name,
        ),
        SubagentMeta(
            agent_id=aid,
            agent_type="explore",
            tool_use_id="(async-agent)",
            is_async=is_async,
            status=status,
        ),
    )


def _qop(operation: str, status: str, agent_id: str = "sub-a") -> dict:
    """构造 queue-operation 行(测试主线 handled 扫描用)。"""
    return {
        "type": "queue-operation",
        "operation": operation,
        "agentId": agent_id,
        "status": status,
    }


@pytest.mark.asyncio
async def test_resume_scans_async_meta_batches_wake(tmp_path):
    """2 个 async meta(未 handled)→ 2 hint(各自 agentId)+ 只 wake 一次(打包)。"""
    store = _RowsStore([], root=tmp_path, project_root=tmp_path)
    _write_meta(store, "sub-a", is_async=True)
    _write_meta(store, "sub-b", is_async=True)
    ctrl = _WakeCtrl()
    await _bare_app(store, ctrl)._resume_pending_notifications()
    assert len(store.appends) == 2
    assert {kw.get("agent_id") for _, kw in store.appends} == {"sub-a", "sub-b"}
    assert all(kw.get("is_task_notification") for _, kw in store.appends)
    assert ctrl.wakes == 1


@pytest.mark.asyncio
async def test_resume_handled_completed_enqueue_skipped(tmp_path):
    """主线有 completed enqueue(X)→ handled → X 不 hint(完成通知已落,#1/#3 收敛)。"""
    rows = [_qop("enqueue", "completed")]
    store = _RowsStore(rows, root=tmp_path, project_root=tmp_path)
    _write_meta(store, "sub-a", is_async=True, status="completed")
    ctrl = _WakeCtrl()
    await _bare_app(store, ctrl)._resume_pending_notifications()
    assert store.appends == []
    assert ctrl.wakes == 0


@pytest.mark.asyncio
async def test_resume_handled_dequeue_skipped(tmp_path):
    """主线有 dequeue(X)→ handled(resume mark_resumed 标记 / 正常消费)→ 不 hint(#3 收敛)。"""
    rows = [_qop("dequeue", "cancelled")]
    store = _RowsStore(rows, root=tmp_path, project_root=tmp_path)
    _write_meta(store, "sub-a", is_async=True, status="cancelled")
    ctrl = _WakeCtrl()
    await _bare_app(store, ctrl)._resume_pending_notifications()
    assert store.appends == []
    assert ctrl.wakes == 0


@pytest.mark.asyncio
async def test_resume_cancelled_enqueue_still_hinted(tmp_path):
    """cancelled enqueue 不算 handled(中断≠完成)→ 仍 hint(让模型 resume_agent 续跑)。"""
    rows = [_qop("enqueue", "cancelled")]
    store = _RowsStore(rows, root=tmp_path, project_root=tmp_path)
    _write_meta(store, "sub-a", is_async=True, status="cancelled")
    ctrl = _WakeCtrl()
    await _bare_app(store, ctrl)._resume_pending_notifications()
    assert len(store.appends) == 1
    assert ctrl.wakes == 1


@pytest.mark.asyncio
async def test_resume_sync_meta_not_scanned(tmp_path):
    """sync meta(is_async=False)不扫(sync 中断走 tool_result 占位,不经 notification hint)。"""
    store = _RowsStore([], root=tmp_path, project_root=tmp_path)
    _write_meta(store, "sub-a", is_async=False)
    ctrl = _WakeCtrl()
    await _bare_app(store, ctrl)._resume_pending_notifications()
    assert store.appends == []
    assert ctrl.wakes == 0
