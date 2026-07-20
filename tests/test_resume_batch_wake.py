# tests/test_resume_batch_wake.py
"""resume 批量注入:多个未通知 async 子 agent 的续跑提示逐个落盘(各自 agentId),
最后一次 notify_wake——所有提示在同一次 _process_wake 前落盘,主 agent 一个 turn 看到全部。

regression:_inject_async_resume_hint 逐个 append+wake 时,先到的 wake 触发第一轮只消费
部分提示 = N 轮往返(ee6e8090 时序问题)。现拆 _append_async_resume_hint(只落盘),
循环外 wake 一次。
"""

from __future__ import annotations

import pytest

from birdcode.ui.app import BirdApp


class _RowsStore:
    """轻量 store:mainline_rows 返回构造行;append 记录(含 is_task_notification/agent_id)。"""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
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


def _async_tooluse_row(*pairs: tuple[str, str]) -> dict:
    """assistant 行:含若干 async tool_use(pairs=(tool_use_id, agent_id))。"""
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tid,
                    "name": "explore",
                    "input": {"run_in_background": True},
                    "agent_id": aid,
                }
                for tid, aid in pairs
            ],
        },
    }


def _bare_app(store: _RowsStore, ctrl: _WakeCtrl) -> BirdApp:
    """绕过 BirdApp.__init__(只填 _resume_pending_notifications 依赖的 _store/_controller)。"""
    app = BirdApp.__new__(BirdApp)
    app._store = store  # type: ignore[attr-defined]
    app._controller = ctrl  # type: ignore[attr-defined]
    return app


@pytest.mark.asyncio
async def test_resume_batches_multiple_async_into_single_wake():
    """2 个未通知 async → 落盘 2 条 hint(各自 agentId)+ 只 wake 一次。"""
    rows = [_async_tooluse_row(("c1", "sub-a"), ("c2", "sub-b"))]
    store, ctrl = _RowsStore(rows), _WakeCtrl()
    app = _bare_app(store, ctrl)
    await app._resume_pending_notifications()
    assert len(store.appends) == 2
    assert {kw.get("agent_id") for _, kw in store.appends} == {"sub-a", "sub-b"}
    assert all(kw.get("is_task_notification") for _, kw in store.appends)
    assert ctrl.wakes == 1  # 批量合并:只 wake 一次


@pytest.mark.asyncio
async def test_resume_noop_when_all_async_already_notified():
    """async 已有完成通知(isTaskNotification 行 agentId)→ 不注入、不 wake。"""
    rows = [
        _async_tooluse_row(("c1", "sub-a")),
        {
            "type": "user",
            "isTaskNotification": True,
            "agentId": "sub-a",
            "message": {"role": "user", "content": [{"type": "text", "text": "done"}]},
        },
    ]
    store, ctrl = _RowsStore(rows), _WakeCtrl()
    app = _bare_app(store, ctrl)
    await app._resume_pending_notifications()
    assert store.appends == []
    assert ctrl.wakes == 0


@pytest.mark.asyncio
async def test_resume_no_async_noop():
    """无 async tool_use → 不注入、不 wake。"""
    rows = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "c1", "name": "explore", "input": {}}],
            },
        }
    ]
    store, ctrl = _RowsStore(rows), _WakeCtrl()
    app = _bare_app(store, ctrl)
    await app._resume_pending_notifications()
    assert store.appends == []
    assert ctrl.wakes == 0
