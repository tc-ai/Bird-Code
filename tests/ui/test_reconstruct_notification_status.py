# tests/ui/test_reconstruct_notification_status.py
"""Task 7: _reconstruct_subagent_notification 以 meta.status 为准(落点 A 前提)。

bug:caller 传 enqueue 行的 status(缺省 'completed'),函数原样用作 is_completed/status,
无视 meta.status——被中断(status=error/running/cancelled)的子 agent 通知被错标成「已完成」。
修复:status 优先取 meta.status;meta 缺失/读失败才回退 caller 传的 enqueue status,
再回退 'completed'。is_completed = (status == 'completed')。

用真实 SessionStore + 真实 meta/sidechain 文件(不 mock),覆盖:
  - error / cancelled:中断态 → Status:<state>, Completed:no
  - running:非终态(侧链未写完即 crash)→ is_completed=False、Status:running
  - completed:正常完成(回归)
  - meta 缺失:回退 caller 传的 enqueue status
  - 侧链缺失:report_text 占位,但 status 仍以 meta 为准
"""

from __future__ import annotations

import json

from birdcode.blocks import TextBlock
from birdcode.conversation import Message
from birdcode.session import codec, paths
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.store import SessionStore
from birdcode.session.subagent_meta import write_subagent_meta
from birdcode.ui.app import _reconstruct_subagent_notification


def _make_store(tmp_path) -> SessionStore:
    ctx = SessionContext(session_id="recon", cwd="C:/p", version="0.1.0", git_branch=None)
    return SessionStore(ctx, tmp_path, root=tmp_path)


def _write_sidechain(store: SessionStore, agent_id: str, text: str) -> None:
    """写一条最小侧链 jsonl(user + assistant),供 _read_sidechain_final_text 重读。"""
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


def test_reconstruct_honors_error_meta_status(tmp_path):
    """meta.status=error → Status:error + Completed:no(不再误标 completed)。

    caller 仍传缺省 'completed'(模拟旧 caller),函数应以 meta.status=error 为准。
    """
    store = _make_store(tmp_path)
    _write_meta(store, "A1", "error")
    _write_sidechain(store, "A1", "子 agent 失败的最终文本")

    notification = _reconstruct_subagent_notification(store, "A1", "completed")

    assert notification is not None
    assert "Status: error" in notification
    assert "Completed: no" in notification
    assert "子 agent 失败的最终文本" in notification


def test_reconstruct_honors_cancelled_meta_status(tmp_path):
    """meta.status=cancelled → Status:cancelled + Completed:no。"""
    store = _make_store(tmp_path)
    _write_meta(store, "A2", "cancelled")
    _write_sidechain(store, "A2", "取消的正文")

    notification = _reconstruct_subagent_notification(store, "A2", "completed")

    assert notification is not None
    assert "Status: cancelled" in notification
    assert "Completed: no" in notification


def test_reconstruct_non_terminal_running_status(tmp_path):
    """meta.status=running(crash 时仍在跑)→ is_completed=False、Status:running。

    非终态(launched/running/idle)说明 crash 打断了运行;notification 不标 completed。
    """
    store = _make_store(tmp_path)
    _write_meta(store, "A3", "running")
    _write_sidechain(store, "A3", "半截正文")

    notification = _reconstruct_subagent_notification(store, "A3", "completed")

    assert notification is not None
    assert "Status: running" in notification
    assert "Completed: no" in notification


def test_reconstruct_completed_meta_status(tmp_path):
    """meta.status=completed(正常完成)→ Status:completed + Completed:yes(回归)。"""
    store = _make_store(tmp_path)
    _write_meta(store, "A4", "completed")
    _write_sidechain(store, "A4", "完成的正文")

    notification = _reconstruct_subagent_notification(store, "A4", "completed")

    assert notification is not None
    assert "Status: completed" in notification
    assert "Completed: yes" in notification


def test_reconstruct_falls_back_when_meta_missing(tmp_path):
    """meta 缺失 → 回退 caller 传的 enqueue status(此处 completed)。"""
    store = _make_store(tmp_path)
    _write_sidechain(store, "A5", "回退正文")
    # 故意不写 meta 文件

    notification = _reconstruct_subagent_notification(store, "A5", "completed")

    assert notification is not None
    assert "Status: completed" in notification
    assert "Completed: yes" in notification
    # 缺失 meta → defn.name 缺省 'subagent'、description '子 agent'
    assert "type=subagent" in notification


def test_reconstruct_status_from_meta_even_when_sidechain_missing(tmp_path):
    """侧链缺失(报告正文占位)但 meta.status=error → Status 仍为 error(以 meta 为准)。"""
    store = _make_store(tmp_path)
    _write_meta(store, "A6", "error")
    # 故意不写侧链 jsonl → report_text 占位

    notification = _reconstruct_subagent_notification(store, "A6", "completed")

    assert notification is not None
    assert "Status: error" in notification
    assert "Completed: no" in notification
    assert "未能恢复" in notification  # 占位正文
