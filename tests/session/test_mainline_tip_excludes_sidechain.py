# tests/session/test_mainline_tip_excludes_sidechain.py
"""#33651 防御:主线 tip 选择排除 sidechain / queue-op。

Claude Code #33651 场景:progress 链 parentUuid 分支时间戳更新、又没标 sidechain →
resume 选错 tip,真对话被孤儿丢失。BirdCode 对应四道防御(均已实现,本文件锁定):
  ① 子 agent 写入恒 isSidechain=true(subagent_store.append 强制);
  ② 主线读 _read_mainline_rows 跳 isSidechain 行(store.py:581-582);
  ③ _last_uuid 仅由主线 user/assistant 写更新,append_queue_operation 显式不推进;
  ④ tip 按主线 uuid(文件序反向首条带 uuid)推,非全局最新时间戳。

锁定目标:防 T2>T1 时间戳竞争让 sidechain 夺 tip、或 queue-op(无 uuid)夺 tip。
"""

import json
from pathlib import Path

from birdcode.blocks import TextBlock
from birdcode.conversation import Message
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore


def _ctx(sid="s1"):
    return SessionContext(
        session_id=sid,
        cwd="C:/proj",
        version="0.1.0",
        git_branch="main",
    )


def _store(tmp_path: Path, sid="s1") -> SessionStore:
    return SessionStore(_ctx(sid), tmp_path, root=tmp_path)


def _jsonl(tmp_path: Path, sid="s1") -> Path:
    from birdcode.session import paths

    return tmp_path / paths.encode_cwd(tmp_path) / f"{sid}.jsonl"


def _append_raw(jf: Path, obj: dict) -> None:
    with jf.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


async def test_sidechain_rows_excluded_from_mainline_tip(tmp_path):
    """#33651 核心(Step 1):主 jsonl 末尾混入 isSidechain=true 的子 agent 消息
    (时间戳更新、带 uuid)→ load_mainline 后 _last_uuid 仍是【最后一条非 sidechain
    主线行】,sidechain 不夺 tip。

    防御 ①+②+④:子 agent 行标 isSidechain、_read_mainline_rows 跳过、tip 按主线
    uuid 推(非全局最新时间戳)。
    """
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="主线问")]))
    mainline_tip_uuid = await store.append(
        Message(role="assistant", content=[TextBlock(text="主线答")]),
        is_assistant=True,
    )
    store.close()

    # 模拟 sidechain 行泄露进主 jsonl 末尾:时间戳更新(T2>T1)、带 uuid、isSidechain=true。
    sidechain_uuid = "sc-tip-hijack"
    _append_raw(
        _jsonl(tmp_path),
        {
            "type": "user",
            "isSidechain": True,
            "agentId": "a0685194cbfe9829",
            "uuid": sidechain_uuid,
            "parentUuid": mainline_tip_uuid,
            "sessionId": "s1",
            "timestamp": "2099-12-31T23:59:59Z",  # 远新于主线时间戳
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "子 agent 侧链消息"}],
            },
        },
    )

    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await s2.load_mainline()
    s2.close()

    # tip = 主线最后一条非 sidechain 行的 uuid,不是 sidechain 的 uuid
    assert s2._last_uuid == mainline_tip_uuid  # noqa: SLF001
    assert s2._last_uuid != sidechain_uuid  # noqa: SLF001
    # sidechain 行不出现在装载的 turns 中(被 _read_mainline_rows 跳过)
    texts = [b.text for t in turns for m in t.messages for b in m.content if hasattr(b, "text")]
    assert "主线答" in texts
    assert "子 agent 侧链消息" not in texts


async def test_tip_by_mainline_file_order_not_global_timestamp(tmp_path):
    """#33651 时间戳竞争:主线 T1(早)在前、sidechain T2(晚)在后且时间戳更新 →
    tip 仍取主线 T1,sidechain 时间戳最新也不夺 tip。

    旧 bug 用「全局最新时间戳」选 tip → T2>T1 使 sidechain 夺 tip,真主线被孤儿化。
    BirdCode 按主线文件序(反向首条带 uuid)推,忽略 sidechain 的时间戳。
    """
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="T1主线问")]))
    mainline_tip_uuid = await store.append(
        Message(role="assistant", content=[TextBlock(text="T1主线答")]),
        is_assistant=True,
    )
    store.close()

    # sidechain 时间戳更新(模拟 #33651 的 T2>T1 分支),带 uuid,但 isSidechain=true
    _append_raw(
        _jsonl(tmp_path),
        {
            "type": "assistant",
            "isSidechain": True,
            "agentId": "a-late-sidechain",
            "uuid": "late-sc-uuid",
            "parentUuid": mainline_tip_uuid,
            "sessionId": "s1",
            "timestamp": "2099-12-31T23:59:59Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "晚到 sidechain"}],
            },
        },
    )

    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await s2.load_mainline()
    s2.close()
    assert s2._last_uuid == mainline_tip_uuid  # noqa: SLF001 - 主线文件序末行,非时间戳最新


async def test_queue_operation_does_not_pollute_tip(tmp_path):
    """防御 ③:queue-op 事件行无 uuid、不进消息 DAG → 不夺 tip、不更新 _last_uuid。

    append_queue_operation 显式不推进 _last_uuid(docstring 明示「对消息 DAG 透明」);
    load_mainline 的 tip 选择 `if r.get("uuid")` 跳过无 uuid 行。两道门合保:即便
    queue-op 落在文件末尾,tip 仍是最后一条带 uuid 的主线对话行。
    """
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="主线提问")]))
    mainline_tip_uuid = await store.append(
        Message(role="assistant", content=[TextBlock(text="主线答")]),
        is_assistant=True,
    )
    await store.append_queue_operation(
        operation="enqueue",
        agent_id="a1",
        tool_use_id="call_x",
        status="completed",
    )
    # 写入侧:queue-op 后 store._last_uuid 不变(append_queue_operation 不推进)
    assert store._last_uuid == mainline_tip_uuid  # noqa: SLF001
    store.close()

    # resume 侧:queue-op 行在文件末尾,但无 uuid → 不会成为 tip
    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await s2.load_mainline()
    s2.close()
    assert s2._last_uuid == mainline_tip_uuid  # noqa: SLF001 - queue-op 无 uuid 不夺 tip


async def test_subagent_store_always_marks_sidechain(tmp_path):
    """防御 ①:SubagentStore.append 每行恒 isSidechain=true(防 #33651 漏标)。

    子 agent 行每行强制 isSidechain=true + agentId(subagent_store.py:77-78),
    即便泄露进主 jsonl 也会被 _read_mainline_rows 跳过 → 不可能夺主线 tip。
    """
    from birdcode.session.subagent_store import SubagentStore

    sa = SubagentStore(_ctx(), tmp_path, "a1", root=tmp_path)
    await sa.append(Message(role="user", content=[TextBlock(text="子 agent 任务")]))
    await sa.append(
        Message(role="assistant", content=[TextBlock(text="子 agent 答")]),
        is_assistant=True,
    )
    sa.close()

    rows = [
        json.loads(ln)
        for ln in sa.jsonl_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 2
    assert all(r["isSidechain"] is True for r in rows), "每行必须标 isSidechain=true"
    assert all(r["agentId"] == "a1" for r in rows)
