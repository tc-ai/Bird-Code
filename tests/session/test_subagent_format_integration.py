"""端到端验证 spec 定义的 subagent jsonl 消息体格式(纯持久化层,不起真子 agent)。

覆盖:同步结果行 / 异步(回执+queue-op+注入) / pending 识别 / 子 agent 侧链日志 /
子 agent meta.json / 主线 load 不受侧链影响。
"""

import json

from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message
from birdcode.session import codec, paths
from birdcode.session.models import AgentToolUseResult, SessionContext, SubagentMeta
from birdcode.session.store import SessionStore
from birdcode.session.subagent_meta import list_subagent_metas, write_subagent_meta
from birdcode.session.subagent_store import SubagentStore


def _ctx(sid="s1"):
    return SessionContext(session_id=sid, cwd="C:/proj", version="0.1.0", git_branch="main")


def _main_rows(tmp_path, sid="s1"):
    jf = paths.session_jsonl(tmp_path, sid, tmp_path)
    return [json.loads(ln) for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip()]


async def test_sync_subagent_result_format(tmp_path):
    """同步:派生 tool_use → 结果 user 行(tool_result + toolUseResult + sourceToolAssistantUUID)。"""
    store = SessionStore(_ctx(), tmp_path, root=tmp_path)
    spawn_uuid = await store.append(
        Message(
            role="assistant",
            content=[
                ToolUseBlock(id="call_x", name="Agent", input={"description": "T7", "prompt": "go"})
            ],
        ),
        is_assistant=True,
    )
    tur = AgentToolUseResult(
        agent_id="a1",
        agent_type="general-purpose",
        status="completed",
        is_async=False,
        content="done",
        total_tokens=10,
    ).model_dump(by_alias=True, exclude_none=True)
    await store.append(
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_x", content="done")]),
        tool_use_results={"call_x": tur},
        source_tool_assistant_uuid=spawn_uuid,
    )
    store.close()
    rows = _main_rows(tmp_path)
    result = [r for r in rows if r.get("toolUseResult")][0]
    assert result["sourceToolAssistantUUID"] == spawn_uuid
    assert result["toolUseResult"]["call_x"]["agentId"] == "a1"
    assert result["toolUseResult"]["call_x"]["status"] == "completed"
    assert result["isSidechain"] is False  # 主线行非侧链


async def test_async_subagent_lifecycle_and_pending(tmp_path):
    """异步:派生 → 回执 → queue-op(enqueue) → 注入行;pending 识别正确。"""
    store = SessionStore(_ctx(), tmp_path, root=tmp_path)
    await store.append(
        Message(
            role="assistant",
            content=[
                ToolUseBlock(
                    id="call_y",
                    name="Agent",
                    input={"prompt": "go"},
                )
            ],
        ),
        is_assistant=True,
    )
    ack = AgentToolUseResult(
        agent_id="a2",
        status="launched",
        is_async=True,
        output_file="/tmp/a2.txt",
        can_read_output_file=True,
    ).model_dump(by_alias=True, exclude_none=True)
    await store.append(
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call_y",
                    content="Async agent launched successfully.\nAgent ID: a2",
                )
            ],
        ),
        tool_use_results={"call_y": ack},
    )
    # 子 agent 完成 → queue-op
    await store.append_queue_operation(
        operation="enqueue",
        agent_id="a2",
        tool_use_id="call_y",
        status="completed",
    )

    # 此时 a2 的 enqueue 尚未注入 → pending
    rows = _main_rows(tmp_path)
    pending = codec.find_pending_notifications(rows)
    assert len(pending) == 1 and pending[0]["agentId"] == "a2"

    # 注入后 → 不再 pending
    await store.append(
        Message(
            role="user",
            content=[
                TextBlock(text="<task-notification><task-id>a2</task-id></task-notification>")
            ],
        ),
        is_task_notification=True,
        agent_id="a2",
    )
    store.close()
    assert codec.find_pending_notifications(_main_rows(tmp_path)) == []


async def test_subagent_sidechain_log_and_meta(tmp_path):
    """子 agent 侧链日志(首行 parentUuid=null、全 isSidechain+agentId)+ meta.json。"""
    sa = SubagentStore(_ctx(), tmp_path, "a3", root=tmp_path)
    await sa.append(Message(role="user", content=[TextBlock(text="task")]))
    await sa.append(
        Message(role="assistant", content=[TextBlock(text="working")]), is_assistant=True
    )
    sa.close()
    sa_rows = [
        json.loads(ln)
        for ln in paths.subagent_jsonl_path(tmp_path, "s1", tmp_path, "a3")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert sa_rows[0]["parentUuid"] is None
    assert all(r["isSidechain"] is True and r["agentId"] == "a3" for r in sa_rows)

    # meta.json
    meta_path = paths.subagent_meta_path(tmp_path, "s1", tmp_path, "a3")
    write_subagent_meta(
        meta_path,
        SubagentMeta(agent_id="a3", tool_use_id="call_z", status="completed"),
    )
    metas = list_subagent_metas(paths.subagents_dir(tmp_path, "s1", tmp_path))
    assert len(metas) == 1 and metas[0].agent_id == "a3"


async def test_mainline_load_unaffected_by_sidechain(tmp_path):
    """主会话 load_mainline 不读子 agent 文件;侧链行(若误入主 jsonl)被跳过。"""
    store = SessionStore(_ctx(), tmp_path, root=tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="主提问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="主答")]), is_assistant=True
    )
    store.close()
    # 子 agent 文件独立存在
    sa = SubagentStore(_ctx(), tmp_path, "a4", root=tmp_path)
    await sa.append(Message(role="user", content=[TextBlock(text="子任务")]))
    sa.close()

    store2 = SessionStore(_ctx(), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()
    store2.close()
    assert len(turns) == 1
    texts = [b.text for t in turns for m in t.messages for b in m.content if hasattr(b, "text")]
    assert "主提问" in texts and "子任务" not in texts  # 子 agent 行不进主线
