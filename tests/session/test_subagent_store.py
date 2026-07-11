import json

from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message
from birdcode.session.models import SessionContext
from birdcode.session.subagent_store import SubagentStore


def _ctx(sid="s1"):
    return SessionContext(session_id=sid, cwd="C:/proj", version="0.1.0", git_branch="main")


def _subagent_jsonl(tmp_path, sid="s1", agent_id="a0685194cbfe9829e"):
    from birdcode.session import paths

    return paths.subagent_jsonl_path(tmp_path, sid, tmp_path, agent_id)


async def test_first_line_parent_null_and_sidechain_flags(tmp_path):
    """首行 user 任务指令:parentUuid=null、isSidechain=true、agentId 带上。"""
    sa = SubagentStore(_ctx(), tmp_path, "a0685194cbfe9829e", root=tmp_path)
    await sa.append(Message(role="user", content=[TextBlock(text="去 review T7")]))
    sa.close()
    rows = [
        json.loads(ln)
        for ln in _subagent_jsonl(tmp_path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["parentUuid"] is None
    assert rows[0]["isSidechain"] is True
    assert rows[0]["agentId"] == "a0685194cbfe9829e"
    assert rows[0]["message"]["content"][0]["text"] == "去 review T7"


async def test_chains_parent_and_flags_every_line(tmp_path):
    """后续行(parentUuid 链 + 每行 isSidechain/agentId)。"""
    sa = SubagentStore(_ctx(), tmp_path, "a1", root=tmp_path)
    await sa.append(Message(role="user", content=[TextBlock(text="task")]))
    await sa.append(
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="read_file", input={})]),
        is_assistant=True,
    )
    await sa.append(Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="ok")]))
    sa.close()
    rows = [
        json.loads(ln)
        for ln in _subagent_jsonl(tmp_path, agent_id="a1").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 3
    assert all(r["isSidechain"] is True for r in rows)
    assert all(r["agentId"] == "a1" for r in rows)
    assert rows[1]["parentUuid"] == rows[0]["uuid"]
    assert rows[2]["parentUuid"] == rows[1]["uuid"]


async def test_does_not_touch_main_meta_sidecar(tmp_path):
    """子 agent 写入不更新主 <sid>.meta(sidecar 是主会话专属)。"""
    sa = SubagentStore(_ctx(), tmp_path, "a1", root=tmp_path)
    await sa.append(Message(role="user", content=[TextBlock(text="task")]))
    sa.close()
    from birdcode.session import paths

    main_meta = paths.session_jsonl(tmp_path, "s1", tmp_path).with_suffix(".meta")
    assert not main_meta.exists()


async def test_append_io_failure_does_not_raise(tmp_path):
    """IO 失败只 log,不杀主循环(与 SessionStore 一致)。"""
    sa = SubagentStore(_ctx(), tmp_path, "a1", root=tmp_path)
    sa._fh.close()  # noqa: SLF001
    await sa.append(Message(role="user", content=[TextBlock(text="x")]))  # 不抛
    sa.close()
