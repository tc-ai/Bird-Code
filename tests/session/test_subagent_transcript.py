# tests/session/test_subagent_transcript.py
"""S7:read_subagent_transcript 读 sidechain jsonl → list[Turn](/agents UI 看 transcript 用)。"""
from birdcode.blocks import TextBlock
from birdcode.conversation import Message
from birdcode.session.models import SessionContext
from birdcode.session.subagent_store import SubagentStore, read_subagent_transcript


async def test_read_subagent_transcript_roundtrip(tmp_path):
    """写 user+assistant sidechain 行 → 读回 list[Turn],含双方文本。"""
    ctx = SessionContext(session_id="s1", cwd=".", version="", git_branch=None)
    store = SubagentStore(ctx, tmp_path, "sub-abc", root=tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="hi")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="ok")]), is_assistant=True
    )
    store.close()

    turns = read_subagent_transcript(store.jsonl_path)
    assert turns
    texts = [
        getattr(b, "text", "")
        for t in turns
        for m in t.messages
        for b in m.content
    ]
    assert "hi" in texts and "ok" in texts


def test_read_subagent_transcript_missing_file(tmp_path):
    """缺文件 → 空 list(韧性,不抛)。"""
    assert read_subagent_transcript(tmp_path / "nope.jsonl") == []
