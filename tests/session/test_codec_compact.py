# tests/session/test_codec_compact.py
"""Task 4: compaction 边界行编解码 + resume 边界截断(split_live_segment)。

边界 system 行(parentUuid=null + logicalParentUuid 续链)由 encode_system_compact 落盘;
resume 时 split_live_segment 找最后一条 compact_boundary 起(含)→ 末尾的段,
更早的边界与被替代的 bulk 全丢;decode_lines 不把 system 行当对话 Turn。
"""

from birdcode.session import codec
from birdcode.session.models import SessionContext

CTX = SessionContext(session_id="s1", cwd=".", version="1", git_branch=None)


def _u(text: str, uuid: str, parent: str | None) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": "2026-07-06T00:00:00.000Z",
        "isSidechain": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _boundary(uuid: str, logical_parent: str) -> dict:
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": uuid,
        "parentUuid": None,
        "logicalParentUuid": logical_parent,
        "sessionId": "s1",
        "timestamp": "2026-07-06T00:00:00.000Z",
        "isSidechain": False,
        "content": "Conversation compacted",
        "compactMetadata": {
            "trigger": "manual",
            "preTokens": 167000,
            "postTokens": 48000,
            "preservedMessages": {"uuids": []},
        },
    }


def _summary(uuid: str, parent: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": "2026-07-06T00:00:01.000Z",
        "isSidechain": False,
        "isCompactSummary": True,
        "message": {"role": "user", "content": [{"type": "text", "text": "SUMMARY..."}]},
    }


def test_encode_system_compact_boundary():
    """encode_system_compact 落 system 边界行。

    校验:parentUuid=null、logicalParentUuid 续链、compactMetadata 落盘。
    """
    line = codec.encode_system_compact(
        subtype="compact_boundary",
        uuid="b1",
        parent_uuid=None,
        ctx=CTX,
        timestamp="2026-07-06T00:00:00.000Z",
        logical_parent_uuid="prev",
        content="Conversation compacted",
        compact_metadata={"trigger": "manual", "preTokens": 167000, "postTokens": 48000},
    )
    assert line["type"] == "system"
    assert line["subtype"] == "compact_boundary"
    assert line["parentUuid"] is None
    assert line["logicalParentUuid"] == "prev"
    assert line["compactMetadata"]["preTokens"] == 167000


def test_split_at_last_boundary_no_boundary_returns_all():
    """无边界行 → 全部保留(现状兼容)。"""
    rows = [_u("a", "1", None), _u("b", "2", "1")]
    out = codec.split_live_segment(rows)
    # 允许返回原列表或等价切片
    assert out is rows or out == rows


def test_split_at_last_boundary_single():
    """一条边界:live = 边界行起(含)到末尾。"""
    rows = [
        _u("old1", "1", None),
        _u("old2", "2", "1"),
        _boundary("b1", "2"),
        _summary("s1", "b1"),
        _u("tail", "3", "s1"),
    ]
    live = codec.split_live_segment(rows)
    uuids = [r["uuid"] for r in live]
    assert uuids == ["b1", "s1", "3"]  # old1/old2 被丢


def test_split_at_last_boundary_multiple_keeps_only_last():
    """多次 compaction:只留最后一条边界起的段;更早的边界与 bulk 全丢。"""
    rows = [
        _u("old1", "1", None),
        _boundary("b1", "1"),
        _summary("s1", "b1"),
        _u("mid", "2", "s1"),
        _boundary("b2", "2"),
        _summary("s2", "b2"),
        _u("tail", "3", "s2"),
    ]
    live = codec.split_live_segment(rows)
    uuids = [r["uuid"] for r in live]
    assert uuids == ["b2", "s2", "3"]


def test_split_does_not_misidentify_non_null_parent_system_as_boundary():
    """system 行 parentUuid 非 null(非典型)不应被识别为截断边界。"""
    rows = [
        _u("old", "1", None),
        # parentUuid 非 null 的 system 行(理论上不该出现,但防御)
        {**_boundary("b1", "1"), "parentUuid": "1"},
        _u("tail", "2", "b1"),
    ]
    live = codec.split_live_segment(rows)
    # 无 parentUuid:null 的边界 → 全量返回
    assert [r["uuid"] for r in live] == ["1", "b1", "2"]


def test_decode_skips_system_boundary_metadata_only():
    """decode_lines 不把 system 边界行当对话 Turn(它是元信息);摘要 user 行正常进 Turn。"""
    rows = [_boundary("b1", "1"), _summary("s1", "b1"), _u("tail", "3", "s1")]
    turns = codec.decode_lines(rows)
    # 摘要 user 行(text block)开一个 Turn + tail user 行开一个 Turn;边界行不产 Turn
    texts = [b.text for t in turns for m in t.messages for b in m.content if hasattr(b, "text")]
    assert "SUMMARY..." in texts and "tail" in texts


def test_encode_system_compact_with_minimal_args_defaults():
    """无 compact_metadata / content 时使用默认值,仍能落合法 system 行。"""
    line = codec.encode_system_compact(
        subtype="compact_boundary",
        uuid="b1",
        parent_uuid=None,
        ctx=CTX,
        timestamp="2026-07-06T00:00:00.000Z",
    )
    assert line["type"] == "system"
    assert line["content"] == ""
    assert line["compactMetadata"] == {}
    assert line["logicalParentUuid"] is None


def test_encode_system_compact_preserves_ctx_fields():
    """encode_system_compact 仍写入 session 公共字段(cwd/version/git_branch)。"""
    line = codec.encode_system_compact(
        subtype="compact_boundary",
        uuid="b1",
        parent_uuid=None,
        ctx=CTX,
        timestamp="2026-07-06T00:00:00.000Z",
    )
    assert line["sessionId"] == "s1"
    assert line["cwd"] == "."
    assert line["version"] == "1"
    assert line["gitBranch"] is None
