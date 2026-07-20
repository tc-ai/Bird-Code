from birdcode.session.models import SystemLine


def test_system_compact_boundary_line():
    line = SystemLine.model_validate(
        {
            "uuid": "u1",
            "parent_uuid": None,  # 边界行:树在此重启
            "session_id": "s1",
            "timestamp": "2026-07-06T00:00:00.000Z",
            "type": "system",
            "subtype": "compact_boundary",
            "logical_parent_uuid": "prev-tail-uuid",  # 逻辑链续链
            "content": "Conversation compacted",
            "compact_metadata": {"trigger": "manual", "preTokens": 167000, "postTokens": 48000},
        }
    )
    assert line.type == "system"
    assert line.subtype == "compact_boundary"
    assert line.parent_uuid is None
    assert line.logical_parent_uuid == "prev-tail-uuid"
    assert line.compact_metadata["trigger"] == "manual"
    # by_alias dump:parentUuid / logicalParentUuid / compactMetadata
    dumped = line.model_dump(by_alias=True, exclude_none=False)
    assert dumped["parentUuid"] is None
    assert dumped["logicalParentUuid"] == "prev-tail-uuid"
    assert dumped["compactMetadata"]["trigger"] == "manual"
