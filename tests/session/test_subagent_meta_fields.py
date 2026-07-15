from birdcode.session.models import SubagentMeta


def test_meta_has_prompt_isolation_base_sha():
    m = SubagentMeta.model_validate({
        "agentId": "sub-abc", "agentType": "general-purpose", "toolUseId": "(async-agent)",
        "isAsync": True, "status": "running", "spawnedAt": "2026-07-15T00:00:00Z",
        "prompt": "分析鉴权模块", "isolation": "worktree", "baseSha": "deadbeef",
    })
    assert m.prompt == "分析鉴权模块"
    assert m.isolation == "worktree"
    assert m.base_sha == "deadbeef"


def test_meta_fields_default_backward_compat():
    m = SubagentMeta.model_validate({
        "agentId": "sub-old", "toolUseId": "(sync-agent)", "status": "completed",
        "spawnedAt": "2026-07-15T00:00:00Z",
    })
    assert m.prompt == ""           # 旧文件无此字段,默认空
    assert m.isolation is None
    assert m.base_sha is None
