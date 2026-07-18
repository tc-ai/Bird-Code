from pathlib import Path

import pytest

from birdcode.agents.resume import ResumeDeps, resume_subagent
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.paths import subagents_dir
from birdcode.session.subagent_meta import write_subagent_meta


async def test_resume_refuses_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SessionContext(
        session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None,
    )
    sdir = subagents_dir(tmp_path, "s1", tmp_path)
    sdir.mkdir(parents=True)
    aid = "sub-lost"
    write_subagent_meta(
        sdir / f"agent-{aid}.meta.json",
        SubagentMeta(
            agentId=aid, agentType="general-purpose", toolUseId="tu",
            status="lost", spawnedAt="2026-07-18T00:00:00Z", prompt="x",
        ),
    )
    deps = ResumeDeps(
        manager=None,  # type: ignore[arg-type]
        root=tmp_path, session_id="s1", project_root=tmp_path,
        worktree_name=None, agent_registry=None,  # type: ignore[arg-type]
        parent_provider=None,  # type: ignore[arg-type]
        parent_registry=None, parent_gate=None, cfg=None,  # type: ignore[arg-type]
        app=None, ctx=ctx,
    )
    result = await resume_subagent(agent_id=aid, direction="继续", deps=deps)
    assert result.outcome == "sync_done"
    assert "丢弃" in result.text
