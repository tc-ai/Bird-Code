from pathlib import Path

from birdcode.agents.discover import find_resumable_subagents
from birdcode.session.models import SubagentMeta
from birdcode.session.subagent_meta import write_subagent_meta


def _write_meta(d: Path, aid: str, status: str) -> None:
    m = SubagentMeta(
        agentId=aid, agentType="general-purpose", toolUseId="tu-" + aid,
        status=status, spawnedAt="2026-07-18T00:00:00Z", prompt="任务 " + aid,
    )
    write_subagent_meta(d / f"agent-{aid}.meta.json", m)


def test_lost_not_resumable(tmp_path: Path) -> None:
    d = tmp_path / "subagents"
    d.mkdir()
    _write_meta(d, "aaa", "cancelled")  # 可续
    _write_meta(d, "bbb", "lost")       # 已丢弃,不应返回
    resumable = find_resumable_subagents(d)
    ids = {m.agent_id for m in resumable}
    assert "aaa" in ids
    assert "bbb" not in ids
