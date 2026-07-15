from pathlib import Path

from birdcode.agents.discover import RESUMABLE_STATUSES, find_resumable_subagents


def _write_meta(d: Path, agent_id: str, status: str, prompt: str = "x") -> None:
    import json

    d.mkdir(parents=True, exist_ok=True)
    (d / f"agent-{agent_id}.meta.json").write_text(
        json.dumps(
            {
                "agentId": agent_id,
                "agentType": "general-purpose",
                "toolUseId": "(async-agent)",
                "isAsync": True,
                "status": status,
                "spawnedAt": "2026-07-15T00:00:00Z",
                "prompt": prompt,
            }
        ),
        encoding="utf-8",
    )


def test_resumable_includes_nonterminal_and_cancelled(tmp_path: Path):
    d = tmp_path / "subagents"
    _write_meta(d, "running-1", "running", "分析鉴权")
    _write_meta(d, "idle-1", "idle", "等消息")
    _write_meta(d, "launched-1", "launched")
    _write_meta(d, "cancel-1", "cancelled", "被 ESC 中断")
    _write_meta(d, "done-1", "completed", "已完成")
    _write_meta(d, "err-1", "error")
    res = find_resumable_subagents(d)
    ids = {m.agent_id for m in res}
    assert ids == {"running-1", "idle-1", "launched-1", "cancel-1"}
    assert RESUMABLE_STATUSES == {"launched", "running", "idle", "cancelled"}


def test_resumable_empty_when_none(tmp_path: Path):
    assert find_resumable_subagents(tmp_path / "nope") == []
