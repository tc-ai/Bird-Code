# tests/agents/test_resume_corrupt_meta.py
"""Task 16:meta 损坏/缺失处理 —— 回归锁定两条不变量。

覆盖:
- resume_subagent:meta 文件不存在 → 友好 ResumeResult(sync_done + 含「缺失」/「损坏」),不抛
- find_resumable_subagents:坏 meta 文件(非合法 JSON)被 list_subagent_metas 跳过,
  不进结果、不抛

不依赖 test_resume_subagent.py 的私有桩:本文件自包含最小 ResumeDeps(manager 在 None 分支
不会被调方法,故用最简 object 即可)。meta 缺失走真实 subagent_meta_path + read_subagent_meta。
"""
from __future__ import annotations

import json
from pathlib import Path

from birdcode.agents.discover import find_resumable_subagents
from birdcode.agents.resume import ResumeDeps, resume_subagent


def _deps(tmp_path: Path) -> ResumeDeps:
    """最小 ResumeDeps:meta None 分支在 manager.has_live 之前返回,manager 无需真实方法。

    root=project_root=tmp_path,worktree_name=None → subagent_meta_path 落
    tmp_path/.birdcode/sessions/s1/subagents/agent-<id>.meta.json(不存在 → read 返 None)。
    """
    return ResumeDeps(
        manager=object(),  # type: ignore[arg-type]  # None 分支不调 has_live
        root=tmp_path,
        session_id="s1",
        project_root=tmp_path,
        worktree_name=None,
        agent_registry=object(),
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        cfg=None,
        app=None,
        ctx=None,
        spawn_depth=1,
        progress_cb=None,
    )


async def test_resume_with_missing_meta_returns_friendly(tmp_path: Path):
    # 不写 meta 文件 → read_subagent_meta 返 None → resume_subagent 友好返回,不抛
    result = await resume_subagent(
        agent_id="sub-gone", direction="继续", deps=_deps(tmp_path),
    )

    assert result.outcome == "sync_done"
    # 文案含 agent_id + 「缺失」或「损坏」,对用户可读
    assert "sub-gone" in result.text
    assert "缺失" in result.text or "损坏" in result.text


def test_corrupt_meta_skipped_in_discover(tmp_path: Path):
    # 坏 meta(非合法 JSON)与好 meta 共存:list_subagent_metas 跳坏,find_resumable 只返好
    d = tmp_path / "subagents"
    d.mkdir()
    (d / "agent-sub-bad.meta.json").write_text("{不是合法 json", encoding="utf-8")
    (d / "agent-sub-good.meta.json").write_text(
        json.dumps(
            {
                "agentId": "sub-good",
                "agentType": "general-purpose",
                "toolUseId": "(async-agent)",
                "isAsync": True,
                "status": "running",  # 可续跑 status
                "spawnedAt": "2026-07-15T00:00:00Z",
                "prompt": "分析鉴权",
            }
        ),
        encoding="utf-8",
    )

    res = find_resumable_subagents(d)

    # 坏的被跳,只返好 meta(顺序无关,用集合断言 agent_id)
    assert {m.agent_id for m in res} == {"sub-good"}
