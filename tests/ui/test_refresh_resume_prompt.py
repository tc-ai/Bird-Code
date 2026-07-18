# tests/ui/test_refresh_resume_prompt.py
"""Task 3: _refresh_resume_prompt 统一入口 — 差集 mount + force_show 全量。

测点:
- 非 force:仅当 R−_shown 非空才 mount 新增批;已展示的不重弹。
- force_show:全量 R 再展示(即便全已 _shown)。
- 新增 delta:第二次扫到新 sub-agent → 只弹新增那一批。
"""

from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

from birdcode.conversation import TurnController
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.paths import subagents_dir
from birdcode.session.subagent_meta import write_subagent_meta
from birdcode.ui.app import BirdApp


def _meta(aid: str, status: str = "cancelled") -> SubagentMeta:
    return SubagentMeta(
        agentId=aid,
        agentType="general-purpose",
        toolUseId="tu-" + aid,
        status=status,
        spawnedAt="2026-07-18T00:00:00Z",
        prompt="t " + aid,
    )


def _seed(tmp_path: Path, aids: list[tuple[str, str]]) -> None:
    d = subagents_dir(tmp_path, "s1", tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    for aid, st in aids:
        write_subagent_meta(d / f"agent-{aid}.meta.json", _meta(aid, st))


def _fake_app(tmp_path: Path) -> SimpleNamespace:
    controller = TurnController.__new__(TurnController)
    controller.history = []  # type: ignore[attr-defined]
    store = SimpleNamespace(
        root=tmp_path,
        ctx=SessionContext(
            session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None,
        ),
        project_root=tmp_path,
        worktree_name=None,
    )
    ns = SimpleNamespace(
        _store=store,
        _controller=controller,
        _reminder_turn=None,
        _dismissal_turn=None,
        _shown_agent_ids=set(),
        _lost_agent_ids=set(),
        _mounted=None,
        query=lambda *a, **k: [],
        query_one=None,
    )
    # _refresh_resume_prompt 调 self._current_subagents_dir() + _rewrite_reminder();
    # 两者只读/写 self._store|_controller|_reminder_turn(fake 已备),绑真实方法即可。
    ns._current_subagents_dir = types.MethodType(
        BirdApp._current_subagents_dir, ns
    )
    ns._rewrite_reminder = types.MethodType(BirdApp._rewrite_reminder, ns)
    return ns


async def test_refresh_mounts_only_new(tmp_path: Path) -> None:
    _seed(tmp_path, [("sub-1", "cancelled"), ("sub-2", "cancelled")])
    fake = _fake_app(tmp_path)
    mounted: list = []

    async def _capture_mount(metas):
        mounted.append([m.agent_id for m in metas])

    # 直接挂到 fake 实例上(_refresh_resume_prompt 经 self._mount_resume_prompt 查找):
    # fake 是 SimpleNamespace,monkeypatch BirdApp 类属性不会被 SimpleNamespace 查到。
    fake._mount_resume_prompt = _capture_mount
    await BirdApp._refresh_resume_prompt(fake)  # type: ignore[arg-type]
    assert mounted == [["sub-1", "sub-2"]]
    # 再 refresh:都已 _shown → 不再 mount(无新)
    await BirdApp._refresh_resume_prompt(fake)  # type: ignore[arg-type]
    assert len(mounted) == 1


async def test_refresh_force_show_mounts_all(tmp_path: Path) -> None:
    _seed(tmp_path, [("sub-1", "cancelled")])
    fake = _fake_app(tmp_path)
    fake._shown_agent_ids = {"sub-1"}  # 已展示
    mounted: list = []

    async def _cap(metas):
        mounted.append([m.agent_id for m in metas])

    fake._mount_resume_prompt = _cap
    # 非 force:已展示 → 不弹
    await BirdApp._refresh_resume_prompt(fake)  # type: ignore[arg-type]
    assert mounted == []
    # force_show:全量 R 再展示
    await BirdApp._refresh_resume_prompt(fake, force_show=True)  # type: ignore[arg-type]
    assert mounted == [["sub-1"]]


async def test_refresh_new_delta_only(tmp_path: Path) -> None:
    _seed(tmp_path, [("sub-1", "cancelled")])
    fake = _fake_app(tmp_path)
    mounted: list = []

    async def _cap(metas):
        mounted.append([m.agent_id for m in metas])

    fake._mount_resume_prompt = _cap
    await BirdApp._refresh_resume_prompt(fake)  # type: ignore[arg-type]
    assert mounted == [["sub-1"]]
    # 新增 sub-2
    _seed(tmp_path, [("sub-2", "cancelled")])
    await BirdApp._refresh_resume_prompt(fake)  # type: ignore[arg-type]
    assert mounted[-1] == ["sub-2"]  # 只弹新增
