from pathlib import Path
from types import SimpleNamespace

from birdcode.conversation import TurnController
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.paths import subagents_dir
from birdcode.session.subagent_meta import read_subagent_meta, write_subagent_meta
from birdcode.ui.app import BirdApp


def _ctx(tmp_path: Path) -> SessionContext:
    return SessionContext(
        session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None,
    )


def _store_ns(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        root=tmp_path, ctx=_ctx(tmp_path), project_root=tmp_path, worktree_name=None,
    )


def _meta(aid: str, status: str = "cancelled") -> SubagentMeta:
    return SubagentMeta(
        agentId=aid, agentType="general-purpose", toolUseId="tu-" + aid,
        status=status, spawnedAt="2026-07-18T00:00:00Z", prompt="任务 " + aid,
    )


async def test_mark_lost_agent_writes_status(tmp_path: Path) -> None:
    sdir = subagents_dir(tmp_path, "s1", tmp_path)
    sdir.mkdir(parents=True)
    write_subagent_meta(sdir / "agent-sub-1.meta.json", _meta("sub-1"))
    fake = SimpleNamespace(_store=_store_ns(tmp_path))
    BirdApp._mark_lost_agent(fake, "sub-1")  # type: ignore[arg-type]
    m = read_subagent_meta(sdir / "agent-sub-1.meta.json")
    assert m is not None and m.status == "lost"


async def test_rewrite_reminder_single_dynamic(tmp_path: Path) -> None:
    controller = TurnController.__new__(TurnController)
    controller.history = []  # type: ignore[attr-defined]
    fake = SimpleNamespace(_controller=controller, _reminder_turn=None)
    metas = [_meta("sub-1"), _meta("sub-2")]
    await BirdApp._rewrite_reminder(fake, metas)  # type: ignore[arg-type]
    assert fake._reminder_turn is not None
    assert controller.history == [fake._reminder_turn]
    # 第二次:撤旧重注(单条,不堆积)
    await BirdApp._rewrite_reminder(fake, [_meta("sub-3")])  # type: ignore[arg-type]
    assert len(controller.history) == 1
    assert controller.history[0] is fake._reminder_turn
    assert "sub-3" in controller.history[0].messages[0].content[0].text  # type: ignore[attr-defined]


async def test_rewrite_reminder_empty_withdraws(tmp_path: Path) -> None:
    controller = TurnController.__new__(TurnController)
    controller.history = []  # type: ignore[attr-defined]
    fake = SimpleNamespace(_controller=controller, _reminder_turn=None)
    await BirdApp._rewrite_reminder(fake, [_meta("sub-1")])  # type: ignore[arg-type]
    assert controller.history and fake._reminder_turn is not None
    # 空 metas → 撤回旧 Turn,不重注
    await BirdApp._rewrite_reminder(fake, [])  # type: ignore[arg-type]
    assert controller.history == []
    assert fake._reminder_turn is None


async def test_rewrite_dismissal_lists_lost(tmp_path: Path) -> None:
    controller = TurnController.__new__(TurnController)
    controller.history = []  # type: ignore[attr-defined]
    fake = SimpleNamespace(
        _controller=controller, _dismissal_turn=None,
        _lost_agent_ids={"sub-1", "sub-2"},
    )
    await BirdApp._rewrite_dismissal(fake)  # type: ignore[arg-type]
    assert fake._dismissal_turn is not None
    text = fake._dismissal_turn.messages[0].content[0].text  # type: ignore[attr-defined]
    assert "agent_id=sub-1" in text and "agent_id=sub-2" in text
    assert "resume_agent" in text  # 反信号点名工具
