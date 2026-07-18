from pathlib import Path
from types import MethodType, SimpleNamespace

from birdcode.conversation import TurnController
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.paths import subagents_dir
from birdcode.session.subagent_meta import read_subagent_meta, write_subagent_meta
from birdcode.ui.app import BirdApp


def _meta(aid: str, status: str = "cancelled") -> SubagentMeta:
    return SubagentMeta(
        agentId=aid, agentType="general-purpose", toolUseId="tu-" + aid,
        status=status, spawnedAt="2026-07-18T00:00:00Z", prompt="t " + aid,
    )


def _fake_app(tmp_path: Path) -> SimpleNamespace:
    controller = TurnController.__new__(TurnController)
    controller.history = []  # type: ignore[attr-defined]
    store = SimpleNamespace(
        root=tmp_path, ctx=SessionContext(
            session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None,
        ),
        project_root=tmp_path, worktree_name=None,
    )
    fake = SimpleNamespace(
        _store=store, _controller=controller, _reminder_turn=None,
        _dismissal_turn=None, _shown_agent_ids=set(), _lost_agent_ids=set(),
    )
    # _on_resume_dismissed 编排 4 个实例方法;unbound 调用下须把真实方法绑到 fake
    # (既有 helper 测试单方法只访问数据属性故无需;本测试跨方法编排,必须绑定)。
    fake._mark_lost_agent = MethodType(BirdApp._mark_lost_agent, fake)
    fake._current_subagents_dir = MethodType(BirdApp._current_subagents_dir, fake)
    fake._rewrite_reminder = MethodType(BirdApp._rewrite_reminder, fake)
    fake._rewrite_dismissal = MethodType(BirdApp._rewrite_dismissal, fake)
    # _on_resume_dismissed 末尾回焦主输入:fake 无 #input,真实方法内 try/except 跳过即可。
    fake._refocus_main_input = MethodType(BirdApp._refocus_main_input, fake)
    return fake


async def test_dismissed_marks_lost_and_rewrites(tmp_path: Path) -> None:
    sdir = subagents_dir(tmp_path, "s1", tmp_path)
    sdir.mkdir(parents=True)
    write_subagent_meta(sdir / "agent-sub-1.meta.json", _meta("sub-1"))
    write_subagent_meta(sdir / "agent-sub-2.meta.json", _meta("sub-2"))
    fake = _fake_app(tmp_path)
    # 先注入 reminder(含 sub-1,sub-2)
    await BirdApp._rewrite_reminder(fake, [_meta("sub-1"), _meta("sub-2")])  # type: ignore[arg-type]
    assert fake._reminder_turn is not None
    # 丢弃 sub-1(整批=[sub-1])
    await BirdApp._on_resume_dismissed(fake, ["sub-1"])  # type: ignore[arg-type]
    # sub-1 meta=lost
    m1 = read_subagent_meta(sdir / "agent-sub-1.meta.json")
    assert m1 is not None and m1.status == "lost"
    # sub-2 仍可续(未丢)
    m2 = read_subagent_meta(sdir / "agent-sub-2.meta.json")
    assert m2 is not None and m2.status == "cancelled"
    # reminder 重写:含 sub-2,不含 sub-1
    assert fake._reminder_turn is not None
    rtext = fake._reminder_turn.messages[0].content[0].text  # type: ignore[attr-defined]
    assert "sub-2" in rtext and "sub-1" not in rtext
    # 反信号出现,含 sub-1
    assert fake._dismissal_turn is not None
    dtext = fake._dismissal_turn.messages[0].content[0].text  # type: ignore[attr-defined]
    assert "sub-1" in dtext and "resume_agent" in dtext
    assert "sub-1" in fake._lost_agent_ids  # type: ignore[attr-defined]
