from pathlib import Path
from types import MethodType, SimpleNamespace

from birdcode.conversation import Message, TurnController
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.paths import subagents_dir
from birdcode.session.subagent_meta import read_subagent_meta, write_subagent_meta
from birdcode.ui.app import BirdApp


def _meta(aid: str, status: str = "cancelled") -> SubagentMeta:
    return SubagentMeta(
        agentId=aid, agentType="general-purpose", toolUseId="tu-" + aid,
        status=status, spawnedAt="2026-07-18T00:00:00Z", prompt="t " + aid,
    )


def _fake_app(tmp_path: Path, persisted: list[Message]) -> SimpleNamespace:
    controller = TurnController.__new__(TurnController)
    controller.history = []  # type: ignore[attr-defined]
    store = SimpleNamespace(
        root=tmp_path, ctx=SessionContext(
            session_id="s1", cwd=str(tmp_path), version="0.1.0", git_branch=None,
        ),
        project_root=tmp_path, worktree_name=None,
    )

    async def _spy_append(msg: Message, **_kw: object) -> None:  # 桩 store.append:记录落盘消息
        persisted.append(msg)
    store.append = _spy_append  # type: ignore[method-assign]

    fake = SimpleNamespace(
        _store=store, _controller=controller, _reminder_turn=None,
        _shown_agent_ids=set(),
    )
    # _on_resume_dismissed 编排多个实例方法;unbound 调用下须绑到 fake。
    fake._mark_lost_agent = MethodType(BirdApp._mark_lost_agent, fake)
    fake._current_subagents_dir = MethodType(BirdApp._current_subagents_dir, fake)
    fake._rewrite_reminder = MethodType(BirdApp._rewrite_reminder, fake)
    fake._refocus_main_input = MethodType(BirdApp._refocus_main_input, fake)
    return fake


async def test_dismissed_marks_lost_and_records_user_message(tmp_path: Path) -> None:
    """2/No:meta 标 lost(硬) + reminder 重写 + 写【真实 user 消息】(落盘 + 进历史),不触发轮次。

    B 方案:store.append 落盘 jsonl + history.append 进历史,但【不 submit】→ 不跑 LLM。
    真实 user 消息(非 system-reminder):带 agent_id + 任务描述 + "勿重新发起",跨会话保留。
    """
    sdir = subagents_dir(tmp_path, "s1", tmp_path)
    sdir.mkdir(parents=True)
    write_subagent_meta(sdir / "agent-sub-1.meta.json", _meta("sub-1"))
    write_subagent_meta(sdir / "agent-sub-2.meta.json", _meta("sub-2"))
    persisted: list[Message] = []
    fake = _fake_app(tmp_path, persisted)
    # 先注入 reminder(含 sub-1,sub-2)
    await BirdApp._rewrite_reminder(fake, [_meta("sub-1"), _meta("sub-2")])  # type: ignore[arg-type]
    assert fake._reminder_turn is not None
    # 丢弃 sub-1(整批=[sub-1])
    await BirdApp._on_resume_dismissed(fake, ["sub-1"])  # type: ignore[arg-type]
    # sub-1 meta=lost(硬)
    m1 = read_subagent_meta(sdir / "agent-sub-1.meta.json")
    assert m1 is not None and m1.status == "lost"
    # sub-2 仍可续(未丢)
    m2 = read_subagent_meta(sdir / "agent-sub-2.meta.json")
    assert m2 is not None and m2.status == "cancelled"
    # reminder 重写:含 sub-2,不含 sub-1
    assert fake._reminder_turn is not None
    rtext = fake._reminder_turn.messages[0].content[0].text  # type: ignore[attr-defined]
    assert "sub-2" in rtext and "sub-1" not in rtext
    # 落盘:store.append 收到一条真实 user 消息(role=user,非 system-reminder)
    assert len(persisted) == 1
    msg = persisted[0]
    assert msg.role == "user"
    notice = msg.content[0].text  # type: ignore[union-attr]
    assert "sub-1" in notice and "t sub-1" in notice
    assert "无需" in notice and "重新发起" in notice and "resume_agent" in notice
    assert "<system-reminder>" not in notice
    # 进历史:controller.history 含此丢弃 Turn
    discard = [
        t for t in fake._controller.history  # type: ignore[union-attr]
        if "我已取消" in t.messages[0].content[0].text  # type: ignore[union-attr]
    ]
    assert len(discard) == 1
