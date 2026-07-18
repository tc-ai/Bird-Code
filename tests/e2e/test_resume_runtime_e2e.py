# tests/e2e/test_resume_runtime_e2e.py
"""e2e:ESC 中断子 agent → 即时弹窗 → 1.Yes 续跑 / 2.No 丢弃(整批 lost)。

参照 tests/e2e/test_continue_after_esc_e2e.py 的 MockProvider + 真实 store 模式,
但走真实 BirdApp(run_test 触发 on_mount/事件回路)+ 真实 SessionStore +
真实 _refresh_resume_prompt / _on_interrupt_scan / _on_resume_confirmed /
_on_resume_dismissed。子 agent meta 直接 seed(status=cancelled,模拟"被中断态"),
而非真实派生子 agent —— 覆盖核心断言(弹窗触发、1/2 决策、lost 持久化、重启不弹),
对应 plan Task 8 Step 3 的"真实步骤"范围(半 e2e,详见用例注释)。

4 个用例:
  1. test_esc_interrupt_shows_prompt_with_new_agent — ESC → 即时弹窗含新中断 agent
  2. test_esc_interrupt_plain_tool_no_prompt — ESC 普通工具(无 meta)→ 不弹
  3. test_press_2_no_marks_lost_and_persists — 按 2/No → 整批 lost + reminder 重写 +
     反信号;再 _refresh(force_show=True) 不弹(discover 排除 lost)
  4. test_press_1_yes_submits_continue — 按 1/Yes → controller.submit("继续")

关键时序约束(来自 Task 7 实测):
  - action_interrupt() 后 _on_interrupt_scan 是 bg task + 内部 sleep(0.05),
    故先 pilot.pause 让 Interrupted 事件派发,再 drain bg tasks,最后断言。
  - 立即断言 app.query("ResumePrompt") 会竞态返回空。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.paths import subagent_meta_path
from birdcode.session.store import SessionStore
from birdcode.session.subagent_meta import read_subagent_meta, write_subagent_meta
from birdcode.ui.app import BirdApp
from birdcode.ui.widgets.input_area import InputArea

# ---- 辅助 ----


def _make_store(tmp_path: Path, session_id: str) -> SessionStore:
    """真实 SessionStore(主 jsonl 落 tmp,不污染 ~/.birdcode)。"""
    ctx = SessionContext(
        session_id=session_id, cwd=str(tmp_path), version="0.1.0", git_branch=None,
    )
    return SessionStore(ctx, tmp_path, root=tmp_path)


def _meta(aid: str, status: str = "cancelled") -> SubagentMeta:
    """标准测试 meta:cancelled ∈ RESUMABLE_STATUSES → discover/refresh 真实路径发现。"""
    return SubagentMeta(
        agentId=aid,
        agentType="general-purpose",
        toolUseId="tu-" + aid,
        status=status,
        spawnedAt="2026-07-18T00:00:00Z",
        prompt="任务 " + aid,
    )


def _seed_meta(store: SessionStore, aid: str, status: str = "cancelled") -> Path:
    """把 meta 写到 store 的 subagents 目录(与 app._current_subagents_dir 同路径)。"""
    p = subagent_meta_path(
        store.root,
        store.ctx.session_id,
        store.project_root,
        aid,
        worktree_name=store.worktree_name,
    )
    write_subagent_meta(p, _meta(aid, status))
    return p


async def _drain_bg_tasks(app: BirdApp) -> None:
    """等 app._bg_tasks 内所有后台 task 跑完(_on_interrupt_scan / _on_resume_* 等)。

    ResumePrompt.action_confirm/action_ignore 把 _on_resume_confirmed/_dismissed
    作 create_task 加入 app._bg_tasks;_on_activity(Interrupted) 把 _on_interrupt_scan
    作 create_task 加入 app._bg_tasks。drain 后所有副作用(标 lost / mount / submit)落定。
    """
    for t in list(app._bg_tasks):
        await t


# ---- 用例 1:ESC 中断子 agent → 即时弹窗含该 agent ----


async def test_esc_interrupt_shows_prompt_with_new_agent(tmp_path: Path) -> None:
    """ESC 中断 → _on_interrupt_scan 扫到新增 meta → mount ResumePrompt 含该 agent。

    半 e2e:真实 BirdApp + 真实 SessionStore + 真实 _on_interrupt_scan/_refresh,但
    "被中断的子 agent"由直接 seed meta(cancelled)模拟,非真实派生。覆盖核心断言:
    action_interrupt → Interrupted 事件 → bg task → _refresh → 差集 mount widget。
    """
    store = _make_store(tmp_path, "e2e-esc-new")
    app = BirdApp(MockProvider(delay=2.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        # 启动一轮(MockProvider delay=2.0s → phase 2 持久 busy,保证可 ESC 中断)
        await app._on_submitted(InputArea.Submitted("hello world"))
        await pilot.pause(delay=0.15)
        assert app.controller.busy is True  # 前置:确在 busy(否则 ESC 不触发 Interrupted)

        # ESC 前 seed sub-B meta(_shown 仍空 → sub-B 即"新增差集")。
        _seed_meta(store, "sub-B", status="cancelled")

        # ESC:pilot.press → BINDINGS.escape → action_interrupt → controller.interrupt
        # → drain 抛 CancelledError → 发 Interrupted 事件 → _on_activity(Interrupted)
        # → create_task(_on_interrupt_scan) 加入 _bg_tasks。
        await pilot.press("escape")
        # _on_interrupt_scan 是 bg task 且内部 sleep(0.05);pilot.pause 让 Interrupted
        # 事件派发到位 + bg task 起来,drain_bg_tasks 再等它跑完。立即断言会竞态空。
        await pilot.pause(delay=0.2)
        await _drain_bg_tasks(app)

        prompts = list(app.query("ResumePrompt"))
        assert len(prompts) == 1, "ESC 中断子 agent 后应即 mount ResumePrompt"
        assert "sub-B" in prompts[0]._agent_ids  # noqa: SLF001
        # busy 已收尾(Interrupted 终态)
        assert app.controller.busy is False


# ---- 用例 2:ESC 中断普通工具 → 不弹窗 ----


async def test_esc_interrupt_plain_tool_no_prompt(tmp_path: Path) -> None:
    """ESC 中断普通工具(bash 跑一半,无子 agent meta)→ _refresh 差集空 → 不弹窗。

    关键:force_show=False 路径只 mount delta(_shown 之外的新增)。ESC 前后无新 meta
    → discover=空 → 不 mount。这是"只对子 agent 弹、不对工具中断弹"的端到保证。
    """
    store = _make_store(tmp_path, "e2e-esc-tool")
    app = BirdApp(MockProvider(delay=2.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("hello world"))
        await pilot.pause(delay=0.15)
        assert app.controller.busy is True

        # 不 seed 任何 meta(普通工具中断,无子 agent)
        await pilot.press("escape")
        await pilot.pause(delay=0.2)
        await _drain_bg_tasks(app)

        assert list(app.query("ResumePrompt")) == [], "普通工具中断不应弹 ResumePrompt"
        assert app.controller.busy is False


# ---- 用例 3:按 2/No → 整批 lost + reminder 重写 + 反信号;重启不再弹 ----


async def test_press_2_no_marks_lost_and_persists(tmp_path: Path) -> None:
    """按 2/No → 整批 meta=lost;reminder 撤回(无可续);反信号含整批;重启不再弹。

    "重启"= 再 _refresh(force_show=True):discover 排除 lost → metas=空 →
    force_show 分支 `if metas:` 不进 → 不 mount。这是"跨会话不再弹"的端到保证
    (meta 落盘 lost,新会话 discover 也不会扫到)。
    """
    store = _make_store(tmp_path, "e2e-no")
    sub_a_path = _seed_meta(store, "sub-A", status="cancelled")
    sub_b_path = _seed_meta(store, "sub-B", status="cancelled")

    app = BirdApp(MockProvider(delay=0.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        # mount ResumePrompt(全量 force_show;_shown 空 → 全量都是 delta)
        await app._refresh_resume_prompt(force_show=True)
        await pilot.pause()
        prompt = app.query_one("ResumePrompt")
        assert set(prompt._agent_ids) == {"sub-A", "sub-B"}  # noqa: SLF001

        # 按 2/No:widget.action_ignore → bg task _on_resume_dismissed(整批 ids)
        await prompt.action_ignore()
        await pilot.pause()
        await _drain_bg_tasks(app)

        # (a) 整批 meta=lost(落盘)
        ma = read_subagent_meta(sub_a_path)
        mb = read_subagent_meta(sub_b_path)
        assert ma is not None and ma.status == "lost"
        assert mb is not None and mb.status == "lost"

        # (b) _reminder_turn 撤回(无可续 → None;_rewrite_reminder([]) 撤旧不重注)
        assert app._reminder_turn is None  # noqa: SLF001

        # (c) _dismissal_turn 出现,含整批 + resume_agent 反信号
        assert app._dismissal_turn is not None  # noqa: SLF001
        dtxt = app._dismissal_turn.messages[0].content[0].text  # noqa: SLF001
        assert "sub-A" in dtxt and "sub-B" in dtxt
        assert "resume_agent" in dtxt

        # (d) _lost_agent_ids 累积整批(反信号源)
        assert app._lost_agent_ids == {"sub-A", "sub-B"}  # noqa: SLF001

        # (e) widget 已卸载(action_ignore 内 self.remove())
        assert list(app.query("ResumePrompt")) == []

        # (f) 重启:再 _refresh(force_show=True) → 不弹(discover 排除 lost)
        await app._refresh_resume_prompt(force_show=True)
        await pilot.pause()
        assert list(app.query("ResumePrompt")) == [], "lost 的 agent 重启不应再弹"


# ---- 用例 4:按 1/Yes → submit("继续") ----


async def test_press_1_yes_submits_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """按 1/Yes → _on_resume_confirmed → controller.submit("继续")。

    用 spy 替换 controller.submit(避免触发真实 agent 轮次);其余 widget/app/controller
    全真实。验证 widget.action_confirm → app._on_resume_confirmed → controller.submit 的
    端到接线。
    """
    store = _make_store(tmp_path, "e2e-yes")
    _seed_meta(store, "sub-A", status="cancelled")

    app = BirdApp(MockProvider(delay=0.0), store=store, resume=False)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._refresh_resume_prompt(force_show=True)
        await pilot.pause()
        prompt = app.query_one("ResumePrompt")
        assert prompt._agent_ids == ["sub-A"]  # noqa: SLF001

        # spy controller.submit(避免跑真实 agent_loop;只验调用入参)
        submitted: list[str] = []

        async def _spy_submit(text: str) -> None:
            submitted.append(text)

        monkeypatch.setattr(app.controller, "submit", _spy_submit)

        # 按 1/Yes:widget.action_confirm → bg task _on_resume_confirmed → submit("继续")
        await prompt.action_confirm()
        await pilot.pause()
        await _drain_bg_tasks(app)

        assert submitted == ["继续"], "1/Yes 应触发 controller.submit('继续')"
        # widget 已卸载(action_confirm 内 self.remove())
        assert list(app.query("ResumePrompt")) == []
