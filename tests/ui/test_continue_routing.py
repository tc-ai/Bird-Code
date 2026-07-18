# tests/ui/test_continue_routing.py
"""Task 12: "继续" 路由 —— 自动发现可续跑子 agent,有则 mount+注入(不透传 submit),无则透传。

入口真实位置:BirdApp._on_submitted(InputArea.Submitted) 在调 controller.submit 之前拦截。
路由判定抽成 _maybe_route_continue(text) -> bool:True=已处理(调用方不再 submit),
False=未处理(透传 submit 当普通消息)。严格匹配 strip 后 == "继续" 才触发。

测试两层:
- _maybe_route_continue 路由判定(单元,SimpleNamespace fake + 真实 store 扫描)。
- _on_submitted 端到接线(brief Step 1 两测试:submit 是否被调 + 参数)。
"""

from __future__ import annotations

import json
import types
from types import SimpleNamespace

import pytest

from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore
from birdcode.ui.app import BirdApp


def _make_store(tmp_path) -> SessionStore:
    ctx = SessionContext(session_id="resume", cwd="C:/p", version="0.1.0", git_branch=None)
    return SessionStore(ctx, tmp_path, root=tmp_path)


def _write_meta_file(
    store: SessionStore, agent_id: str, status: str, prompt: str = "分析模块"
) -> None:
    """写 agent-<id>.meta.json 到 store 的 subagents 目录(镜像 test_resumable_reminder)。"""
    from birdcode.session import paths

    p = paths.subagent_meta_path(
        store.root,
        store.ctx.session_id,
        store.project_root,
        agent_id,
        worktree_name=store.worktree_name,
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "agentId": agent_id,
                "agentType": "general-purpose",
                "toolUseId": f"tu-{agent_id}",
                "status": status,
                "prompt": prompt,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class _FakeController:
    """记录 submit 调用(供 _on_submitted 端到测试观测是否透传 + 参数)。"""

    def __init__(self) -> None:
        self.history: list = []
        self.submitted: list[str] = []

    async def submit(self, text: str) -> None:
        self.submitted.append(text)


def _build_fake_self(store: SessionStore | None, controller: _FakeController) -> SimpleNamespace:
    """构造覆盖 _on_submitted + _maybe_route_continue 全部 self 属性的 fake。

    _on_submitted 用 self.controller.submit;_maybe_route_continue 用 self._controller 校验存在。
    两名指向同一 ctrl 对象(真实 BirdApp 里 controller 是 _controller 的 property)。

    Task 7 后:_maybe_route_continue 命中"继续"调 self._refresh_resume_prompt(force_show=True),
    故 fake 挂 _refresh_resume_prompt(记录 force_show 值)替代旧 _inject_resumable_reminder。
    """
    refresh_calls: list[bool] = []

    async def _fake_refresh(*, force_show: bool = False) -> None:
        refresh_calls.append(force_show)

    ns = SimpleNamespace(
        _store=store,
        _controller=controller,
        controller=controller,
        _bg_tasks=set(),
        _reap_bg_task=lambda _t: None,
    )
    ns.refresh_calls = refresh_calls
    ns._refresh_resume_prompt = _fake_refresh
    # _on_submitted 内调 self._maybe_route_continue → 需把真方法绑到 fake self 上
    # (测试里 _on_submitted 是以未绑形式 BirdApp._on_submitted(fake_self, event) 调的)。
    ns._maybe_route_continue = types.MethodType(BirdApp._maybe_route_continue, ns)
    return ns


# —— _maybe_route_continue 路由判定(单元)——


@pytest.mark.asyncio
async def test_route_continue_resumable_triggers_refresh_and_blocks_submit(tmp_path):
    """有可续跑 → "继续" → 调 _refresh_resume_prompt(force_show=True) + 返回 True(拦截 submit)。"""
    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "running")  # 非终态 → 可续跑
    ctrl = _FakeController()
    fake_self = _build_fake_self(store, ctrl)

    handled = await BirdApp._maybe_route_continue(fake_self, "继续")

    assert handled is True  # 已处理 → 调用方不 submit
    assert fake_self.refresh_calls == [True]  # force_show=True 被调(Task 7)


@pytest.mark.asyncio
async def test_route_continue_resumable_tolerates_surrounding_whitespace(tmp_path):
    """strip 后 == "继续" 即触发(前后空白容忍)。"""
    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "idle")
    ctrl = _FakeController()
    fake_self = _build_fake_self(store, ctrl)

    handled = await BirdApp._maybe_route_continue(fake_self, "  继续  ")

    assert handled is True
    assert fake_self.refresh_calls == [True]


@pytest.mark.asyncio
async def test_route_continue_none_passes_through(tmp_path):
    """无可续跑 → "继续" 返回 False(透传 submit),不触发 refresh。"""
    store = _make_store(tmp_path)  # 无 meta 文件
    ctrl = _FakeController()
    fake_self = _build_fake_self(store, ctrl)

    handled = await BirdApp._maybe_route_continue(fake_self, "继续")

    assert handled is False
    assert fake_self.refresh_calls == []


@pytest.mark.asyncio
async def test_route_non_continue_input_passes_through():
    """非"继续"输入一律 False(strip 后不等"继续"),且不触 _store(无 store 也不抛)。"""
    # strip != "继续" 先返回,fake 无 _store/_controller 也不应抛
    fake_self = SimpleNamespace(_refresh_resume_prompt=lambda **kw: None)

    for text in ["你好", "继续 分析", "继续一下", "/继续", "continue", "", "继续abc"]:
        handled = await BirdApp._maybe_route_continue(fake_self, text)
        assert handled is False, f"{text!r} 不应被路由拦截"


@pytest.mark.asyncio
async def test_route_continue_no_store_passes_through():
    """无 store → "继续" 透传(不抛、不 refresh)。"""
    ctrl = _FakeController()
    fake_self = _build_fake_self(None, ctrl)

    handled = await BirdApp._maybe_route_continue(fake_self, "继续")

    assert handled is False
    assert fake_self.refresh_calls == []


# —— _on_submitted 端到接线(brief Step 1 两测试)——


async def _drain_bg_tasks(fake_self: SimpleNamespace) -> None:
    """等 _on_submitted 起的后台 submit task 跑完(若有)。"""
    for t in list(fake_self._bg_tasks):
        await t


@pytest.mark.asyncio
async def test_continue_with_resumable_surfaces_prompt_and_blocks_submit(tmp_path):
    """brief 1:有可续跑 → 输入"继续" → refresh+mount,_on_submitted 不透传 submit。"""
    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "running")
    ctrl = _FakeController()
    fake_self = _build_fake_self(store, ctrl)
    event = SimpleNamespace(text="继续")

    await BirdApp._on_submitted(fake_self, event)
    await _drain_bg_tasks(fake_self)

    assert fake_self.refresh_calls == [True]  # refresh(含 mount ResumePrompt)force_show=True
    assert ctrl.submitted == []  # 不透传 submit


@pytest.mark.asyncio
async def test_continue_with_none_passes_through(tmp_path):
    """brief 测试 2:无可续跑 → "继续" 透传 controller.submit 当普通消息。"""
    store = _make_store(tmp_path)  # 无 meta
    ctrl = _FakeController()
    fake_self = _build_fake_self(store, ctrl)
    event = SimpleNamespace(text="继续")

    await BirdApp._on_submitted(fake_self, event)
    await _drain_bg_tasks(fake_self)

    assert ctrl.submitted == ["继续"]
    assert fake_self.refresh_calls == []  # 不 refresh


@pytest.mark.asyncio
async def test_normal_text_still_submits(tmp_path):
    """回归:普通(非"继续")输入不受路由影响,正常 submit。"""
    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "running")  # 即便有可续跑,非"继续"也不拦截
    ctrl = _FakeController()
    fake_self = _build_fake_self(store, ctrl)
    event = SimpleNamespace(text="帮我改一下测试")

    await BirdApp._on_submitted(fake_self, event)
    await _drain_bg_tasks(fake_self)

    assert ctrl.submitted == ["帮我改一下测试"]
    assert fake_self.refresh_calls == []


# —— T12 dedup 端到端:重复"继续"不累积 reminder Turn / ResumePrompt widget ——


class _FakeScrollWithQuery:
    """伪 #scroll 容器 + 配套 fake_self.query:支持 widget 去重检查。

    mounted 跟踪已挂 ResumePrompt;query(selector) 返回当前 mounted 中匹配的(模拟真实 DOM 查询)。
    """

    def __init__(self) -> None:
        self.mounted: list = []

    async def mount(self, widget):
        self.mounted.append(widget)

    def remove(self, widget):
        if widget in self.mounted:
            self.mounted.remove(widget)


def _build_real_inject_fake_self(store, controller, scroll):
    """构造 fake_self,绑真实 _inject/_refresh/_mount 全链(测 dedup 副作用)。

    Task 3 后:_inject_resumable_reminder → _refresh_resume_prompt → _rewrite_reminder +
    _mount_resume_prompt。绑真实 _refresh + 依赖(_current_subagents_dir、_rewrite_reminder)。
    """
    from birdcode.ui.widgets.resume_prompt import ResumePrompt

    ns = SimpleNamespace(
        _store=store,
        _controller=controller,
        controller=controller,
        _reminder_turn=None,
        _dismissal_turn=None,
        _shown_agent_ids=set(),
        _lost_agent_ids=set(),
    )

    def _query(selector):
        # 模拟 Textual DOM 查询:selector="ResumePrompt" → 返回当前 scroll 中该类型 widget
        if selector == "ResumePrompt":
            return [w for w in scroll.mounted if isinstance(w, ResumePrompt)]
        return []

    def _query_one(selector):
        if selector == "#scroll":
            return scroll
        return None

    ns.query = _query
    ns.query_one = _query_one
    # 绑真实方法(未绑形式 → 用 types.MethodType 绑到 fake_self,使 self 正确传递)
    ns._current_subagents_dir = types.MethodType(BirdApp._current_subagents_dir, ns)
    ns._rewrite_reminder = types.MethodType(BirdApp._rewrite_reminder, ns)
    ns._refresh_resume_prompt = types.MethodType(BirdApp._refresh_resume_prompt, ns)
    ns._inject_resumable_reminder = types.MethodType(
        BirdApp._inject_resumable_reminder, ns
    )
    ns._mount_resume_prompt = types.MethodType(BirdApp._mount_resume_prompt, ns)
    ns._maybe_route_continue = types.MethodType(BirdApp._maybe_route_continue, ns)
    return ns


@pytest.mark.asyncio
async def test_repeated_continue_does_not_accumulate_turn_or_widget(tmp_path):
    """Task 3/7 dedup e2e:连输 3 次"继续" → reminder Turn 仍 1 条 + ResumePrompt 仍 1 个。

    Task 7 后 _maybe_route_continue 走 force_show=True:_rewrite_reminder 撤旧重注
    (单条动态,Turn 恒 1 条)+ _mount_resume_prompt 撤旧重挂(单条 widget,恒 1 个)。
    submit 阻断语义不变。
    """
    from birdcode.ui.widgets.resume_prompt import ResumePrompt

    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "running")
    ctrl = _FakeController()
    scroll = _FakeScrollWithQuery()
    fake_self = _build_real_inject_fake_self(store, ctrl, scroll)

    for _ in range(3):
        await BirdApp._maybe_route_continue(fake_self, "继续")

    # Turn 去重:3 次"继续" → history 仍只 1 条 reminder Turn(撤旧重注)
    assert len(ctrl.history) == 1
    assert fake_self._reminder_turn is ctrl.history[0]
    # widget:force_show=True 每次 remove 旧 + mount 新 → 任意时刻 #scroll 恒 1 个 ResumePrompt
    resume_widgets = [w for w in scroll.mounted if isinstance(w, ResumePrompt)]
    assert len(resume_widgets) == 1
    # submit 阻断语义不变:3 次都拦截,submit 未被调
    assert ctrl.submitted == []
    # _shown 已记录 A1(Task 3 差集基准,替代旧 _resume_reminder_injected flag)
    assert fake_self._shown_agent_ids == {"A1"}


@pytest.mark.asyncio
async def test_repeated_continue_remounts_widget_after_user_ignores(tmp_path):
    """Task 7 dedup e2e 边界:用户忽略移除 ResumePrompt 后,再"继续"(force_show=True)→ 重挂 widget。

    Task 7 后 _maybe_route_continue 调 _refresh_resume_prompt(force_show=True):
    force_show 绕过 _shown 差集 → 每次都 mount 全量 R(_mount_resume_prompt remove 旧 + mount 新)。
    Turn 仍不累积(撤旧重注恒 1 条)。
    """
    from birdcode.ui.widgets.resume_prompt import ResumePrompt

    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "running")
    ctrl = _FakeController()
    scroll = _FakeScrollWithQuery()
    fake_self = _build_real_inject_fake_self(store, ctrl, scroll)

    # 第一次"继续":注入 Turn + mount widget(force_show=True → 全量 R 展示)
    await BirdApp._maybe_route_continue(fake_self, "继续")
    assert len(ctrl.history) == 1
    assert len([w for w in scroll.mounted if isinstance(w, ResumePrompt)]) == 1

    # 用户忽略 → widget 被移除
    scroll.remove(scroll.mounted[0])
    assert [w for w in scroll.mounted if isinstance(w, ResumePrompt)] == []

    # 第二次"继续"(force_show=True):绕过差集 → 重挂 widget;Turn 仍不累积(撤旧重注)
    await BirdApp._maybe_route_continue(fake_self, "继续")
    assert len(ctrl.history) == 1  # Turn 仍只 1 条(撤旧重注)
    assert len([w for w in scroll.mounted if isinstance(w, ResumePrompt)]) == 1  # force_show 重挂
