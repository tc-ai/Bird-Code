# tests/ui/test_resume_authorization_gate.py
"""授权闸门——所有续跑必须用户显式确认。

闸门机制:ResumePrompt 挂载后只注入 LLM reminder + 显示橙色提示,
**不**触发 resume_subagent。唯有用户聚焦本 widget 按 1(action_confirm)→
app._on_resume_confirmed → controller.submit("继续") → 主 agent 调 resume_agent →
resume_subagent(Task 5 才在 BirdApp 实装 _on_resume_confirmed;本测试在 _GateApp
直接提供桩)。

Task 4 改造后:widget 不再直调 controller.submit,改调 app._on_resume_confirmed();
按 2(action_ignore)→ app._on_resume_dismissed(整批 ids)。整批统一决策。

测试策略:用最小 _GateApp(仅 #scroll + _on_resume_confirmed/_on_resume_dismissed 桩 +
_bg_tasks/_reap_bg_task),绕过完整 BirdApp/真实 LLM。pilot 驱动聚焦 + 按键,
断言桩被调 + widget 卸载。
"""
from __future__ import annotations

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical

from birdcode.ui.widgets.resume_prompt import ResumePrompt


class _GateApp(App):
    """最小 App:#scroll 挂载点 + _on_resume_confirmed/_on_resume_dismissed 桩。

    Task 4 widget 接 app 这俩方法;Task 5 才在 BirdApp 实装(confirmed → submit("继续"),
    dismissed → 整批 lost + 撤回 reminder + 反信号)。此处桩仅计数 / 可选阻塞,供闸门测试。
    _bg_tasks/_reap_bg_task 镜像 BirdApp:action_confirm 把 _on_resume_confirmed 作后台 task,
    需 app 提供强引用容器 + 异常回收。
    """

    def __init__(self) -> None:
        super().__init__()
        self._bg_tasks: set[asyncio.Task] = set()
        self.confirmed_calls: int = 0
        self.dismissed_calls: list[list[str]] = []
        # 阻塞闸门(None=不阻塞);_on_resume_confirmed 跑期间 await 它 → 模拟整轮 agent 永不完成。
        self.confirmed_gate: asyncio.Event | None = None

    def compose(self) -> ComposeResult:
        yield Vertical(id="scroll")

    async def _on_resume_confirmed(self) -> None:
        """Task 5 会落 controller.submit("继续");此处仅计数 + 可选阻塞(测泵不卡死)。"""
        self.confirmed_calls += 1
        if self.confirmed_gate is not None:
            await self.confirmed_gate.wait()

    async def _on_resume_dismissed(self, ids: list[str]) -> None:
        """Task 5 会落 _mark_lost_agent + 撤回 reminder + 反信号;此处仅记录入参。"""
        self.dismissed_calls.append(list(ids))

    def _reap_bg_task(self, task: asyncio.Task) -> None:
        """镜像 BirdApp._reap_bg_task:回收引用 + 取走异常(免 'never retrieved' 警告)。"""
        self._bg_tasks.discard(task)
        if task.cancelled():
            return
        task.exception()  # retrieve(测试桩不 log)


@pytest.mark.asyncio
async def test_resume_not_triggered_without_authorization():
    """mount ResumePrompt 但用户未确认 → _on_resume_confirmed 不被调。"""
    async with _GateApp().run_test(size=(80, 24)) as pilot:
        await pilot.app.query_one("#scroll").mount(
            ResumePrompt([("sub-1", "分析鉴权")])
        )
        await pilot.pause()
        # 仅 mount,不 action_confirm
        assert pilot.app.confirmed_calls == 0
        assert pilot.app.dismissed_calls == []


@pytest.mark.asyncio
async def test_resume_triggered_after_authorization():
    """用户聚焦 ResumePrompt 按 1(action_confirm)→ app._on_resume_confirmed 被调。

    同时验证:widget 被 remove(闸门确认后即卸载),不再留在对话区。
    """
    async with _GateApp().run_test(size=(80, 24)) as pilot:
        prompt = ResumePrompt([("sub-1", "分析鉴权")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()

        # 用户授权:聚焦 + 1 触发 action_confirm(binding: 1 → confirm)
        prompt.focus()
        await pilot.press("1")
        await pilot.pause()

        assert pilot.app.confirmed_calls == 1
        # widget 已卸载(确认后即 remove)
        assert not list(pilot.app.query("ResumePrompt"))


@pytest.mark.asyncio
async def test_action_ignore_triggers_dismissed_with_ids():
    """用户聚焦按 2(action_ignore)→ app._on_resume_dismissed(整批 ids)被调;widget 卸载。"""
    async with _GateApp().run_test(size=(80, 24)) as pilot:
        prompt = ResumePrompt([("sub-1", "分析"), ("sub-2", "改测试")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()

        prompt.focus()
        await pilot.press("2")
        await pilot.pause()

        assert pilot.app.confirmed_calls == 0
        assert pilot.app.dismissed_calls == [["sub-1", "sub-2"]]
        assert not list(pilot.app.query("ResumePrompt"))


@pytest.mark.asyncio
async def test_action_confirm_idempotent_against_double_press():
    """1 连按在 _on_resume_confirmed 跑期间不得重复触发(幂等 _done 把关)。"""
    async with _GateApp().run_test(size=(80, 24)) as pilot:
        prompt = ResumePrompt([("sub-1", "分析")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()

        prompt.focus()
        await prompt.action_confirm()  # 第一次:授权
        await pilot.pause()
        await prompt.action_confirm()  # 第二次:已 _done,应跳过(且 widget 已 unmount)
        await pilot.pause()

        assert pilot.app.confirmed_calls == 1  # 不重复


def test_resume_prompt_has_confirm_binding():
    """1 键绑定 action_confirm(BINDINGS 注册,1=续跑/确认)。"""
    assert any(
        b.key == "1" and b.action == "confirm" for b in ResumePrompt.BINDINGS
    )


def test_resume_prompt_has_ignore_binding():
    """2 键绑定 action_ignore(BINDINGS 注册,2=丢弃)。"""
    assert any(
        b.key == "2" and b.action == "ignore" for b in ResumePrompt.BINDINGS
    )


@pytest.mark.asyncio
async def test_action_confirm_does_not_block_pump_on_long_confirmed():
    """review #5 / 箭头键 bug:action_confirm 不得内联 await _on_resume_confirmed(阻塞泵)。

    resume 经橙色 ResumePrompt 的 1 → action_confirm → _on_resume_confirmed 触发整轮 agent,
    agent 会调 resume_agent 挂权限提示。若 action_confirm 内联 await,Textual 消息泵被占住 →
    权限提示无法响应任何按键(箭头/y/n/enter),且 resume_agent 在 await 权限 future(只能由按键
    resolve)→ 死锁。修复:create_task 后台跑 _on_resume_confirmed,action_confirm 立即返回。

    此处用「_on_resume_confirmed 永不完成」的 gate 验证:action_confirm 能返回(pilot.press 不卡死,
    测试不超时)、_on_resume_confirmed 已调度进 _bg_tasks、widget 已移除。修复前内联 await 会让
    pilot.press 挂起直至 gate.set()(永不),测试超时失败。
    """
    async with _GateApp().run_test(size=(80, 24)) as pilot:
        pilot.app.confirmed_gate = asyncio.Event()  # _on_resume_confirmed 永不完成
        prompt = ResumePrompt([("sub-1", "任务")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()
        prompt.focus()
        # 若 action_confirm 内联 await,此行会挂起直到 gate.set()(永不)→ 测试超时。
        await pilot.press("1")
        await pilot.pause()
        # _on_resume_confirmed 已调度(进后台 task),action_confirm 已返回(widget 已移除)
        assert pilot.app.confirmed_calls == 1
        # _on_resume_confirmed 跑在后台 task 里,未占住 pump
        assert pilot.app._bg_tasks  # noqa: SLF001
        assert not list(pilot.app.query("ResumePrompt"))


def test_resume_prompt_action_confirm_method_exists():
    """action_confirm 方法存在(实际按键行为见上方 async 测试)。"""
    assert callable(getattr(ResumePrompt, "action_confirm", None))


@pytest.mark.asyncio
async def test_enter_default_highlights_yes_and_confirms():
    """enter 提交当前 ▶ 高亮;默认 sel=0(Yes)→ action_confirm(对齐 PermissionPrompt)。

    防 review Bug A:旧 ResumePrompt 无 ↑↓/enter,只认 1/2,与另两套菜单交互不一致。
    """
    async with _GateApp().run_test(size=(80, 24)) as pilot:
        prompt = ResumePrompt([("sub-1", "分析鉴权")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()
        prompt.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.confirmed_calls == 1
        assert pilot.app.dismissed_calls == []


@pytest.mark.asyncio
async def test_down_then_enter_ignores():
    """↓ 移光标到 No(sel=1)→ enter → action_ignore(整批 ids)。"""
    async with _GateApp().run_test(size=(80, 24)) as pilot:
        prompt = ResumePrompt([("sub-1", "分析"), ("sub-2", "改测试")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()
        prompt.focus()
        await pilot.press("down")
        await pilot.pause()
        assert prompt._sel == 1  # noqa: SLF001 - 光标移到 No
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.confirmed_calls == 0
        assert pilot.app.dismissed_calls == [["sub-1", "sub-2"]]


@pytest.mark.asyncio
async def test_up_down_move_cursor_without_deciding():
    """↑↓ 在 Yes/No 间循环移动光标(回绕),不触发任何决策。"""
    async with _GateApp().run_test(size=(80, 24)) as pilot:
        prompt = ResumePrompt([("sub-1", "分析")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()
        prompt.focus()
        assert prompt._sel == 0  # noqa: SLF001
        await pilot.press("down")
        await pilot.pause()
        assert prompt._sel == 1  # noqa: SLF001
        await pilot.press("down")  # 回绕到 Yes
        await pilot.pause()
        assert prompt._sel == 0  # noqa: SLF001
        await pilot.press("up")  # 回绕到 No
        await pilot.pause()
        assert prompt._sel == 1  # noqa: SLF001
        assert pilot.app.confirmed_calls == 0
        assert pilot.app.dismissed_calls == []
