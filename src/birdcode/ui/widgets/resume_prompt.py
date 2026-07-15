# src/birdcode/ui/widgets/resume_prompt.py
"""橙色"有未完成任务"提示 widget(resume 检测到非终态子 agent meta 时 mount)。

仿 ChoicePrompt(Vertical, can_focus=True,见 memory Textual Static can_focus 坑):
自带快捷键的容器必须 can_focus=True,否则按键无反应(只剩 Esc/shift+tab 走 app)。
列各可续跑子 agent(agent_id + 描述≤100)+ 授权/忽略 affordance。

授权闸门(T13):未确认前 resume_subagent 不被触发。确认 = 聚焦本 widget 按 Enter
(action_confirm)→ controller.submit("继续") 把"继续"作为用户轮送主 agent;主 agent
见 mount 时已注入的 reminder 后调 resume_agent → resume_subagent。action_confirm 直调
TurnController.submit,不经 InputArea.Submitted → _maybe_route_continue(T12 路由),
故不会重复 mount/重复注入 reminder/死循环。或按 i 忽略(remove,不触发 resume)。
"""
from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static


class ResumePrompt(Vertical):
    """橙色"有未完成任务"提示(可续跑子 agent 清单 + 忽略 affordance)。"""

    can_focus = True  # 容器聚焦才消费自带 BINDINGS(i=忽略),否则按键无反应(见 memory)

    DEFAULT_CSS = """
    ResumePrompt {
        border: solid orange;
        background: $surface;
        color: $text;
        padding: 0 1;
        margin: 1 0;
        width: 100%;
        height: auto;
    }
    ResumePrompt > .title {
        color: orange;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "继续", show=True),
        Binding("i", "ignore", "忽略", show=True),
    ]

    def __init__(self, descriptions: list[tuple[str, str]]) -> None:
        super().__init__()
        self._desc = descriptions
        # 幂等:防 Enter 连按在 submit await(可能很长,等主 agent 整轮)期间重复触发。
        self._done = False

    def compose(self) -> ComposeResult:
        # markup=False:描述可能含 [aid] 形式的文本,避免被 Rich 当 markup 标签解析;
        # 橙色样式由 CSS .title 类提供,不依赖 markup。
        yield Static("⚠ 有未完成的任务,是否需要继续执行?", classes="title", markup=False)
        for aid, desc in self._desc:
            yield Static(f"  • [{aid}] {desc[:100]}", markup=False)
        yield Static("  (聚焦本框按 Enter 确认继续;或按 i 忽略)", markup=False)

    async def action_confirm(self) -> None:
        """用户授权(Enter)→ 把"继续"送主 agent → 主 agent 见 reminder 后调 resume_agent。

        直调 controller.submit(经 TurnController 入队 → 主 agent 循环),不经
        InputArea.Submitted → _on_submitted → _maybe_route_continue(T12 "继续"路由):
        T12 路由只由 InputArea.Submitted UI 事件触发,本路径与之井水不犯河水,故无死循环 /
        无重复 mount(reminder Turn 由 _resume_reminder_injected flag + mount 去重把关)。
        reminder 已在 mount 本 widget 时(_inject_resumable_reminder)注入 LLM 上下文,
        此处只补"继续"用户轮。

        submit 必须【后台 create_task】,绝不内联 await:内联 await 会阻塞 Textual 消息泵(实测
        确认——awaited async action 占住消息派发)。submit 整轮里主 agent 会调 resume_agent →
        挂底部 PermissionPrompt;若 pump 被占住,该提示无法响应任何按键(箭头/y/n/enter 全卡死),
        而 resume_agent 又在 await 权限 future(只能由按键 resolve)→ 死锁。改 create_task 后
        action 立即返回、pump 放行,权限提示正常响应按键(箭头恢复)。与 send_user_message
        (app.py)同模式,经 app._bg_tasks 追踪 + _reap_bg_task 回收异常。

        幂等:_done 防 Enter 连按重复 submit(remove 后 widget 已不在,亦不再收事件)。
        """
        if self._done:
            return
        self._done = True
        self.remove()  # 先卸载(即时反馈),再后台 submit(不阻塞消息泵)
        task = asyncio.create_task(self.app.controller.submit("继续"))  # type: ignore[attr-defined]
        self.app._bg_tasks.add(task)  # type: ignore[attr-defined]
        task.add_done_callback(self.app._reap_bg_task)  # type: ignore[attr-defined]

    def action_ignore(self) -> None:
        """用户按 i → 卸载本 widget(不触发 resume)。"""
        self.remove()
