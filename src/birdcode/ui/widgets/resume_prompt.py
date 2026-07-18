# src/birdcode/ui/widgets/resume_prompt.py
"""橙色"有未完成任务"提示 widget:数字键 1.Yes 续跑 / 2.No 丢弃(整批统一)。

仿 ChoicePrompt(Vertical, can_focus=True,见 memory Textual Static can_focus 坑):
自带快捷键的容器必须 can_focus=True,否则按键无反应。上面列子 agent(agent_id + 描述≤100),
下面 "1. Yes(续跑)" / "2. No(丢弃)"。整批统一决策:1→submit("继续"),2→整批标 lost。

确认闸门(原 T13):1/Yes 走 app._on_resume_confirmed → submit("继续") → 主 agent 见 reminder
后调 resume_agent;2/No 走 app._on_resume_dismissed(整批 lost + 撤回 reminder + 反信号)。
两个 action 都 create_task 后台跑(不阻塞消息泵;否则后续 PermissionPrompt 按键死锁,见旧注释)。
"""
from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static


class ResumePrompt(Vertical):
    """橙色"有未完成任务"提示(子 agent 清单 + 1.Yes / 2.No)。整批统一决策。"""

    can_focus = True  # 容器聚焦才消费自带 BINDINGS,否则按键无反应(见 memory)

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
        Binding("1", "confirm", "续跑", show=True),
        Binding("2", "ignore", "丢弃", show=True),
    ]

    def __init__(self, descriptions: list[tuple[str, str]]) -> None:
        super().__init__()
        self._desc = descriptions
        self._agent_ids = [aid for aid, _ in descriptions]
        # 幂等:防连按在 create_task 跑完前重复触发。
        self._done = False

    def compose(self) -> ComposeResult:
        # markup=False:描述含 [aid] 文本,避免被 Rich 当 markup 标签解析。
        yield Static("⚠ 有未完成的任务，是否续跑？", classes="title", markup=False)
        for aid, desc in self._desc:
            yield Static(f"  • [{aid}] {desc[:100]}", markup=False)
        yield Static("  1. Yes（续跑）", markup=False)
        yield Static("  2. No（丢弃）", markup=False)

    async def action_confirm(self) -> None:
        """1/Yes → app._on_resume_confirmed → submit("继续")。整批交主 agent 续跑。"""
        if self._done:
            return
        self._done = True
        self.remove()
        task = asyncio.create_task(self.app._on_resume_confirmed())  # type: ignore[attr-defined]
        self.app._bg_tasks.add(task)  # type: ignore[attr-defined]
        task.add_done_callback(self.app._reap_bg_task)  # type: ignore[attr-defined]

    async def action_ignore(self) -> None:
        """2/No → app._on_resume_dismissed(整批 ids):标 lost + 撤回 reminder + 反信号。"""
        if self._done:
            return
        self._done = True
        ids = list(self._agent_ids)
        self.remove()
        task = asyncio.create_task(  # type: ignore[attr-defined]
            self.app._on_resume_dismissed(ids)
        )
        self.app._bg_tasks.add(task)  # type: ignore[attr-defined]
        task.add_done_callback(self.app._reap_bg_task)  # type: ignore[attr-defined]
