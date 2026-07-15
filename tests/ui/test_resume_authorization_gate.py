# tests/ui/test_resume_authorization_gate.py
"""Task 13:授权闸门——所有续跑必须用户显式确认。

闸门机制:ResumePrompt 挂载后(T12 路由 / on_mount)只注入 LLM reminder + 显示橙色提示,
**不**触发 resume_subagent。唯有用户聚焦本 widget 按 Enter(action_confirm)→
controller.submit("继续") → 主 agent 见 [reminder + "继续"] → 调 resume_agent 工具 →
resume_subagent。

关键交互(与 T12 "继续"路由的关系):action_confirm 调 controller.submit 直进 TurnController
队列(conversation.py),**不经 InputArea.Submitted → _on_submitted → _maybe_route_continue**,
故不会重复 mount / 重复注入 reminder / 死循环。详见 test_resume_prompt_no_reroute_on_confirm。

测试策略:用最小 _GateApp(仅 #scroll + 可替换 controller),绕过完整 BirdApp/真实 LLM。
_ForwardingController 模拟「主 agent 收到"继续"后调 resume_agent 工具」(resume_subagent 被
monkeypatch 成 fake),从而在单测里验完整闸门路径:confirm → submit → 工具 → resume_subagent。
"""
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical

from birdcode.agents.resume import ResumeResult
from birdcode.tools.resume_agent_tool import ResumeAgentTool
from birdcode.ui.widgets.resume_prompt import ResumePrompt


class _ForwardingController:
    """模拟主 agent:submit("继续") → 调 resume_agent 工具(绕过真实 LLM)。

    闸门测试用:验 action_confirm 的 submit 经工具链触达 resume_subagent。
    resume_subagent 已被 monkeypatch,deps 不被真实消费,故 ResumeAgentTool 的 deps 随便给。
    """

    def __init__(self, tool: ResumeAgentTool | None = None) -> None:
        self.tool = tool
        self.submits: list[str] = []

    async def submit(self, text: str) -> None:
        self.submits.append(text)
        if self.tool is not None and text == "继续":
            await self.tool.execute(agent_id="sub-1", direction="继续")


class _GateApp(App):
    """最小 App:#scroll 挂载点 + controller 属性(供 ResumePrompt.action_confirm 访问)。"""

    def __init__(self, controller: _ForwardingController) -> None:
        super().__init__()
        self._ctrl = controller

    def compose(self) -> ComposeResult:
        yield Vertical(id="scroll")

    @property
    def controller(self) -> _ForwardingController:
        return self._ctrl


def _patch_resume(monkeypatch) -> dict:
    """monkeypatch resume_subagent 成计数 fake,返回计数盒。"""
    called: dict[str, int] = {"n": 0}

    async def fake_resume(**kw):  # noqa: ANN003 - 测试 fake,吞所有参数
        called["n"] += 1
        return ResumeResult(outcome="sync_done", text="ok")

    monkeypatch.setattr("birdcode.tools.resume_agent_tool.resume_subagent", fake_resume)
    return called


@pytest.mark.asyncio
async def test_resume_not_triggered_without_authorization(monkeypatch):
    """mount ResumePrompt 但用户未确认 → resume_subagent 不被调,submit 不被调。"""
    called = _patch_resume(monkeypatch)
    tool = ResumeAgentTool(deps=object())  # deps 经 fake **kw 吞掉,无需真实 ResumeDeps
    ctrl = _ForwardingController(tool=tool)

    async with _GateApp(ctrl).run_test(size=(80, 24)) as pilot:
        await pilot.app.query_one("#scroll").mount(
            ResumePrompt([("sub-1", "分析鉴权")])
        )
        await pilot.pause()
        # 仅 mount,不 action_confirm
        assert called["n"] == 0
        assert ctrl.submits == []


@pytest.mark.asyncio
async def test_resume_triggered_after_authorization(monkeypatch):
    """用户聚焦 ResumePrompt 按 Enter(action_confirm)→ submit("继续") → 工具 → resume_subagent。

    同时验证:widget 被 remove(闸门确认后即卸载),不再留在对话区。
    """
    called = _patch_resume(monkeypatch)
    tool = ResumeAgentTool(deps=object())
    ctrl = _ForwardingController(tool=tool)

    async with _GateApp(ctrl).run_test(size=(80, 24)) as pilot:
        prompt = ResumePrompt([("sub-1", "分析鉴权")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()

        # 用户授权:聚焦 + Enter 触发 action_confirm(binding: enter → confirm)
        prompt.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert called["n"] == 1  # 经 submit → resume_agent 工具 → resume_subagent
        assert ctrl.submits == ["继续"]
        # widget 已卸载(确认后即 remove)
        assert not list(pilot.app.query("ResumePrompt"))


@pytest.mark.asyncio
async def test_action_confirm_idempotent_against_double_press(monkeypatch):
    """Enter 连按在 submit await 期间不得重复触发 submit/恢复(幂等 _done 把关)。"""
    called = _patch_resume(monkeypatch)
    tool = ResumeAgentTool(deps=object())
    ctrl = _ForwardingController(tool=tool)

    async with _GateApp(ctrl).run_test(size=(80, 24)) as pilot:
        prompt = ResumePrompt([("sub-1", "分析")])
        await pilot.app.query_one("#scroll").mount(prompt)
        await pilot.pause()

        prompt.focus()
        await prompt.action_confirm()  # 第一次:授权
        await pilot.pause()
        await prompt.action_confirm()  # 第二次:已 _done,应跳过(且 widget 已 unmount)
        await pilot.pause()

        assert called["n"] == 1  # 不重复
        assert ctrl.submits == ["继续"]


def test_resume_prompt_has_confirm_binding():
    """enter 键绑定 action_confirm(BINDINGS 注册,enter=继续/确认)。"""
    assert any(
        b.key == "enter" and b.action == "confirm" for b in ResumePrompt.BINDINGS
    )


def test_resume_prompt_action_confirm_method_exists():
    """action_confirm 方法存在(实际 submit/remove 需 mount,见上方 async 测试)。"""
    assert callable(getattr(ResumePrompt, "action_confirm", None))
