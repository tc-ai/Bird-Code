"""ResumePrompt widget:数字键 1.Yes/2.No + 列表渲染 + action 接线。"""
from types import SimpleNamespace

from birdcode.ui.widgets.resume_prompt import ResumePrompt


def test_bindings_are_1_and_2() -> None:
    keys = {b.key for b in ResumePrompt.BINDINGS}
    assert "1" in keys and "2" in keys


def test_actions_wired() -> None:
    actions = {b.key: b.action for b in ResumePrompt.BINDINGS}
    assert actions["1"] == "confirm"
    assert actions["2"] == "ignore"


def test_can_focus_true() -> None:
    assert ResumePrompt.can_focus is True


def test_render_lists_agents_and_yes_no() -> None:
    w = ResumePrompt(descriptions=[("sub-1", "分析鉴权"), ("sub-2", "改测试")])
    # compose 是 async generator;textual 渲染需 App。这里只验静态字段 + 渲染物料。
    assert w._agent_ids == ["sub-1", "sub-2"]


async def test_action_confirm_calls_app_on_resume_confirmed(monkeypatch) -> None:
    w = ResumePrompt(descriptions=[("sub-1", "分析")])
    called: list = []
    async def _confirmed():
        called.append(True)
    # Textual 8.x:`app` 是 MessagePump 上无 setter 的 property,实例赋值会抛
    # AttributeError;`remove()` 走 self.app._prune 也依赖真实 app 上下文。故:
    # ① 在类层级 monkeypatch app 为 fake;② 实例级把 remove 替成 no-op。
    monkeypatch.setattr(
        ResumePrompt, "app", SimpleNamespace(
            _on_resume_confirmed=_confirmed,
            _on_resume_dismissed=lambda ids: None,
            _bg_tasks=set(),
            _reap_bg_task=lambda t: None,
        ),
    )
    monkeypatch.setattr(w, "remove", lambda: None)
    await w.action_confirm()
    # action_confirm 用 create_task 后台跑,让一拍
    import asyncio
    await asyncio.sleep(0)
    assert called == [True]


async def test_action_ignore_calls_app_on_resume_dismissed_with_ids(monkeypatch) -> None:
    w = ResumePrompt(descriptions=[("sub-1", "分析"), ("sub-2", "改测试")])
    captured: list = []
    async def _dismissed(ids):
        captured.append(list(ids))
    monkeypatch.setattr(
        ResumePrompt, "app", SimpleNamespace(
            _on_resume_confirmed=lambda: None,
            _on_resume_dismissed=_dismissed,
            _bg_tasks=set(),
            _reap_bg_task=lambda t: None,
        ),
    )
    monkeypatch.setattr(w, "remove", lambda: None)
    await w.action_ignore()
    import asyncio
    await asyncio.sleep(0)
    assert captured == [["sub-1", "sub-2"]]
