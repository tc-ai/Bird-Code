# tests/ui/test_resume_prompt.py
"""Task 11: ResumePrompt widget —— 橙色"有未完成任务"提示渲染。

仿 test_choice_prompt.py:测不依赖 mount 的 compose 文本逻辑(渲染含标题/描述/affordance)。
query_one/remove/focus 等 UI 行为需 mount(Textual pilot),此处留手动验证,只测结构。
"""
from __future__ import annotations

from birdcode.ui.widgets.resume_prompt import ResumePrompt


def _rendered(prompt: ResumePrompt) -> str:
    """compose 产出的 Static.content 拼成单串(便于断言子串)。

    Static.content 是 Textual 8.x 存储原始文本的公共属性(.renderable 已不存在);
    compose() 不需 mount 即可调用(仅构造 Static,不触发布局)。
    """
    return "\n".join(widget.content for widget in prompt.compose())


def test_resume_prompt_shows_header_and_descriptions():
    """标题("有未完成的任务"+"是否需要继续执行") + 各任务 agent_id/描述。"""
    w = ResumePrompt(descriptions=[("sub-1", "分析鉴权"), ("sub-2", "改测试")])
    rendered = _rendered(w)
    assert "有未完成的任务" in rendered
    assert "是否需要继续执行" in rendered
    assert "sub-1" in rendered and "分析鉴权" in rendered
    assert "sub-2" in rendered and "改测试" in rendered


def test_resume_prompt_can_focus_true():
    """自带快捷键的容器必须 can_focus=True(见 memory Textual Static can_focus 坑)。"""
    assert ResumePrompt.can_focus is True


def test_resume_prompt_truncates_long_description_to_100():
    """描述截断 ≤100(与 build_resumable_reminder 同策略,防御未预截断的入参)。"""
    w = ResumePrompt(descriptions=[("sub-1", "x" * 200)])
    rendered = _rendered(w)
    line = [ln for ln in rendered.splitlines() if "sub-1" in ln][0]
    # "[sub-1] " 后是描述,截断后恰为 100
    desc = line.split("] ", 1)[1]
    assert len(desc) == 100


def test_resume_prompt_shows_affordance():
    """affordance 提示用户如何续跑("继续")/忽略("忽略")。"""
    w = ResumePrompt(descriptions=[("sub-1", "分析")])
    rendered = _rendered(w)
    assert "继续" in rendered
    assert "忽略" in rendered


def test_resume_prompt_has_ignore_binding():
    """i 键绑定 action_ignore(BINDINGS 注册,i=忽略)。"""
    assert any(b.key == "i" and b.action == "ignore" for b in ResumePrompt.BINDINGS)


def test_resume_prompt_action_ignore_method_exists():
    """action_ignore 方法定义存在(实际 remove 需 mount,留 pilot/手动验证)。"""
    assert callable(getattr(ResumePrompt, "action_ignore", None))


def test_resume_prompt_empty_descriptions_renders_only_header_and_affordance():
    """空描述列表 → 仍渲染标题 + affordance(防御:不崩,正常调用方传非空)。"""
    w = ResumePrompt(descriptions=[])
    rendered = _rendered(w)
    assert "有未完成的任务" in rendered
    assert "是否需要继续执行" in rendered
    # 无任务行(只有标题 + affordance 两行)
    assert "•" not in rendered
