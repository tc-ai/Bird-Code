# tests/ui/test_choice_prompt.py
"""ChoicePrompt(Vertical 容器)逻辑层:_menu_text 渲染、sel 切换、choose 幂等、内嵌 InputArea 提交。

query_one/display/focus 等 UI 行为需 mount(Textual run_test),留手动验证;此处测不依赖
mount 的逻辑(_menu_text 用 size 兜底 80、sel/choose/_on_inner_submitted 不 query_one 或静默)。
"""
from birdcode.ui.widgets.choice_prompt import ChoicePrompt


def _capture() -> tuple[list[str], object]:
    box: list[str] = []

    def cb(value: object) -> None:
        box.append(str(value))

    return box, cb


def _opts(*labels: str) -> list[dict]:
    return [{"label": lb, "description": f"方案 {lb}"} for lb in labels]


def _highlighted(p: ChoicePrompt) -> list[str]:
    return [line for line in p._menu_text().splitlines() if line.startswith("▶ ")]


def test_menu_text_question_opts_custom():
    p = ChoicePrompt("选哪个?", _opts("A", "B"), on_result=_capture()[1])
    txt = p._menu_text()
    assert "选哪个?" in txt
    assert "A" in txt and "方案 A" in txt
    assert "B" in txt
    assert "Type something." in txt  # sel<N 时显示


def test_default_sel_first_option():
    p = ChoicePrompt("q", _opts("A", "B"), on_result=_capture()[1])
    assert _highlighted(p) and "A" in _highlighted(p)[0]


def test_move_down_advances_sel():
    p = ChoicePrompt("q", _opts("A", "B"), on_result=_capture()[1])
    p.action_move_down()  # _refresh_menu 静默失败(无 mount),但 _sel 已变
    assert p._sel == 1
    assert any("B" in line for line in _highlighted(p))


def test_move_to_custom_hides_type_something():
    p = ChoicePrompt("q", _opts("A"), on_result=_capture()[1])  # N=1,末项=1
    p.action_move_down()  # sel 0→1(末项)
    assert p._sel == 1
    assert "Type something." not in p._menu_text()  # sel==N,menu 去掉 Type something 行


def test_select_preset_choose():
    box, cb = _capture()
    ChoicePrompt("q", _opts("A", "B"), on_result=cb).action_select(2)  # 数字键 2 → B
    assert box == ["B"]


def test_select_custom_enters_input():
    p = ChoicePrompt("q", _opts("A"), on_result=_capture()[1])  # N=1
    p.action_select(2)  # N+1=2 → 末项输入
    assert p._sel == 1


def test_choose_label():
    box, cb = _capture()
    ChoicePrompt("q", _opts("A"), on_result=cb).choose("A")
    assert box == ["A"]


def test_choose_idempotent():
    box, cb = _capture()
    p = ChoicePrompt("q", _opts("A"), on_result=cb)
    p.choose("A")
    p.choose("B")  # 已 done,忽略
    assert box == ["A"]


class _FakeSubmitted:
    """模拟 InputArea.Submitted(text + stop)。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def stop(self) -> None:  # noqa: D401
        pass


def test_inner_submitted_choose():
    box, cb = _capture()
    p = ChoicePrompt("q", _opts("A"), on_result=cb)
    p._on_inner_submitted(_FakeSubmitted("我的意见"))
    assert box == ["我的意见"]


def test_inner_submitted_empty_ignored():
    box, cb = _capture()
    p = ChoicePrompt("q", _opts("A"), on_result=cb)
    p._on_inner_submitted(_FakeSubmitted("   "))  # 空白 → 忽略
    assert box == []
