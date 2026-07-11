# src/birdcode/ui/widgets/choice_prompt.py
"""方案选择行内菜单(模型 ask_user 给若干方案,用户选预设 / 原地输入自定义 / Esc 取消)。

Vertical 容器:question + N 预设 + "Type something." 合成一个 Static(#cp-menu,sel 切换刷新 ▶);
末项选中 → Type something 行隐藏 + #choice_input(InputArea) 显示 + focus,在菜单里原地输入。
InputArea 提交在容器内 @on+event.stop 捕获(不冒泡 app 主输入)。挂载后须 focus() 消费 BINDINGS。
"""
from __future__ import annotations

import unicodedata
from collections.abc import Callable

from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static

from birdcode.ui.widgets.input_area import InputArea


def _cjk_width(ch: str) -> int:
    """CJK/全角字符显示占 2 列,其余 1 列(east_asian_width W/F 判定)。"""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _wrap_cjk(text: str, width: int) -> list[str]:
    """按【显示宽度】折行(CJK 算 2 列),保留原换行。

    Static/Rich 的 word-wrap 对纯中文长串不按字符断词 → 超终端宽被截断。
    在 _menu_text 里预先按显示宽折行,每行不超 width 显示列。
    """
    if width <= 0:
        return text.splitlines() or [text]
    out: list[str] = []
    for raw in text.splitlines() or [""]:
        cur, cur_w = "", 0
        for ch in raw:
            cw = _cjk_width(ch)
            if cur and cur_w + cw > width:
                out.append(cur)
                cur, cur_w = ch, cw
            else:
                cur += ch
                cur_w += cw
        out.append(cur)
    return out


class ChoicePrompt(Vertical):
    """对话区方案选择菜单,inline 自定义输入(菜单保留、第 N+1 项原位变 InputArea)。"""

    can_focus = True  # 容器聚焦才消费自带 BINDINGS(↑↓/数字/enter)

    DEFAULT_CSS = """
    ChoicePrompt {
        border: tall $accent;
        background: $surface;
        color: $text;
        padding: 0 1;
        width: 100%;
        height: auto;
    }
    """

    BINDINGS = [
        # up 用 priority:输入态(InputArea 聚焦)时 ↑ 也能拦截回菜单,优先于 TextArea cursor_up。
        Binding("up", "move_up", show=False, priority=True),
        Binding("down", "move_down", show=False),
        Binding("enter", "confirm", show=False),
        *(Binding(str(n), f"select({n})", show=False) for n in range(1, 10)),
    ]

    def __init__(
        self, question: str, options: list[dict], *, on_result: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._question = question
        # 预设 (value, label, description);value=label。sel 取值 0..N,N=末项输入位。
        self._opts: list[tuple[str, str, str]] = [
            (str(o.get("label", "")), str(o.get("label", "")), str(o.get("description", "")))
            for o in options
        ]
        self._n = len(self._opts)
        self._on_result = on_result
        self._sel = 0
        self._done = False

    def compose(self):  # type: ignore[override]
        yield Static(self._menu_text(), id="cp-menu")
        yield InputArea(id="choice_input")  # 初始 display=False(on_mount 设),别用 id="input"

    def on_mount(self) -> None:
        # InputArea 初始隐藏(sel 到末项才显示 + focus)。
        try:
            self.query_one("#choice_input", InputArea).display = False
        except Exception:  # noqa: BLE001
            pass

    def _menu_text(self) -> str:
        width = self.size.width if self.size.width > 0 else 80
        inner = max(20, width - 10)  # 减 border(2)+padding(2)+缩进(6)
        lines = [self._question]
        for i, (_v, label, desc) in enumerate(self._opts):
            marker = "▶ " if i == self._sel else "  "
            lines.append(f"{marker}{i + 1}. {label}")
            if desc:
                for dl in _wrap_cjk(desc, inner - 6):
                    lines.append(f"      {dl}")
        if self._sel != self._n:  # 没在输入项 → 显示 Type something(非高亮)
            lines.append(f"  {self._n + 1}. Type something.")
        return "\n".join(lines)

    def _refresh_menu(self) -> None:
        try:
            self.query_one("#cp-menu", Static).update(self._menu_text())
        except Exception:  # noqa: BLE001
            pass

    def action_move_down(self) -> None:
        if self._done:
            return
        self._sel = (self._sel + 1) % (self._n + 1)
        if self._sel == self._n:
            self._enter_input()
        else:
            self._refresh_menu()

    def action_move_up(self) -> None:
        if self._done:
            return
        if self._sel == self._n:  # 输入态 → 回最后一项预设
            self._exit_input()
            return
        self._sel = (self._sel - 1) % (self._n + 1)
        if self._sel == self._n:  # 从 0 ↑ 回绕到末项(罕见)
            self._enter_input()
        else:
            self._refresh_menu()

    def _exit_input(self) -> None:
        """输入态 → 回菜单:隐藏 InputArea + sel 回最后一项预设 + 容器 focus。"""
        self._sel = self._n - 1
        try:
            self.query_one("#choice_input", InputArea).display = False
        except Exception:  # noqa: BLE001
            pass
        self._refresh_menu()
        self.focus()

    def _enter_input(self) -> None:
        """sel 到末项 → Type something 行隐藏(menu 刷新去掉该行)+ InputArea 显示 + focus。"""
        self._refresh_menu()
        try:
            ia = self.query_one("#choice_input", InputArea)
            ia.display = True
            ia.focus()
        except Exception:  # noqa: BLE001
            pass

    def action_confirm(self) -> None:
        """enter = 选当前预设(sel<N)。sel==N(输入态)的 enter 由 InputArea Submitted 处理。"""
        if self._done or self._sel >= self._n:
            return
        self.choose(self._opts[self._sel][0])

    def action_select(self, n: int) -> None:
        """数字键 1..N = 直选预设;N+1 = 进输入态。"""
        if self._done:
            return
        idx = int(n)
        if 1 <= idx <= self._n:
            self._sel = idx - 1
            self.choose(self._opts[self._sel][0])
        elif idx == self._n + 1:
            self._sel = self._n
            self._enter_input()

    @on(InputArea.Submitted)
    def _on_inner_submitted(self, event: InputArea.Submitted) -> None:
        """容器内 InputArea 提交 → choose(文本);event.stop 阻断冒泡到 app 主输入。"""
        event.stop()
        text = event.text.strip()
        if text and not self._done:
            self.choose(text)

    def choose(self, value: str) -> None:
        """提交结果(预设 label / 输入文本 / 取消哨兵)。幂等单出口。"""
        if self._done:
            return
        self._done = True
        self._on_result(value)
