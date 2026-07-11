# src/birdcode/ui/widgets/completion_menu.py
"""命令补全浮层:输入 "/" 时挂在 #composer 末尾(StatusBar 下方,输入框上移让位),
↑↓ 移动、Tab 填名/Enter 执行选中、Esc 关闭。

键位不在这里管——InputArea._on_key 在浮层打开时拦截 up/down/enter/tab,
Esc 由 BirdApp.action_interrupt 收口(打开时关浮层、否则中断)。本 widget 只管
渲染候选 + 高亮选中项 + move/selected_name/update 供外部调用;候选按 name 升序
排列,多于窗口时用滑动窗口(选中项恒在可视区),命令再多也不会"选中却看不见"。
"""

from __future__ import annotations

from textual.widgets import Static

from birdcode.ui.commands.command import Command


class CompletionMenu(Static):
    DEFAULT_CSS = """
    CompletionMenu {
        border: tall $primary;
        background: $surface;
        color: $text;
        padding: 0 1;
        width: auto;
        height: auto;
        max-height: 16;
    }
    """

    _WINDOW = 8  # 最多同时显示的候选行数;超出走滑动窗口(选中项恒可见,无省略号)

    def __init__(self, candidates: list[Command], *, window: int | None = None) -> None:
        super().__init__()
        self._candidates = sorted(candidates, key=lambda c: c.name)
        self._sel = 0
        self._window = window if window is not None else self._WINDOW

    def update(self, candidates: list[Command]) -> None:
        """更新候选列表(输入实时过滤时复用同一浮层,避免重挂闪烁);按 name 排序、重置选中到首项。"""
        self._candidates = sorted(candidates, key=lambda c: c.name)
        self._sel = 0
        self.refresh()

    @property
    def candidates(self) -> list[Command]:
        return self._candidates

    def render(self) -> str:  # type: ignore[override]
        n = len(self._candidates)
        if n == 0:
            return ""
        win = min(self._window, n)
        # 滑动窗口:让 sel 尽量居中、到头尾贴边 → 选中项永远落在可视区内。
        # 不显示 … 省略号(用户偏好干净列表);超出窗口的项靠 ↑↓ 滑动查看。
        start = max(0, min(self._sel - win // 2, n - win))
        end = min(n, start + win)
        lines: list[str] = []
        for i in range(start, end):
            marker = "▶ " if i == self._sel else "  "
            c = self._candidates[i]
            lines.append(f"{marker}{c.name}  {c.description}")
        return "\n".join(lines)

    def move(self, delta: int) -> None:
        n = len(self._candidates)
        if n == 0:
            return
        self._sel = (self._sel + delta) % n
        self.refresh()

    def selected_name(self) -> str:
        return self._candidates[self._sel].name
