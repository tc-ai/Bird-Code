# src/birdcode/ui/widgets/file_completion_menu.py
"""@ 文件补全浮层:挂在 #composer 末尾,↑↓ 移动、Tab 补全/展开、Enter 插入、Esc 关闭。

与命令补全 CompletionMenu 同形(▶ 标记 + 滑动窗口,选中项恒可见),
但候选是 FileCandidate(目录带尾 /、排在前),并支持限流提示行。
键位仍由 InputArea._on_key 在浮层打开时拦截;本 widget 只管渲染 + 高亮 +
move/selected_candidate/update。
"""

from __future__ import annotations

from textual.widgets import Static

from birdcode.ui.pure import FileCandidate


class FileCompletionMenu(Static):
    DEFAULT_CSS = """
    FileCompletionMenu {
        background: transparent;
        color: #969dac;
        padding: 0 1;
        width: auto;
        height: auto;
        max-height: 16;
    }
    """

    _WINDOW = 8  # 最多同时显示的候选行数;超出走滑动窗口(选中项恒可见)

    def __init__(
        self,
        candidates: list[FileCandidate],
        truncated: bool = False,
        *,
        window: int | None = None,
    ) -> None:
        super().__init__()
        self._candidates = list(candidates)
        self._truncated = truncated
        self._window = window if window is not None else self._WINDOW
        self._sel = 0

    def update(self, candidates: list[FileCandidate], truncated: bool = False) -> None:
        """更新候选(实时过滤时复用同一浮层),重置选中到首项。"""
        self._candidates = list(candidates)
        self._truncated = truncated
        self._sel = 0
        self.refresh()

    @property
    def candidates(self) -> list[FileCandidate]:
        return self._candidates

    @property
    def truncated(self) -> bool:
        return self._truncated

    def render(self) -> str:  # type: ignore[override]
        n = len(self._candidates)
        if n == 0:
            return ""
        win = min(self._window, n)
        start = max(0, min(self._sel - win // 2, n - win))
        end = min(n, start + win)
        lines: list[str] = []
        if self._truncated:
            lines.append("…（仅显示前 100 项，继续输入以过滤）")
        for i in range(start, end):
            marker = "▶ " if i == self._sel else "  "
            lines.append(f"{marker}{self._candidates[i].name}")
        return "\n".join(lines)

    def move(self, delta: int) -> None:
        n = len(self._candidates)
        if n == 0:
            return
        self._sel = (self._sel + delta) % n
        self.refresh()

    def selected_candidate(self) -> FileCandidate | None:
        if not self._candidates:
            return None
        return self._candidates[self._sel]
