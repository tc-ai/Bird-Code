# src/birdcode/ui/widgets/thinking_block.py
"""可折叠 ✻ Thinking… 流式块（仿 Claude Code）。自含 spinner；默认折叠。"""

from __future__ import annotations

import time

from textual import events
from textual.timer import Timer
from textual.widgets import Static

from birdcode.ui.icons import Icons


def format_thinking_running(frame: str, seconds: int) -> str:
    return f"{frame} Thinking… ({seconds}s)"


def format_thinking_done(icons: Icons, *, seconds: int, round_no: int = 0) -> str:
    # 多轮 agent_loop 中每轮一个 ThinkingBlock;第 2 轮起标注「第 K 轮」(第 1 轮不标,避免噪音)。
    # 形参用 round_no 而非 round,避免遮蔽内置 round()。
    suffix = f"(第 {round_no} 轮)" if round_no > 1 else ""
    return f"{icons.thinking} Thought for {seconds}s {suffix}".rstrip()


class ThinkingBlock(Static):
    """reasoning 流式展示。默认折叠（仅标题行）；展开看完整文本。"""

    DEFAULT_CSS = "ThinkingBlock { color: $secondary; }"

    def __init__(self, icons: Icons, *, round_no: int = 0) -> None:
        # markup=False:思考正文(模型 reasoning)可能含 [..](如代码片段 [TextBlock]),
        # 默认 markup 解析会 MarkupError 崩溃退出整个 TUI。
        super().__init__(markup=False)
        self._icons = icons
        self._round = round_no
        self._text: list[str] = []
        self._expanded = False
        self._seconds = 0
        self._frame = 0
        self._started: float = 0.0
        self._timer: Timer | None = None

    def start(self) -> None:
        self._started = time.monotonic()
        self._seconds = 0
        self._frame = 0
        self._refresh()
        self._timer = self.set_interval(0.12, self._tick)

    def write(self, text: str) -> None:
        self._text.append(text)
        self._refresh()

    def finish(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._refresh()

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._refresh()

    def on_click(self, event: events.Click) -> None:
        """点击 head 行(第 0 行)展开/收起;展开后点正文(y≥1)不收起——便于选中拷贝。

        否则展开后想拖选正文里的某几行,鼠标抬起就被当成 click 触发收起,无法选中。
        """
        if self._expanded and event.y >= 1:
            return  # 点正文:不 toggle,留给选中
        event.stop()
        self.toggle()

    def _tick(self) -> None:
        self._seconds = int(time.monotonic() - self._started)
        self._frame = (self._frame + 1) % len(self._icons.spinner)
        self._refresh()

    def _refresh(self) -> None:
        running = self._timer is not None
        if running:
            head = format_thinking_running(self._icons.spinner[self._frame], self._seconds)
        else:
            head = format_thinking_done(self._icons, seconds=self._seconds, round_no=self._round)
        hint = "点击收起" if self._expanded else "点击展开"
        line = f"{head} · {hint}"
        if self._expanded:
            body = "".join(self._text).rstrip()
            self.update(f"{line}\n{body}")
        else:
            self.update(line)
