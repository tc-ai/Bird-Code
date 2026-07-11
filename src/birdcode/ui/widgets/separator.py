# src/birdcode/ui/widgets/separator.py
"""满宽分隔线（resize 安全）。"""

from __future__ import annotations

from textual.strip import (  # type: ignore[attr-defined]  # Segment lives in textual.strip (no __all__)
    Segment,
    Strip,
)
from textual.widget import Widget


def separator_segments(width: int, char: str) -> list[Segment]:
    return [Segment(char * width)] if width > 0 else [Segment("")]


class Separator(Widget):
    DEFAULT_CSS = "Separator { height: 1; color: $secondary; }"

    def __init__(self, char: str = "─") -> None:
        super().__init__()
        self.char = char

    def render_line(self, y: int) -> Strip:  # height=1，y 必为 0
        return Strip(separator_segments(self.size.width, self.char))
