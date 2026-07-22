# src/birdcode/ui/widgets/subagent_card.py
"""同步子 agent 进度卡(非流式;仅 name+时长+tokens)。"""

from __future__ import annotations

from textual.widgets import Static

from birdcode.agents.report import SubagentProgress


class SubagentCard(Static):
    """✻ <description> … (<duration> · ↓ <context_tokens>/<window>)。

    ↓ = 最近一轮 full input(当前上下文占用),分母为窗口上限。取最新一轮、非累计——
    累计会随轮次堆叠前缀超过窗口,造成「超限」误导。时长/计数由 SubagentProgress 更新
    (Done + 每秒 tick)。
    """

    DEFAULT_CSS = """
    SubagentCard {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._progress: SubagentProgress | None = None

    @staticmethod
    def _format_duration(ms: int) -> str:
        s = ms // 1000
        if s < 60:
            return f"{s}s"
        return f"{s // 60}m {s % 60}s"

    @staticmethod
    def _format_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)

    def update_progress(self, progress: SubagentProgress) -> None:
        self._progress = progress
        self.update(self._markup())

    def _markup(self) -> str:
        if self._progress is None:
            return "✻ 子 agent 运行中…"
        p = self._progress
        return (
            f"✻ {p.description} … ({self._format_duration(p.elapsed_ms)} · "
            f"↓ {self._format_tokens(p.context_tokens)}/{self._format_tokens(p.context_window)})"
        )

    def render(self) -> str:
        # Textual Static.render 返回 RenderableType(str 是其合法子类型),无需 ignore。
        # 测试直接读 _markup;运行时 Textual 渲染管线接收 str。
        return self._markup()
