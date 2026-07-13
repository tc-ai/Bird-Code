# src/birdcode/ui/widgets/status_bar.py
"""状态行:彩色当前模式 + 快捷键提示(纯格式化 + Static widget)。

品牌/模型/路径移到顶端 banner;底端只留「彩色模式 + 操作提示」。模式颜色随 agent
权限递进:plan/default 绿(安全)→ accept-edits 橙(醒目)→ bypass 红(危险)。
"""

from __future__ import annotations

from textual.widgets import Static


def _fmt_tokens(n: int) -> str:
    """token 数 → 紧凑展示(1.0k / 1.2M)。保留供 /context 等处与测试复用。"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        k = round(n / 1_000, 1)
        if k >= 1000.0:  # 999_950+ rounds up to 1000.0k → 显示为 M
            return f"{k / 1000:.1f}M"
        return f"{k:.1f}k"
    return str(n)


# 模式 → (显示标签, 颜色)。用 hex(Tokyo Night 调色)而非命名色——命名色经终端调色板
# 映射可能偏色(如 dark_orange 被误映成蓝),hex 走 truecolor 最稳。绿=安全→橙=醒目→红=危险。
_MODE_DISPLAY: dict[str, tuple[str, str]] = {
    "plan": ("plan", "#9ece6a"),
    "default": ("default", "#9ece6a"),
    "accept-edits": ("accept edits", "#ff9e64"),
    "bypass": ("bypass", "#f7768e"),
}

# 底端固定的快捷键提示(与 banner 分离:banner 放品牌/模型/路径)。
_HINTS = "\\+Enter 换行，Esc 中断，Ctrl+Q 退出"
_HINTS_ASCII = "\\+Enter 换行 | Esc 中断 | Ctrl+Q 退出"


def _mode_display(mode: str) -> tuple[str, str]:
    """mode → (可读标签, 颜色)。未知 mode 退化为原值 + 绿。"""
    return _MODE_DISPLAY.get(mode, (mode, "#9ece6a"))


def _queued_segment(queue_size: int, unicode_supported: bool) -> str:
    """排队中的用户消息计数片段(queue_size>0 才显示)。type-ahead 时提示有消息在排队。"""
    if queue_size <= 0:
        return ""
    icon = "⌛" if unicode_supported else "#"
    return f" {icon}{queue_size}"


def format_status(*, mode: str, unicode_supported: bool, queue_size: int = 0) -> str:
    """彩色模式 + (shift+tab cycle) + [排队计数] + 快捷键。模式段 hex 着色,余继承 $secondary。"""
    label, color = _mode_display(mode)
    sep = " · " if unicode_supported else " | "
    hints = _HINTS if unicode_supported else _HINTS_ASCII
    queued = _queued_segment(queue_size, unicode_supported)
    return f"[{color}]{label}[/] (shift+tab cycle){queued}{sep}{hints}"


class StatusBar(Static):
    # 中性白(无蓝调):$secondary=#6b7080 蓝灰在黑底上偏蓝看不清;模式标签靠 markup 自带色覆盖。
    DEFAULT_CSS = "StatusBar { color: #e6e6e6; }"

    def refresh_from(
        self, *, mode: str, unicode_supported: bool, queue_size: int = 0
    ) -> None:
        self.update(
            format_status(
                mode=mode, unicode_supported=unicode_supported, queue_size=queue_size
            )
        )
