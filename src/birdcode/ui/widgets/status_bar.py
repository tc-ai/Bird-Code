# src/birdcode/ui/widgets/status_bar.py
"""状态行：纯格式化 + Static widget。"""

from __future__ import annotations

from textual.widgets import Static

from birdcode.agent.provider import TokenUsage


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        k = round(n / 1_000, 1)
        if k >= 1000.0:  # 999_950+ rounds up to 1000.0k → 显示为 M
            return f"{k / 1000:.1f}M"
        return f"{k:.1f}k"
    return str(n)


def format_status(
    *,
    model: str,
    mode: str,
    cwd: str,
    usage: TokenUsage | None,
    queue_size: int,
    busy: bool,
    unicode_supported: bool,
) -> str:
    sep = " · " if unicode_supported else " | "
    queued = "⌛" if unicode_supported else "#"
    up, down = ("↑", "↓") if unicode_supported else ("^", "v")
    seg = ["birdcode", model, mode, cwd]
    tail: list[str] = []
    if usage is not None and (usage.input_tokens or usage.output_tokens):
        io = f"{up}{_fmt_tokens(usage.input_tokens)} {down}{_fmt_tokens(usage.output_tokens)}"
        cache_bits: list[str] = []
        if usage.cache_read_tokens:
            cache_bits.append(f"cache{up}{_fmt_tokens(usage.cache_read_tokens)}")
        if usage.cache_creation_tokens:
            cache_bits.append(f"new{up}{_fmt_tokens(usage.cache_creation_tokens)}")
        cache_seg = (" " + " ".join(cache_bits)) if cache_bits else ""
        tail.append(f"{io}{cache_seg} ${usage.cost_usd:.4f}")
    if queue_size:
        tail.append(f"{queued}{queue_size}")
    if busy:
        tail.append("working")
    return sep.join(seg + tail)


class StatusBar(Static):
    DEFAULT_CSS = "StatusBar { color: $secondary; }"

    def refresh_from(
        self,
        *,
        model: str,
        mode: str,
        cwd: str,
        usage: TokenUsage | None,
        queue_size: int,
        busy: bool,
        unicode_supported: bool,
    ) -> None:
        self.update(
            format_status(
                model=model,
                mode=mode,
                cwd=cwd,
                usage=usage,
                queue_size=queue_size,
                busy=busy,
                unicode_supported=unicode_supported,
            )
        )
