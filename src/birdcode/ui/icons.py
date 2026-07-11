# src/birdcode/ui/icons.py
"""UI glyphs with ASCII fallbacks for terminals without Unicode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Icons:
    tool: str
    result: str
    queued: str
    success: str
    fail: str
    spinner: tuple[str, ...]
    thinking: str


_UNICODE = Icons(
    tool="●",
    result="⎿",
    queued="⌛",
    success="✓",
    fail="✗",
    spinner=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
    thinking="✻",
)

_ASCII = Icons(
    tool="*",
    result="'",
    queued="#",
    success="+",
    fail="x",
    spinner=("|", "/", "-", "\\"),
    thinking="*",
)


def select_icons(*, unicode_supported: bool) -> Icons:
    return _UNICODE if unicode_supported else _ASCII
