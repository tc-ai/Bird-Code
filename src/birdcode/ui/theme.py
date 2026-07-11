# src/birdcode/ui/theme.py
"""Color palette + terminal capability detection (NO_COLOR, Unicode)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from rich.theme import Theme


@dataclass(frozen=True)
class AppTheme:
    name: str
    rich_theme: Theme
    design: dict[str, str] = field(default_factory=dict)


_DARK_DESIGN = {
    "background": "#1a1b26",
    "surface": "#1a1b26",
    "primary": "#7aa2f7",
    "secondary": "#6b7080",
    "success": "#9ece6a",
    "error": "#f7768e",
    "accent": "#e0af68",
    "frame": "#565f89",
    "text": "#c0caf5",
}

_DARK_RICH = Theme(
    {
        "markdown.code": "#73daca on #1a1b26",
        "markdown.item.bullet": "#7aa2f7",
        "info": "#7aa2f7",
        "success": "#9ece6a",
        "error": "#f7768e",
    }
)

_LIGHT_DESIGN = {
    "background": "#f6f8fa",
    "surface": "#f6f8fa",
    "primary": "#2e5cd6",
    "secondary": "#707088",
    "success": "#1a7f37",
    "error": "#cf222e",
    "accent": "#9a6700",
    "frame": "#d0d7de",
    "text": "#1f2328",
}

_LIGHT_RICH = Theme(
    {
        "markdown.code": "#0a7b83 on #f0f0f0",
        "markdown.item.bullet": "#2e5cd6",
        "info": "#2e5cd6",
        "success": "#1a7f37",
        "error": "#cf222e",
    }
)

_THEMES = {
    "dark": AppTheme("dark", _DARK_RICH, _DARK_DESIGN),
    "light": AppTheme("light", _LIGHT_RICH, _LIGHT_DESIGN),
}


def supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or ""
    return "utf" in enc.lower()


def get_theme(name: str) -> AppTheme:
    return _THEMES.get(name, _THEMES["dark"])
