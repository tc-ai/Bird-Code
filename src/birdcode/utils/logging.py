# src/birdcode/utils/logging.py
"""File-only logging for BirdCode. Never writes to stdout/stderr."""

from __future__ import annotations

import logging
from pathlib import Path

_log_dir: Path = Path.home() / ".birdcode"


def _ensure_handlers(logger: logging.Logger, log_path: Path) -> None:
    if any(getattr(h, "_birdcode", False) for h in logger.handlers):
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler._birdcode = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def get_logger(name: str = "birdcode") -> logging.Logger:
    logger = logging.getLogger(name)
    _ensure_handlers(logger, _log_dir / "debug.log")
    return logger
