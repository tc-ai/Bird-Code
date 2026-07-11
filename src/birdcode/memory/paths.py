# src/birdcode/memory/paths.py
"""记忆目录解析(纯函数):type → 目录,user/project 两级。

- user/feedback → ~/.birdcode/memory/(跨项目复用)
- project/reference → <project_root>/.birdcode/memory/(项目特定)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

MemoryType = Literal["user", "feedback", "project", "reference"]

_USER_TYPES: frozenset[str] = frozenset({"user", "feedback"})
_PROJECT_TYPES: frozenset[str] = frozenset({"project", "reference"})


def user_memory_dir() -> Path:
    """用户级记忆目录:~/.birdcode/memory/(跨项目复用)。"""
    return Path.home() / ".birdcode" / "memory"


def project_memory_dir(project_root: Path) -> Path:
    """项目级记忆目录:<project_root>/.birdcode/memory/。"""
    return project_root / ".birdcode" / "memory"


def type_to_dir(type_: str, project_root: Path) -> Path:
    """type → 目录:user/feedback→用户级;project/reference→项目级。非法 type 抛 ValueError。"""
    if type_ in _USER_TYPES:
        return user_memory_dir()
    if type_ in _PROJECT_TYPES:
        return project_memory_dir(project_root)
    raise ValueError(f"非法记忆类型 {type_!r}(须 user/feedback/project/reference)")
