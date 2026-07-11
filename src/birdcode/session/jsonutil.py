# src/birdcode/session/jsonutil.py
"""JSON 文件读取共享工具:读 .json → dict | None,统一缺/坏处理。

抽离自 store._read_meta 与 subagent_meta.read_subagent_meta 的重复读块:
两者原本各自 `json.loads(read_text) + except (OSError, ValueError) + isinstance(dict)`
——修 UnicodeDecodeError 时被迫在两处并行扩 except,正是此重复的实证。集中后
未来再加错误类/编码守卫只需改一处,杜绝两个读取器对同一坏字节一个返 None 一个崩的漂移。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_dict(path: Path) -> dict[str, Any] | None:
    """读 JSON 文件 → dict;缺/坏(非 JSON / 非 UTF-8 / 非 dict)→ None。

    ValueError 覆盖 JSONDecodeError + UnicodeDecodeError(非 UTF-8 字节,如崩溃转储/
    外部编辑)。调用方据此 lazy backfill / 跳过坏文件,不杀主循环。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
