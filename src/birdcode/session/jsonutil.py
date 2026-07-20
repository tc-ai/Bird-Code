# src/birdcode/session/jsonutil.py
"""JSON 文件读写共享工具:读 .json → dict | None(缺/坏统一处理)+ 原子写文本。

读(read_json_dict):抽离自 store._read_meta 与 subagent_meta.read_subagent_meta 的
重复读块,集中后未来再加错误类/编码守卫只需改一处,杜绝两个读取器对同一坏字节一个
返 None 一个崩的漂移。
写(write_json_atomic):temp + os.replace 原子写,杜绝覆盖写中途崩溃留半截文件
(如 meta 半截 → read 返 None → 子 agent 永久悬挂)。
"""

from __future__ import annotations

import contextlib
import json
import os
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


def write_json_atomic(path: Path, text: str) -> None:
    """原子写文本到 path(temp + os.replace)。

    先写同目录 tmp(<name>.tmp),成功后 os.replace 原子替换(同文件系统 rename 原子,
    Windows 走 MoveFileExW)。tmp 写/replace 失败时 target 原文件不动(杜绝半截 JSON);
    OSError 向上传播由调用方兜底。finally 清残留 tmp(成功路径 os.replace 已移走 tmp,
    unlink 抑制 FileNotFoundError;失败路径清半截 tmp,避免目录堆积)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
