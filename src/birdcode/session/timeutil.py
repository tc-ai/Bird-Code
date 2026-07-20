# src/birdcode/session/timeutil.py
"""会话时间戳工具(共享:store + subagent_store)。

从 store.py 提升(原 _utc_iso),去下划线——它是纯时间戳工具,与 SessionStore 无耦合,
供 subagent_store 跨模块复用,避免 import 模块私有符号。
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_iso() -> str:
    """ISO-8601 带 Z(毫秒精度,无 Date.now 依赖,可测)。"""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
