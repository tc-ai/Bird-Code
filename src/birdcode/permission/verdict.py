# src/birdcode/permission/verdict.py
"""权限判定结果。Verdict 携带 action / reason / layer,供 executor 把 reason 回传模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["allow", "deny"]


@dataclass
class Verdict:
    """一次权限判定的结果。

    action: allow/deny。
    reason: 进 ToolResult.error_message 回传模型(L1 hint / "plan 模式拒绝" / 越界原因)。
    layer:  哪层判的("L1".."L5"/"read-default"),便于日志/调试。
    """

    action: Decision
    reason: str = ""
    layer: str = ""
