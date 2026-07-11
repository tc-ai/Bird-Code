# src/birdcode/permission/blacklist.py
"""L1 危险命令黑名单:正则硬拦截,bypass 不可放开。从 bash_tool.py 迁移,单一来源。

gate L1 与 BashTool 都从此调用——保证「同一份正则」。
"""

from __future__ import annotations

import re

_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-[a-zA-Z]*[rR][a-zA-Z]*\s+(?:/(?:\s|$)|~|\$HOME)", "rm -r 根/家目录"),
    (r"\brm\s+-[a-zA-Z]*[rR][a-zA-Z]*\s+\.(?:/\*|/\s*$|\*|\s*$)", "rm -r 当前目录/通配"),
    (r"\bfind\s+/\s.*-delete\b", "find / -delete"),
    (r"\bchmod\s+-R\s+0?000\s+/", "chmod -R 000 根"),
    (r"\bsudo\b", "sudo 提权"),
    (r">\s*/dev/(?:sd[a-z]+|nvme|disk)", "直接写块设备"),
    (r"\bmkfs\b", "mkfs 格式化"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;", "fork bomb"),
    (r"\bdd\b.*\bof\s*=\s*/dev/(?:sd|nvme|disk)", "dd 写块设备"),
    # 关机/重启:仅匹配命令动词位置(^ 或 shell 分隔符 ;|& 之后),不匹配参数/引号内的
    # 同名词——否则 git commit -m "fix reboot" / grep shutdown 等会被误拦。
    (r"(?:^|[;|&]\s*)(?:shutdown|reboot|poweroff|halt)\b", "关机/重启"),
]


def is_dangerous(command: str) -> tuple[bool, str]:
    """命中黑名单 → (True, 命中模式描述);否则 (False, "")。"""
    for pattern, label in _DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True, label
    return False, ""


def dangerous_hint(label: str) -> str:
    return (
        f"<error>危险命令模式拦截: {label}</error>"
        "<hint>改用更安全的方式(如分步、加确认);若确需该操作,"
        "拆解为非破坏性步骤(例如限定到具体子目录、先 dry-run、或手动逐条执行并核对)。</hint>"
    )
