# src/birdcode/permission/sandbox.py
"""L2 路径沙箱:resolve()(解析符号链接+..)后判断是否落在允许根下。硬约束。

roots 与被检路径都先 resolve,防 /proj/link→/etc 逃逸。Windows 路径大小写由
pathlib 自身处理(PureWindowsPath 的 relative_to 不区分大小写)。
"""

from __future__ import annotations

from pathlib import Path

from birdcode.utils.logging import get_logger

log = get_logger("birdcode.permission.sandbox")  # 进 debug.log


def resolve_roots(roots: list[Path]) -> list[Path]:
    """resolve 所有根(项目根本身若是符号链接也要 resolve);去重。"""
    out: list[Path] = []
    for r in roots:
        try:
            rr = r.resolve()
        except (OSError, ValueError):
            log.warning("沙箱根无法 resolve,已忽略: %s", r)
            continue
        if rr not in out:
            out.append(rr)
    return out


def check_path(file_path: str, roots: list[Path]) -> str | None:
    """返回 None=允许;返回 str=越界原因(供 Verdict.reason)。

    roots 必须已 resolve(用 resolve_roots 预处理)。
    """
    try:
        real = Path(file_path).resolve()
    except (OSError, ValueError):
        return f"<error>路径无法解析: {file_path}</error>"
    for root in roots:
        try:
            real.relative_to(root)
            return None
        except ValueError:
            continue
    return f"<error>路径越界: {file_path} 不在允许目录内</error>"
