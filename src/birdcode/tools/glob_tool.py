# src/birdcode/tools/glob_tool.py
"""Glob 工具:按 glob 模式查找文件路径,按 mtime 倒序,限 100 条。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from birdcode.tools.base import Tool

_GLOB_LIMIT = 100
_WORKTREE_SKIP = (".birdcode", "worktrees")


def _is_under_worktrees(p: Path, root: Path) -> bool:
    """p 是否在 root/.birdcode/worktrees/ 下(任意深度)。"""
    try:
        rel = p.relative_to(root)
    except ValueError:
        return False
    parts = rel.parts
    return any(
        parts[i] == _WORKTREE_SKIP[0] and i + 1 < len(parts) and parts[i + 1] == _WORKTREE_SKIP[1]
        for i in range(len(parts) - 1)
    )


class _GlobInput(BaseModel):
    pattern: str = Field(..., description="glob 模式,如 '*.py' 或 '**/*.py'(必填)。")
    path: str | None = Field(default=None, description="搜索根目录,默认当前工作目录(cwd)。")


class GlobTool(Tool):
    name = "glob"
    description = (
        "按 glob 模式(如 **/*.ts, src/**/*.js)快速查找并列出文件路径。\n"
        "注意:\n"
        "1. 仅返回文件路径,不读取内容;如需内容请结合 read_file 或 grep。\n"
        "2. 适用于了解项目目录结构、查找特定后缀/命名的文件、定位未知路径。"
    )
    parameters = _GlobInput
    kind: Literal["read"] = "read"
    parallel_safe = True
    max_result_chars = 30_000  # BirdCode 选 30K(CC GlobTool=100K):更早落盘护 context

    def __init__(self, *, cwd: str | None = None) -> None:
        # per-agent cwd。worktree 子 agent 经 fork_for_child 注入 worktree dir;
        # 主 agent(注册表单例)cwd=None → os.getcwd()(进程 cwd,Phase 1 已 chdir 到 worktree/主仓)。
        self._cwd: str = cwd or os.getcwd()

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """worktree 子 agent:cwd=worktree dir → 新实例 _cwd=worktree;否则继承父 _cwd。"""
        return type(self)(cwd=cwd or self._cwd)

    async def execute(self, *, pattern: str, path: str | None = None) -> str:  # type: ignore[override]
        root = Path(path) if path else Path(self._cwd)
        # 显式相对 path(如 "src")必须相对 _cwd 解析——与其它 5 个文件工具口径一致
        # (worktree 子 agent:root.glob() 否则会相对进程 cwd=主仓,串进主仓目录)。
        if not root.is_absolute():
            root = Path(self._cwd) / root
        if not root.exists():
            return (
                f"<error>path 不存在: {root}</error>"
                "<hint>检查相对/绝对路径,或先用 glob(pattern='**') 列出根下文件。</hint>"
            )
        try:
            all_matched = list(root.glob(pattern))
        except OSError:
            all_matched = []
        # 跳过 .birdcode/worktrees/(Path.glob 不认 gitignore,会串进兄弟 worktree)
        matched = sorted(
            (p for p in all_matched if not _is_under_worktrees(p, root)),
            key=_safe_mtime,
            reverse=True,
        )
        if not matched:
            return "<error>无匹配</error><hint>放宽 pattern(如 *.py → **/*.py)或确认 path。</hint>"
        capped = matched[:_GLOB_LIMIT]
        # 相对 path 给出更短可读;path 为 None 时相对 cwd
        base = root
        lines = [str(p.relative_to(base)) if _is_relative(p, base) else str(p) for p in capped]
        body = "\n".join(lines)
        if len(matched) > _GLOB_LIMIT:
            body += (
                f"\n... [已截断 — 共 {len(matched)} 条,限 {_GLOB_LIMIT};"
                "用更具体的 pattern 缩小范围] ..."
            )
        return body


def _is_relative(p: Path, base: Path) -> bool:
    try:
        p.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_mtime(p: Path) -> float:
    """glob 与 stat 之间文件可能被删(竞态)——单文件 stat 失败给最小 mtime,
    使其排末尾而非让整个 sorted 抛 OSError 抹掉全部命中。"""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0
