# src/birdcode/tools/grep_tool.py
"""Grep 工具:用 ripgrep(rg)按正则搜文件内容,按 output_mode 解析输出。"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from birdcode.tools.base import Tool


def _which_rg() -> str | None:
    """可被测试 monkeypatch 的 rg 探测入口。"""
    return shutil.which("rg")


class _GrepInput(BaseModel):
    pattern: str = Field(..., description="正则表达式(必填)。")
    glob: str | None = Field(default=None, description="文件过滤 glob,如 '*.py'。")
    path: str | None = Field(default=None, description="搜索根目录,默认 cwd。")
    output_mode: Literal["content", "files_with_matches", "count"] = Field(
        default="files_with_matches",
        description="content=匹配行;files_with_matches=命中文件列表;count=每文件命中次数。",
    )
    line_number: bool = Field(default=False, description="content 模式下带行号。")
    ignore_case: bool = Field(default=False, description="忽略大小写。")
    head_limit: int = Field(default=250, description="输出行数上限(截断)。")


class GrepTool(Tool):
    name = "grep"
    description = (
        "使用 ripgrep 按正则表达式在文件内容中快速搜索。\n"
        "注意:\n"
        "1. 用于在代码库中查找函数名、变量、字符串或代码模式;默认返回命中文件列表,"
        "设 output_mode=content 可看匹配行。\n"
        "2. 若只需查找文件路径而不看内容,请用 glob。\n"
        "3. 默认遵循 .gitignore;可用 path 指定搜索目录、glob 过滤文件类型(如 *.py)。"
    )
    parameters = _GrepInput
    kind: Literal["read"] = "read"
    parallel_safe = True
    max_result_chars = 20_000  # 仿 CC GrepTool:20K 字符落盘(命中多时最易爆)

    def __init__(self, *, cwd: str | None = None) -> None:
        # per-agent cwd。worktree 子 agent 经 fork_for_child 注入 worktree dir;
        # 主 agent(注册表单例)cwd=None → os.getcwd()(进程 cwd,Phase 1 已 chdir 到 worktree/主仓)。
        self._cwd: str = cwd or os.getcwd()

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """worktree 子 agent:cwd=worktree dir → 新实例 _cwd=worktree;否则继承父 _cwd。"""
        return type(self)(cwd=cwd or self._cwd)

    async def execute(  # type: ignore[override]
        self,
        *,
        pattern: str,
        glob: str | None = None,
        path: str | None = None,
        output_mode: str = "files_with_matches",
        line_number: bool = False,
        ignore_case: bool = False,
        head_limit: int = 250,
    ) -> str:
        if _which_rg() is None:
            return (
                "<error>ripgrep 未安装</error>"
                "<hint>用 read_file 确认 PATH,或安装 rg(工业级代码搜索标准)。</hint>"
            )
        # 显式相对 path 相对 _cwd 解析(与其它 5 个文件工具口径一致),传 rg 绝对路径。
        # worktree 子 agent:_cwd=worktree;若不解析,root 相对进程 cwd=主仓,口径分叉
        # (今日靠 _run_rg cwd=_cwd 传播等价,但改 _run_gl cwd 即漏)。
        _root = Path(path) if path else Path(self._cwd)
        if not _root.is_absolute():
            _root = Path(self._cwd) / _root
        root = str(_root)
        args: list[str] = ["rg", "--no-heading", "--color", "never"]
        if ignore_case:
            args.append("--ignore-case")
        if line_number:
            args.append("--line-number")
        if glob:
            args += ["--glob", glob]
        # output_mode → rg flag
        mode_flags = {
            "files_with_matches": ["--files-with-matches"],
            "count": ["--count"],
            "content": [],
        }
        args += mode_flags[output_mode]
        args += [pattern, root]
        code, out, err = await self._run_rg(args, cwd=self._cwd)
        if code == 1 or (code == 0 and not out.strip()):
            # rg 退出码 1 = 无匹配;0 且空也视为无匹配
            return (
                "<error>无匹配</error><hint>放宽 pattern、去掉 glob 限制、或加 ignore_case。</hint>"
            )
        if code not in (0, 1):
            # rg 自身报错(如路径不存在、正则非法)
            raw = (err or out).strip()
            tail = raw.splitlines()[-1] if raw else f"exit {code}"
            return (
                f"<error>rg 执行失败(exit {code}): {tail}</error>"
                "<hint>检查正则合法性、path 是否存在;用 glob 确认目录。</hint>"
            )
        lines = out.splitlines()
        if len(lines) > head_limit:
            kept = lines[:head_limit]
            return (
                "\n".join(kept) + f"\n... [已截断 — 共 {len(lines)} 行,显示前 {head_limit};"
                "用更精确的 pattern 或 glob 缩小范围] ..."
            )
        return "\n".join(lines).rstrip()

    @staticmethod
    async def _run_rg(args: list[str], *, cwd: str) -> tuple[int, str, str]:
        """执行 rg 子进程,cwd 指定(Phase 2:worktree 子 agent 用 worktree dir)。可 monkeypatch。"""
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        code = proc.returncode if proc.returncode is not None else 0
        return (
            code,
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
        )
