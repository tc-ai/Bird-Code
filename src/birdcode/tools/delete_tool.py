# src/birdcode/tools/delete_tool.py
"""Delete 工具:删除单个文件。kind=write, parallel_safe=False。

文件 CRUD 之"D"(write_file 创建/覆盖、edit_file 改、delete_file 删)。让"删除"也成为
可被 agent_loop `last_partial` 精确追踪的原子 tool_use——避免用 bash rm 批量删:
ESC 杀进程组杀在 `rm a; rm b; rm c` 中间,已删文件不回滚(半完成);且 bash rm 非幂等,
中断后被当"未执行"盲目重试会误删更多。逐个 delete_file 则每个都是独立 tool_use,中断时
`last_partial` 精确记录"删了哪些、没删哪些",模型下轮只补做未删的。
删目录不在本工具范围(用 bash rm -r)。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel, Field

from birdcode.tools.base import Tool
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.tools.delete_tool")  # 进 debug.log


class _DeleteInput(BaseModel):
    file_path: str = Field(..., description="要删除的文件路径(必填)。")


class DeleteTool(Tool):
    name = "delete_file"
    description = (
        "删除单个文件。\n"
        "注意:\n"
        "1. 仅删文件,不删目录(删目录用 bash rm -r)。\n"
        "2. 删除不可逆,执行前确认工作区有 git 或其它备份可恢复。\n"
        "3. 删多个文件请逐个调用本工具,不要用 bash 批量删(便于中断恢复、避免半完成)。"
    )
    parameters = _DeleteInput
    kind = "write"
    parallel_safe = False

    def __init__(self, *, cwd: str | None = None) -> None:
        # per-agent cwd。worktree 子 agent 经 fork_for_child 注入 worktree dir;
        # 主 agent(注册表单例)cwd=None → os.getcwd()(进程 cwd,Phase 1 已 chdir 到 worktree/主仓)。
        self._cwd: str = cwd or os.getcwd()

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """worktree 子 agent:cwd=worktree dir → 新实例 _cwd=worktree;否则继承父 _cwd。"""
        return type(self)(cwd=cwd or self._cwd)

    async def execute(self, *, file_path: str) -> str:  # type: ignore[override]
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self._cwd) / path
        if not path.exists():
            return (
                f"<error>文件不存在: {file_path}</error>"
                f"<hint>用 glob 或 ls 确认路径、检查拼写。</hint>"
            )
        if path.is_dir():
            return (
                f"<error>{file_path} 是目录,delete_file 仅删文件。</error>"
                f'<hint>删目录用 bash: rm -r "{file_path}"。</hint>'
            )
        try:
            await asyncio.to_thread(_unlink_sync, path)
        except PermissionError:
            return (
                f"<error>删除被拒绝(权限不足): {file_path}</error>"
                f"<hint>检查文件权限或是否被占用。</hint>"
            )
        except OSError as exc:
            log.exception("delete failed: %s", file_path)
            return f"<error>删除失败: {exc}</error><hint>检查路径/权限/是否被占用。</hint>"
        return f"已删除 {file_path}。"


def _unlink_sync(path: Path) -> None:
    path.unlink()
