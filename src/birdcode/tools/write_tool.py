# src/birdcode/tools/write.py
"""Write 工具:完整覆盖写(非追加)。kind=write, parallel_safe=False。

用途=完整覆盖写入;何时用=创建新文件/整体重写;何时不用=改局部用 Edit;
参数=file_path/content;输出=成功 + 写入行数;写入类需用户确认。
stage4:成功写入后把 (old="", new=content) 写入模块级侧信道 _last_diff,
供 executor 读取塞进 ToolResult.diff(供 UI 挂 diff 高亮子区)。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel, Field

from birdcode.tools.base import Tool
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.tools.write_tool")  # 进 debug.log

_last_diff: tuple[str, str] | None = None  # 侧信道:最近一次成功写入的 (old, new)


def take_last_diff() -> tuple[str, str] | None:
    """executor 在 execute 成功后调用,取走并清空侧信道。"""
    global _last_diff
    d = _last_diff
    _last_diff = None
    return d


class _WriteInput(BaseModel):
    file_path: str = Field(..., description="要写入的文件路径(必填)。")
    content: str = Field(..., description="文件的完整新内容(必填)。")


class WriteTool(Tool):
    name = "write_file"
    description = (
        "将提供的内容完整覆盖写入指定文件。\n"
        "注意:\n"
        "1. 主要用于新建文件,或对现有文件整体重写。\n"
        "2. 若只是局部修改现有文件,请用 edit_file,以省 token 并避免破坏未修改内容。\n"
        "3. 父目录须已存在(不存在请先用 bash mkdir);覆盖现有文件前建议先 read_file 了解原内容。"
    )
    parameters = _WriteInput
    kind = "write"
    parallel_safe = False
    max_result_chars = 30_000  # 同 edit_file:30K(CC 未列;write 输出与 edit 同量级)

    def __init__(self, *, cwd: str | None = None) -> None:
        # per-agent cwd。worktree 子 agent 经 fork_for_child 注入 worktree dir;
        # 主 agent(注册表单例)cwd=None → os.getcwd()(进程 cwd,Phase 1 已 chdir 到 worktree/主仓)。
        self._cwd: str = cwd or os.getcwd()

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """worktree 子 agent:cwd=worktree dir → 新实例 _cwd=worktree;否则继承父 _cwd。"""
        return type(self)(cwd=cwd or self._cwd)

    async def execute(self, *, file_path: str, content: str) -> str:  # type: ignore[override]
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self._cwd) / path
        parent = path.parent
        if not parent.exists():
            # 父目录不存在提示词
            return (
                f"<error>父目录不存在: {parent}</error><hint>先 mkdir,或用 bash 创建目录。</hint>"
            )
        try:
            # 标准库同步写 + to_thread:避免引入 aiofiles(优先标准库)。
            await asyncio.to_thread(_write_text_sync, path, content)
        except PermissionError:
            return (
                f"<error>写入被拒绝(权限不足): {file_path}</error>"
                f"<hint>改用只读方式、换路径,或检查文件权限。</hint>"
            )
        except OSError as exc:
            log.exception("write failed: %s", file_path)
            return f"<error>写入失败: {exc}</error><hint>检查路径/权限/磁盘空间。</hint>"
        global _last_diff
        # 覆盖写的 diff 用 ("", content) 表示「全新内容」(unified diff 会整段标增)。
        _last_diff = ("", content)
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"已写入 {file_path}({lines} 行)。"


def _write_text_sync(path: Path, content: str) -> None:
    # newline="":保留 \n 不在 Windows 上转成 \r\n,避免引入全文件 whitespace diff。
    path.write_text(content, encoding="utf-8", newline="")
