# src/birdcode/tools/edit.py
"""Edit 工具:str_replace(Search/Replace)。kind=write, parallel_safe=False。

old_string 须在文件中唯一匹配(0/多匹配给自纠提示词);replace_all=True 全替换。
stage4:成功写入后把 (original, new_text) 写入模块级侧信道 _last_diff,
供 executor 读取塞进 ToolResult.diff(unified diff 自然只标改动 hunk)。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel, Field

from birdcode.tools.base import Tool
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.tools.edit_tool")  # 进 debug.log

_last_diff: tuple[str, str] | None = None  # 侧信道:最近一次成功编辑的 (old, new)


def take_last_diff() -> tuple[str, str] | None:
    """executor 在 execute 成功后调用,取走并清空侧信道。"""
    global _last_diff
    d = _last_diff
    _last_diff = None
    return d


class _EditInput(BaseModel):
    file_path: str = Field(..., description="要编辑的文件路径(必填)。")
    old_string: str = Field(..., min_length=1, description="须在文件中唯一匹配的原文(非空)")
    new_string: str = Field(..., description="替换后的新文本(必填)。")
    replace_all: bool = Field(
        default=False, description="True 全替换;默认仅替换首个唯一匹配。"
    )


class EditTool(Tool):
    name = "edit_file"
    description = (
        "对现有文件进行精确的局部字符串替换(Search/Replace)。\n"
        "注意:\n"
        "1. 局部修改优先使用此工具,而不是重写整个文件。\n"
        "2. old_string 必须在文件中绝对唯一匹配;若存在重复内容,"
        "扩大上下文(纳入前后几行)以确保唯一。\n"
        "3. old_string 必须与文件实际内容完全一致(严格匹配缩进、空格、换行)。\n"
        "4. 编辑前先用 read_file 读取文件最新内容,避免基于过时内容替换。"
    )
    parameters = _EditInput
    kind = "write"
    parallel_safe = False
    max_result_chars = 30_000  # BirdCode 选 30K(CC FileEditTool=100K):更早落盘护 context

    def __init__(self, *, cwd: str | None = None) -> None:
        # per-agent cwd。worktree 子 agent 经 fork_for_child 注入 worktree dir;
        # 主 agent(注册表单例)cwd=None → os.getcwd()(进程 cwd,Phase 1 已 chdir 到 worktree/主仓)。
        self._cwd: str = cwd or os.getcwd()

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """worktree 子 agent:cwd=worktree dir → 新实例 _cwd=worktree;否则继承父 _cwd。"""
        return type(self)(cwd=cwd or self._cwd)

    async def execute(  # type: ignore[override]
        self, *, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self._cwd) / path
        if not path.exists():
            return (
                f"<error>文件不存在: {file_path}</error>"
                f"<hint>用 read_file 或 glob 确认路径、检查拼写。</hint>"
            )
        try:
            original = await asyncio.to_thread(_read_text_sync, path)
        except UnicodeDecodeError:
            return (
                f"<error>无法以文本读取(可能是二进制): {file_path}</error>"
                f"<hint>edit_file 仅支持文本文件。</hint>"
            )

        count = original.count(old_string)
        if count == 0:
            # 0 匹配:引导核对缩进/首尾空白/大小写,用 Read 带行号核对。
            return (
                "<error>old_string 未找到(0 匹配)</error>"
                "<hint>检查缩进/首尾空白/大小写;用 read_file(file_path) 带行号核对目标段落;"
                "注意 old_string 须与文件完全一致(含缩进)。</hint>"
            )
        if count > 1 and not replace_all:
            # 多匹配:引导扩大上下文唯一化,或设 replace_all=true。
            return (
                f"<error>old_string 匹配 {count} 次(需唯一)</error>"
                "<hint>扩大 old_string 范围纳入更多上下文以唯一化,"
                "或设 replace_all=true 全替换。</hint>"
            )

        new_text = (
            original.replace(old_string, new_string)
            if replace_all
            else original.replace(old_string, new_string, 1)
        )
        try:
            await asyncio.to_thread(_write_text_sync, path, new_text)
        except PermissionError:
            return (
                f"<error>写入被拒绝(权限不足): {file_path}</error>"
                f"<hint>检查文件权限或换路径。</hint>"
            )

        global _last_diff
        _last_diff = (original, new_text)  # 完整文件前后;unified diff 只标改动 hunk

        changed = new_text.count("\n") - original.count("\n")
        n = abs(changed) if changed != 0 else new_string.count("\n") + 1
        return f"已编辑 {file_path}(改动约 {n} 行)。"


def _read_text_sync(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text_sync(path: Path, content: str) -> None:
    # newline="":保留 \n 不在 Windows 上转 \r\n,避免引入全文件 whitespace diff。
    path.write_text(content, encoding="utf-8", newline="")
