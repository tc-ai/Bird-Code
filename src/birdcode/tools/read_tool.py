# src/birdcode/tools/read_tool.py
"""Read 工具:读文件正文,cat -n 式带行号,offset/limit 分页。"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from birdcode.tools.base import Tool

_DEFAULT_LIMIT = 500
_MAX_READ_BYTES = 500 * 1024  # 500KB 尺寸护栏:超限拒,引导分页/Grep
# inline 截断阈值(字符):read 永不落盘(inf,防「读落盘文件→再超阈」循环),故不能靠
# persist 兜底 context;limit 按行不按字符,长行文件(minified / 大日志)500 行仍可达
# 数百 KB → 超 LLM 上下文 → API 400。此 cap 是 context 安全的最后一道闸,inline 截断 + 提示
# 分页(模型用 offset/limit 取后续)。50K:正常源文件(<20K)极少触发,仅兜病理大读。
_MAX_INLINE_CHARS = 50_000
# 二进制探测:前 NKB 中 NUL 字节占比超阈值 → 判二进制
_BINARY_SNIFF_BYTES = 4096
_BINARY_NUL_RATIO = 0.30


class _ReadInput(BaseModel):
    file_path: str = Field(..., description="要读取的文件路径(必填)。")
    offset: int | None = Field(default=None, description="起始行号(从 1 起),用于分页。")
    limit: int = Field(default=500, description="读取的行数上限,默认 500。")


class ReadTool(Tool):
    name = "read_file"
    description = (
        "读取文件内容,以带行号(类似 cat -n)的格式输出。\n"
        "注意:\n"
        "1. 默认读取前 500 行;大文件请用 offset 和 limit 分页读取所需部分,避免上下文溢出。\n"
        "2. 优先使用此工具,而不是在 bash 中执行 cat/head/tail。\n"
        "3. 修改文件前(edit_file/write_file)建议先用此工具读取最新内容与准确行号。"
    )
    parameters = _ReadInput
    kind: Literal["read"] = "read"
    parallel_safe = True
    # 永不落盘(max_result_chars=inf):若落盘,模型再去 read 那个落盘文件又超阈 → 死循环。
    # 仿 CC FileReadTool=Infinity。输出由三道闸界定(防 context 溢出,见 execute 末尾):
    # ① limit=500 行(分页)② _MAX_READ_BYTES=500KB(拒读)③ _MAX_INLINE_CHARS inline 截断。
    max_result_chars: float = math.inf

    def __init__(self, *, cwd: str | None = None) -> None:
        # per-agent cwd。worktree 子 agent 经 fork_for_child 注入 worktree dir;
        # 主 agent(注册表单例)cwd=None → os.getcwd()(进程 cwd,Phase 1 已 chdir 到 worktree/主仓)。
        self._cwd: str = cwd or os.getcwd()
        # 本会话 tool-results 目录(ui/app.py _wire_read_file 注入,照搬 read_history.set_jsonl)。
        # None=未启用持久化/子 agent → 尺寸闸不豁免,行为同今天。
        self._tool_results_dir: Path | None = None

    def set_tool_results_dir(self, path: Path | None) -> None:
        """注入本会话 tool-results 目录(<sessiondir>/tool-results/)。

        挂载 + /clear(切新 session)时调,与 read_history.set_jsonl / executor
        update_output_sink 同模式。read_file 据此豁免 sidecar 的 500KB 拒读
        (Tier1 落盘的大输出需能分页读回,见尺寸闸 _MAX_READ_BYTES)。
        """
        self._tool_results_dir = path

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """worktree 子 agent:cwd=worktree dir → 新实例 _cwd=worktree;否则继承父 _cwd。

        _tool_results_dir **不透传**(子默认 None):子 agent sidecar 目录与父不同,
        且一次性、child_context=None、无 compaction,不产生 sidecar,无需豁免。
        """
        return type(self)(cwd=cwd or self._cwd)

    async def execute(  # type: ignore[override]
        self, *, file_path: str, offset: int | None = None, limit: int = _DEFAULT_LIMIT
    ) -> str:
        p = Path(file_path)
        if not p.is_absolute():
            p = Path(self._cwd) / p
        if not p.exists():
            return (
                f"<error>文件不存在: {file_path}</error><hint>用 glob 确认路径或检查拼写。</hint>"
            )
        if p.is_dir():
            return (
                f"<error>路径是目录,不是文件: {file_path}</error>"
                "<hint>用 glob 列出该目录下文件,再 read_file 具体文件。</hint>"
            )
        # 尺寸护栏:超限直接拒(避免整文件入内存 OOM),引导分页/Grep 定位。
        # sidecar 豁免:本会话 tool-results 目录下的文件(Tier1 落盘的大输出)跳过
        # 拒读,让 demand-paging 能分页读回;进 context 的量仍由 limit/_MAX_INLINE_CHARS 兜底。
        is_sidecar = self._tool_results_dir is not None and p.resolve().is_relative_to(
            self._tool_results_dir.resolve()
        )
        try:
            stat_result = await asyncio.to_thread(p.stat)
        except PermissionError:
            return (
                f"<error>权限拒绝: {file_path}</error>"
                "<hint>检查文件权限,或换有读权限的路径。</hint>"
            )
        except OSError as exc:
            return f"<error>读取失败: {exc}</error><hint>检查路径/权限。</hint>"
        if not is_sidecar and stat_result.st_size > _MAX_READ_BYTES:
            return (
                f"<error>文件过大: {stat_result.st_size} 字节(上限 {_MAX_READ_BYTES})</error>"
                "<hint>read_file 会整文件读入内存,offset/limit 也绕不过此限。"
                "用 grep 按内容定位,或用 bash 的 sed/head 流式取局部。</hint>"
            )
        # 同步 read_bytes 放进 to_thread,避免 async 里阻塞(核心 I/O 异步)。
        try:
            raw = await asyncio.to_thread(p.read_bytes)
        except PermissionError:
            return (
                f"<error>权限拒绝: {file_path}</error>"
                "<hint>检查文件权限,或换有读权限的路径。</hint>"
            )
        if _looks_binary(raw):
            return (
                "<error>二进制文件,无法文本读取</error>"
                "<hint>若是图片/数据,改用专门工具;若确是文本,检查编码。</hint>"
            )
        text = raw.decode("utf-8", errors="replace")
        all_lines = text.splitlines()
        total = len(all_lines)
        start = max(1, offset or 1)
        idx = start - 1  # 0-based 起始
        window = all_lines[idx : idx + limit]
        if not window:
            return (
                f"<error>offset {start} 超出文件总行数 {total}</error>"
                "<hint>用 read_file(offset=1) 从头读,或减小 offset。</hint>"
            )
        body = "\n".join(f"{start + i}\t{ln}" for i, ln in enumerate(window))
        if start + limit - 1 < total:
            next_offset = start + len(window)
            body += (
                f"\n... [已截断 — 共 {total} 行,已读第 {start}–{next_offset - 1} 行] ..."
                f"<hint>用 read_file(file_path={file_path!r}, offset={next_offset}, "
                f"limit={limit}) 读后续段落。</hint>"
            )
        # context 安全兜底:limit 按行不按字符,长行文件仍可能巨大(超 LLM 上下文 → API 400)。
        # read 永不落盘(inf),故 inline 截断 + 提示分页(非 persist),模型用 offset/limit 取后续。
        if len(body) > _MAX_INLINE_CHARS:
            hint = (
                f"\n... [输出超 {_MAX_INLINE_CHARS} 字符已截断 —"
                " 用 offset/limit 分页读完整内容] ..."
            )
            body = body[:_MAX_INLINE_CHARS] + hint
        return body


def _looks_binary(raw: bytes) -> bool:
    """启发式:前几 KB 中 NUL 字节占比高 → 判二进制(仿 ripgrep 的过滤)。"""
    if not raw:
        return False
    sniff = raw[:_BINARY_SNIFF_BYTES]
    if b"\x00" not in sniff:
        return False
    nul = sniff.count(0)
    return nul / len(sniff) > _BINARY_NUL_RATIO
