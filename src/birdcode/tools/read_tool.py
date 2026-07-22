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
# 大 sidecar 流式读:按 _SIDECAR_CHUNK 块读、按行切;总扫描字节封顶(防 OOM——sidecar 无上限:
# bash communicate 无上限捕获、persist 无 size 守卫)。超限 → 提示 bash sed/head 按字节定位。
_SIDECAR_CHUNK = 64 * 1024
_MAX_SIDECAR_SCAN_BYTES = 16 * 1024 * 1024


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

        缓存 resolve():该目录整会话固定,免得每次 read_file 都做 p.resolve()+目录.resolve()
        两次文件系统解析(Windows 上不便宜)。None=未启用持久化/子 agent → 不豁免。
        """
        self._tool_results_dir = path.resolve() if path is not None else None

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """worktree 子 agent:cwd=worktree dir → 新实例 _cwd=worktree;否则继承父 _cwd。

        _tool_results_dir **不透传**(子默认 None):子 agent 有独立(且默认空)的工具结果
        sidecar 语义,不应共享父的 tool-results 目录(否则子能读父的 Tier1 大输出落盘)。
        子虽接 ContextManager 全量压缩,但 store=None 走内存压缩、且不挂 tool-results 输出
        sink → 不产生 sidecar,尺寸闸不豁免,行为同无持久化场景。
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
        # sidecar(本会话 tool-results 目录下,Tier1 落盘的大输出)豁免拒读,让 demand-paging
        # 能分页读回。大 sidecar 走流式分页(免 read_bytes 整文件入内存 OOM);小 sidecar 仍走
        # 下方 read_bytes 快路径(小、安全)。进 context 的量由 limit/_MAX_INLINE_CHARS 兜底。
        is_sidecar = self._tool_results_dir is not None and p.resolve().is_relative_to(
            self._tool_results_dir
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
        if is_sidecar and stat_result.st_size > _MAX_READ_BYTES:
            # 大 sidecar(无上限:bash 捕获/落盘均无 size 守卫)→ 流式分页读,内存恒定不 OOM。
            return await self._read_sidecar_window(p, offset, limit, stat_result.st_size)
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

    async def _read_sidecar_window(self, p: Path, offset: int, limit: int, size: int) -> str:
        """流式读大 sidecar(>500KB)的 [offset, offset+limit) 行,扫描字节 ≤ _MAX_SIDECAR_SCAN_BYTES。

        大 sidecar 无上限(bash communicate 无上限捕获 / persist 无 size 守卫),read_bytes() 整文件
        入内存有 OOM 风险。此处按 _SIDECAR_CHUNK 块流式读、按 \\n 切行,内存 O(limit×行长 + 块),
        总扫描字节封顶 → sidecar 多大都不 OOM。offset 超出 EOF / 超出可扫描预算 → 提示。进 context
        的量仍由末尾 _MAX_INLINE_CHARS 兜底。
        """
        start = max(1, offset or 1)

        def sniff_binary() -> bytes:
            with p.open("rb") as fh:
                return fh.read(_BINARY_SNIFF_BYTES)

        try:
            head = await asyncio.to_thread(sniff_binary)
        except OSError as exc:
            return f"<error>读取失败: {exc}</error><hint>检查路径/权限。</hint>"
        if _looks_binary(head):
            return (
                "<error>二进制文件,无法文本读取</error>"
                "<hint>若是图片/数据,改用专门工具;若确是文本,检查编码。</hint>"
            )

        def scan() -> tuple[list[str], int, bool, bool]:
            """流式扫行,返回 (window, line_no, reached_eof, budget_hit)。

            单行过长也安全:cur 只累积到下个换行或预算耗尽(≤ 预算+块),不整文件入内存。
            line_no 仅在 reached_eof 时是真实总行数(否则是已扫行数)。
            """
            window: list[str] = []
            line_no = 0
            scanned = 0
            cur = ""
            reached_eof = False
            budget_hit = False
            with p.open("r", encoding="utf-8", errors="replace") as fh:
                while True:
                    chunk = fh.read(_SIDECAR_CHUNK)
                    if not chunk:
                        if cur:  # 末尾无换行的不完整行
                            line_no += 1
                            if start <= line_no < start + limit:
                                window.append(cur)
                        reached_eof = True
                        break
                    scanned += len(chunk)
                    cur += chunk
                    # 从 cur 切出完整行(按换行);窗口满即停,余下留在 cur(已无需)
                    while True:
                        nl = cur.find("\n")
                        if nl == -1:
                            break
                        line = cur[:nl]
                        cur = cur[nl + 1 :]
                        if line.endswith("\r"):
                            line = line[:-1]
                        line_no += 1
                        if start <= line_no < start + limit:
                            window.append(line)
                            if len(window) >= limit:
                                break
                    if len(window) >= limit:
                        break
                    if scanned > _MAX_SIDECAR_SCAN_BYTES:
                        budget_hit = True
                        break
            return window, line_no, reached_eof, budget_hit

        window, total_lines, reached_eof, budget_hit = await asyncio.to_thread(scan)

        if not window:
            if reached_eof:
                return (
                    f"<error>offset {start} 超出文件总行数 {total_lines}</error>"
                    "<hint>用 read_file(offset=1) 从头读,或减小 offset。</hint>"
                )
            return (  # 预算耗尽仍未扫到 offset(文件大、offset 更远)→ 不无限扫
                f"<error>offset {start} 超出可扫描范围(文件 {size} 字节,扫描上限 "
                f"{_MAX_SIDECAR_SCAN_BYTES} 字节)</error>"
                "<hint>大文件请用 bash 的 sed/head 按字节或行定位,或减小 offset 分段读。</hint>"
            )
        body = "\n".join(f"{start + i}\t{ln}" for i, ln in enumerate(window))
        next_offset = start + len(window)
        if budget_hit:
            body += (
                f"\n... [扫描达 {size} 字节、第 {next_offset - 1} 行处触及字节上限 "
                f"{_MAX_SIDECAR_SCAN_BYTES},已停] ..."
                "<hint>需要更靠后的内容请用 bash 的 sed/head 按字节定位。</hint>"
            )
        elif not reached_eof and len(window) >= limit:
            body += (
                f"\n... [已读第 {start}–{next_offset - 1} 行] ..."
                f"<hint>用 read_file(file_path={str(p)!r}, offset={next_offset}, "
                f"limit={limit}) 读后续段落。</hint>"
            )
        # else:reached_eof 且未拿满 limit → 已到末尾,无后续提示
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
