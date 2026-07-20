# src/birdcode/tools/read_history.py
"""ReadHistory 工具:读取【当前会话】被压缩掉的历史对话原文。

压缩(compact)后,LLM 只看到【摘要 + 尾段】;前文真实对话被摘要替代、细节丢失。
本工具从当前会话的 jsonl 里,以【最近一次压缩点】为基点,分页取回压缩点之前、已不在
上下文中的对话原文(用户原话 / assistant 回复 / 工具调用与结果)。

**只读当前会话**:不扫会话目录、不接受路径参数 —— jsonl 路径由 app 从 SessionStore
注入(set_jsonl),结构上杜绝读到其它会话。未注入路径(未启用持久化)→ 提示无可读历史。
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from birdcode.blocks import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Turn
from birdcode.session import codec
from birdcode.tools.base import Tool
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.tools.read_history")

_DEFAULT_LIMIT = 3  # 默认读取的轮数上限(一轮 = 一对 user-assistant)
# 单条 tool_result / thinking 块的渲染截断(字符):工具结果可能极大(整文件 read),截断防爆。
_MAX_BLOCK_CHARS = 2000
# 总输出 inline 截断阈值(字符):read_history 永不落盘(inf,避免「读落盘文件→再超阈」循环,
# 同 read_file 理由),超此 → inline 截断 + 提示减小 limit。
_MAX_OUTPUT_CHARS = 50_000


class _ReadHistoryInput(BaseModel):
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "从【最近一次压缩点】往回数的轮次偏移(0-based)。0=紧邻压缩点的最新一轮已压缩对话,"
            "数字越大越早(更旧)。默认 0,即先取最近被压缩的内容。"
        ),
    )
    limit: int = Field(
        default=_DEFAULT_LIMIT,
        ge=1,
        description="本次最多返回多少轮(一轮 = 一对 user-assistant)。",
    )


class ReadHistoryTool(Tool):
    name = "read_history"
    description = (
        "读取【当前会话】被压缩掉的历史对话原文(最近一次压缩点之前、已不在上下文中的内容)。\n"
        "用途:压缩后若摘要不够、需回看之前的真实对话(用户原话 / 工具结果等),用本工具取回。\n"
        "参数:\n"
        "1. offset(默认 0):已压缩历史的偏移。0=最近一段丢失的对话,数字越大越早。\n"
        "2. limit(默认 3):本次最多返回多少轮(一轮=一对 user-assistant)。\n"
        "返回结果按【最新 → 最早】展示;取更早的历史用更大的 offset(如 offset += limit)。\n"
        "注意:只读当前会话(不读其它会话);当前会话尚未压缩时无可读历史。"
    )
    parameters = _ReadHistoryInput
    kind: Literal["read"] = "read"
    parallel_safe = True
    # 持有【主线会话】jsonl 路径(由 app 注入)。子 agent 的对话在侧链、格式不同且无压缩,
    # 复用本实例会让子 agent 读到【主 agent】的历史 → 跨 agent 信息泄漏 + 数据错配。
    # 标 session_scoped → build_child_registry 排除(子 agent 不持有 read_history)。
    session_scoped = True
    # 永不落盘(inf):同 read_file —— 落盘后模型再去 read 那个文件又超阈 → 死循环。
    # 输出由两道闸界定:① limit 轮(分页)② _MAX_OUTPUT_CHARS inline 截断。
    max_result_chars: float = math.inf

    def __init__(self) -> None:
        # 当前会话的 jsonl 路径,由 app(SessionStore.jsonl_path)注入;None=未启用持久化。
        # 不设路径参数 / 不扫目录 → 只可能是当前会话,绝不读到其它会话。
        self._jsonl: Path | None = None

    def set_jsonl(self, path: Path | None) -> None:
        """注入当前会话的 jsonl 路径(app 在 on_mount / /clear 时调)。

        None 表示未启用持久化(无 store),execute 会返回「未启用」提示。
        """
        self._jsonl = path

    async def execute(  # type: ignore[override]
        self, *, offset: int = 0, limit: int = _DEFAULT_LIMIT
    ) -> str:
        path = self._jsonl
        if path is None:
            return (
                "<error>会话持久化未启用,无历史记录可读</error>"
                "<hint>启用会话持久化(配置 store)后再使用 read_history。</hint>"
            )
        # 读 + 解码 + 渲染放 to_thread,避免 async 里阻塞大 jsonl(核心 I/O 异步)。
        return await asyncio.to_thread(self._load_window, path, offset, limit)

    # —— 同步主体(在 to_thread 里跑)——

    def _load_window(self, path: Path, offset: int, limit: int) -> str:
        """读当前会话 jsonl → 定位最近压缩点 → 取压缩点之前 [offset, offset+limit) 轮 → 渲染。"""
        try:
            rows = _read_jsonl_rows(path)
        except PermissionError:
            log.warning("read_history 权限拒绝: %s", path)
            return f"<error>权限拒绝: {path}</error><hint>检查文件读权限。</hint>"
        except OSError as exc:
            # 文件级失败落 debug.log,使「详见 debug.log」提示成立。
            log.warning("read_history 读取会话记录失败: %s", path, exc_info=True)
            return f"<error>读取会话记录失败: {exc}</error><hint>详见 debug.log。</hint>"

        boundary_idx = codec.find_last_boundary_idx(rows)
        if boundary_idx < 0:
            # 无压缩边界 → 全部对话仍在 live 上下文,read_history 无可读历史。
            return (
                "(当前会话尚未压缩,全部对话均在上下文中,无需 read_history。)"
                "<hint>压缩(/compact 或自动触发)后,被压缩掉的前文才可用本工具取回。</hint>"
            )
        # 压缩点(boundary,system 行)是定界符:切片排除它 + decode_lines 跳过 system 行,
        # 永不占 limit。更早的【压缩摘要行】(isCompactSummary 的合成 user 行)也不是
        # 真实 user-assistant 对(需求:消息=一对 user-assistant),过滤掉 —— 否则它会占
        # limit 名额、还作为一条「轮次」展示,把真实对话顶出窗口。多次压缩时中间的
        # boundary + 摘要均不计入,limit 只数真实对话轮。
        historical = [r for r in rows[:boundary_idx] if not r.get("isCompactSummary")]
        turns = codec.decode_lines(historical)
        # 尾段(tail)压缩时被原样保留进 live 上下文、又重写到边界之后,其【原始行】仍留在
        # 边界之前。preservedTurnCount(压缩时落 compactMetadata)= 尾段轮数 → 丢掉解码出的
        # 最后 N 轮(= 尾段原始副本,已在当前上下文),只剩真正被摘要替代、已丢失的 prefix。
        # 缺失/非法(老边界、本修复前的会话)→ N=0,退回「读全部」旧行为(头几页冗余但不报错)。
        n_tail = _preserved_turn_count(rows[boundary_idx])
        prefix_turns = turns[:-n_tail] if n_tail > 0 else turns
        if not prefix_turns:
            return "(最近一次压缩的尾段已包含全部对话,无被压缩丢失的前文可读。)"

        total = len(prefix_turns)
        # prefix_turns 最早→最新;反转成最新在前(最新优先展示)。
        newest_first = prefix_turns[::-1]
        window = newest_first[offset : offset + limit]
        if not window:
            return (
                f"<error>offset {offset} 超出已压缩历史共 {total} 轮</error>"
                f"<hint>用 read_history(offset=0, limit={limit}) 从最近一段开始读。</hint>"
            )

        # 渲染:表头 + 各轮(最新→最早)+ 翻页提示。
        lines: list[str] = [f"被压缩丢失的历史共 {total} 轮(按 最新→最早 展示)。"]
        if n_tail > 0:
            lines.append(f"(尾段最近 {n_tail} 轮已在当前上下文,已自动跳过。)")
        lines += [
            f"本次返回第 {offset + 1}–{offset + len(window)} 段(limit={limit})。",
            f"会话来源: {path}",
            "",
        ]
        for i, turn in enumerate(window):
            lines.extend(_render_turn(turn, seq=offset + i + 1))
            lines.append("")
        hint_lines = _pagination_hints(offset, len(window), total, limit)
        body = "\n".join(lines + hint_lines)

        # 总输出 inline 截断(永不落盘):长会话单窗也可能爆,提示减小 limit。
        if len(body) > _MAX_OUTPUT_CHARS:
            if limit <= 1:
                # 已是最小 limit:本窗含超大块(单轮就超限),无法靠缩小 limit 取回。
                hint = (
                    f"\n... [输出超 {_MAX_OUTPUT_CHARS} 字符已截断 —"
                    " 本窗含超大块,limit=1 仍超限,该段历史无法完整取回] ..."
                )
            else:
                smaller = max(1, limit // 2)
                hint = (
                    f"\n... [输出超 {_MAX_OUTPUT_CHARS} 字符已截断 —"
                    f" 用更小 limit(如 {smaller})分页读完整内容] ..."
                )
            body = body[:_MAX_OUTPUT_CHARS] + hint
        return body


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读 jsonl 主线行:跳过空/坏 JSON/非 dict/isSidechain(镜像 SessionStore._read_mainline_rows)。

    静默跳过坏行(best-effort,只读视图):工具层不告警每一条坏行,保持输出干净;
    残缺行/块的容错由 codec.decode_lines 内部告警处理。
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("isSidechain"):
                continue
            rows.append(obj)
    return rows


def _render_turn(turn: Turn, *, seq: int) -> list[str]:
    """渲染一个 Turn(=一对 user-assistant)为多行块。seq=本窗内序号(1=最新的一段已压缩历史)。

    每个块都按 _MAX_BLOCK_CHARS 截断:text/args/tool_result 均可能极大(大段粘贴、
    write_file 的 content、整文件 read 结果)—— 不截断则单块即可撑爆 _MAX_OUTPUT_CHARS、
    把同窗其余轮次挤掉。tag 标注被截断的块。
    """
    lines = [f"━━━ 已压缩历史第 {seq} 段 ━━━"]
    for m in turn.messages:
        for b in m.content:
            if isinstance(b, TextBlock):
                t, tag = _trunc(b.text)
                lines.append(f"[{m.role}{tag}] {t}")
            elif isinstance(b, ThinkingBlock):
                t, tag = _trunc(b.text)
                lines.append(f"[thinking{tag}] {t}")
            elif isinstance(b, ToolUseBlock):
                t, tag = _trunc(json.dumps(b.input, ensure_ascii=False))
                lines.append(f"[tool_use {b.name}{tag}] {t}")
            elif isinstance(b, ToolResultBlock):
                t, tag = _trunc(b.content)
                lines.append(f"[tool_result{tag}] {t}")
    return lines


def _trunc(text: str) -> tuple[str, str]:
    """超 _MAX_BLOCK_CHARS → (截断文本, ' (截断) 标记);否则 (原文, '')。"""
    if len(text) <= _MAX_BLOCK_CHARS:
        return text, ""
    return text[:_MAX_BLOCK_CHARS], " (截断)"


def _preserved_turn_count(boundary_row: dict[str, Any]) -> int:
    """从边界行 compactMetadata 读 preservedTurnCount(尾段轮数);缺失/非法 → 0。

    0 = 不跳过:老边界 / 本修复前的会话没有此字段 → 退回「读全部边界前轮次」的旧行为
    (头几页与当前上下文重复,但不报错、不丢数据)。仅接受非负 int(防御 bool/漂移)。
    """
    meta = boundary_row.get("compactMetadata")
    if not isinstance(meta, dict):
        return 0
    n = meta.get("preservedTurnCount")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return 0
    return n


def _pagination_hints(offset: int, got: int, total: int, limit: int) -> list[str]:
    """翻页提示:更早(offset+) / 更新(offset-,向压缩点靠拢)。已到端点则省略对应行。"""
    hints: list[str] = []
    if offset + got < total:
        hints.append(f"[hint] 更早(更旧)的历史: read_history(offset={offset + got}, limit={limit})")
    if offset > 0:
        prev = max(0, offset - limit)
        hints.append(f"[hint] 更新(更接近压缩点)的历史: read_history(offset={prev}, limit={limit})")
    return hints
