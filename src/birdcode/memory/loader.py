# src/birdcode/memory/loader.py
"""启动加载:镜像 BirdCode.md 两级管线,扫 .md 文件拼成注入块(带上限)。

项目级在前(高优先级)、用户级在后;都无 → ""。合并后超 MAX_INDEX_LINES 行或
MAX_INDEX_BYTES 字节则截断 + 附 <!-- --> 警告,防记忆撑爆上下文。

**直接扫 .md 文件(源真相),不读预构建的 MEMORY.md**——后者只是 rebuild_index
生成的人读派生物,删/损/手改它不应让记忆从系统提示词消失;手改/新增 .md 下次
启动即生效。MEMORY.md 仍由 rebuild_index 写出供人查阅,但加载端不依赖它。

本会话内不热加载(同 BirdCode.md):启动读一次,新记忆重启后才注入——换取
system 前缀字节稳定、利于 Anthropic 前缀缓存命中。
"""
from __future__ import annotations

from pathlib import Path

from birdcode.memory.paths import project_memory_dir, user_memory_dir
from birdcode.memory.store import format_pointers, list_memories

MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25 * 1024  # 25KB


def load_memory_index(project_root: Path) -> str:
    """加载并拼装记忆索引(项目级在前)。都无 → ""。超限截断 + 警告。"""
    proj = format_pointers(list_memories(project_memory_dir(project_root)))
    user = format_pointers(list_memories(user_memory_dir()))
    if not proj and not user:
        return ""
    body = "\n---\n".join([s for s in (proj, user) if s])
    out = (
        f"记忆（· 项目级 + 用户级｜）\n\n{body}\n "
        "切勿尝试修改记忆，这只是用于给你提供信息"
    )
    return _cap(out)


def _cap(text: str) -> str:
    """行数与字节双重上限,先到先截;截断附警告。字节截断按 UTF-8 边界 errors=ignore。"""
    truncated = False
    lines = text.split("\n")
    if len(lines) > MAX_INDEX_LINES:
        text = "\n".join(lines[:MAX_INDEX_LINES])
        truncated = True
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_INDEX_BYTES:
        text = encoded[:MAX_INDEX_BYTES].decode("utf-8", errors="ignore")
        truncated = True
    if truncated:
        text = text + "\n<!-- 记忆索引超限,已截断 -->"
    return text
