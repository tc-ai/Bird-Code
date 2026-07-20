# src/birdcode/memory/store.py
"""记忆文件存储:frontmatter 解析/序列化、slugify、原子读写、列表、索引重建。

frontmatter(YAML):
    ---
    name: prefers-any
    description: 用户偏好 any 而非 interface{}
    type: feedback          # user/feedback/project/reference
    ---
    <body>

MEMORY.md 是从 list_memories 扫描重生成的纯指针索引(自动维护,非手写)。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from birdcode.utils.logging import get_logger

log = get_logger("birdcode.memory.store")

INDEX_FILE = "MEMORY.md"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class MemoryFrontmatter(BaseModel):
    """单条记忆的元信息(type 决定落哪个目录)。"""

    name: str
    description: str
    type: Literal["user", "feedback", "project", "reference"]


@dataclass
class MemoryMeta:
    """list_memories 返回项:name/description/type + 文件路径。"""

    name: str
    description: str
    type: str
    path: Path


def slugify(name: str) -> str:
    """规范化为 kebab-case slug:[a-z0-9-]+。空/全非法 → untitled。"""
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return s or "untitled"


def serialize_memory(*, name: str, description: str, type_: str, body: str) -> str:
    """序列化 frontmatter + body 为完整 .md 文本。"""
    fm = {"name": name, "description": description, "type": type_}
    head = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{head}\n---\n{body}"


def _is_fm_delim(line: str) -> bool:
    """是否 YAML frontmatter 分隔符:列 0(无缩进)、内容恰为 ---(允许尾随空白)。

    列 0 判定是关键——YAML 多行标量(如 description 含换行)的续行是**缩进的** `  ---`,
    若用 `strip()=="---"` 会把它误判为闭合边界,致 frontmatter 被截断、记忆静默丢失。
    """
    return line.rstrip() == "---" and not line.startswith((" ", "\t"))


def parse_memory(text: str) -> tuple[MemoryFrontmatter, str] | None:
    """解析 frontmatter + body。无 frontmatter / 不闭合 / YAML 错 / 校验错 → None + log。

    行级匹配分隔符(列 0 的 ---),而非子串 split——否则 name/description 里含 ---
    会被误判为边界,致静默丢记忆。容忍首部 BOM(win32 Notepad 等另存为 UTF-8 BOM)。
    """
    if text.startswith("﻿"):  # 容忍 UTF-8 BOM
        text = text[1:]
    lines = text.split("\n")
    if not lines or not _is_fm_delim(lines[0]):
        log.warning("记忆文件缺 frontmatter,跳过")
        return None
    end: int | None = None
    for i in range(1, len(lines)):
        if _is_fm_delim(lines[i]):
            end = i
            break
    if end is None:
        log.warning("记忆文件 frontmatter 不闭合,跳过")
        return None
    fm_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :])
    try:
        fm_raw = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        log.warning("记忆 frontmatter YAML 解析失败,跳过")
        return None
    try:
        fm = MemoryFrontmatter.model_validate(fm_raw)
    except ValidationError:
        log.warning("记忆 frontmatter 校验失败,跳过")
        return None
    return fm, body.lstrip("\n")


def _memory_path(dir_: Path, name: str) -> Path:
    """name(slugify 后)→ <dir>/<slug>.md。"""
    return dir_ / f"{slugify(name)}.md"


def write_memory(
    dir_: Path,
    *,
    name: str,
    description: str,
    type_: str,
    body: str,
) -> Path:
    """原子写(temp + os.replace)。dir 自动创建。返回写入路径。

    原子写保证:写到一半被取消/崩溃不留半截文件(temp 未 rename 前目标不变)。
    """
    dir_.mkdir(parents=True, exist_ok=True)
    target = _memory_path(dir_, name)
    text = serialize_memory(name=name, description=description, type_=type_, body=body)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    return target


def delete_memory(dir_: Path, name: str) -> bool:
    """删 <slug>.md。不存在返回 False(幂等)。"""
    try:
        _memory_path(dir_, name).unlink()
        return True
    except FileNotFoundError:
        return False


def memory_exists(dir_: Path, name: str) -> bool:
    return _memory_path(dir_, name).is_file()


def list_memories(dir_: Path) -> list[MemoryMeta]:
    """扫描 dir/*.md(排除 MEMORY.md 与 .tmp),解析 frontmatter。畸形跳过 + log。

    按 name 排序(稳定输出,供 rebuild_index 与摘要清单复用)。
    """
    if not dir_.is_dir():
        return []
    out: list[MemoryMeta] = []
    for p in sorted(dir_.glob("*.md")):
        if p.name == INDEX_FILE:
            continue
        try:
            text = p.read_text(encoding="utf-8-sig")  # utf-8-sig:兼容 win32 BOM 头
        except UnicodeDecodeError:
            # 非 UTF-8(win32 GBK/GB18030 等编辑器默认)→ 跳过 + 提示,绝不崩启动
            log.warning("记忆文件非 UTF-8,跳过(请存为 UTF-8): %s", p)
            continue
        except OSError:
            continue
        parsed = parse_memory(text)
        if parsed is None:
            continue
        fm, _body = parsed
        out.append(MemoryMeta(name=fm.name, description=fm.description, type=fm.type, path=p))
    out.sort(key=lambda m: m.name)
    return out


def format_pointers(metas: list[MemoryMeta]) -> str:
    """metas → MEMORY.md 指针行(每行 `- [description](filename)`,按入参顺序)。

    loader 注入(系统提示词)与 rebuild_index 写盘(人读 MEMORY.md)的**单一来源**,
    防两处格式各自演化而漂移。返回不含尾随换行——文件写由调用方按需补 \\n。
    """
    return "\n".join(f"- [{m.description}]({m.path.name})" for m in metas)


def rebuild_index(dir_: Path) -> None:
    """从 list_memories 重生成该目录的 MEMORY.md(原子写,按 name 排序)。

    目录不存在且无记忆 → 不创建(避免空目录污染)。否则写(空记忆→空索引)。
    """
    metas = list_memories(dir_)
    if not metas and not dir_.is_dir():
        return
    dir_.mkdir(parents=True, exist_ok=True)
    body = format_pointers(metas)
    text = (body + "\n") if body else ""
    idx = dir_ / INDEX_FILE
    tmp = idx.with_name(idx.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, idx)


def read_index(dir_: Path) -> str:
    """读 MEMORY.md;不存在/读失败 → ""。"""
    idx = dir_ / INDEX_FILE
    if not idx.is_file():
        return ""
    try:
        return idx.read_text(encoding="utf-8")
    except OSError:
        log.warning("读取记忆索引失败: %s", idx)
        return ""
