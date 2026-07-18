# src/birdcode/session/subagent_meta.py
"""子 agent meta.json 读写(enriched,可变)。/agents 视图扫 *.meta.json 即得全局状态。"""
from __future__ import annotations

import json
from pathlib import Path

from birdcode.session.jsonutil import read_json_dict
from birdcode.session.models import SubagentMeta
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.session.subagent_meta")


def write_subagent_meta(path: Path, meta: SubagentMeta) -> None:
    """覆盖写 meta.json(唯一可变文件:状态/计时更新)。IO 失败只 log。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(meta.model_dump(by_alias=True, exclude_none=True), ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        log.exception("write_subagent_meta 失败(不杀主循环)")


def read_subagent_meta(path: Path) -> SubagentMeta | None:
    """读 meta.json;缺/坏(非 JSON / 非 dict / 校验失败)→ None。"""
    # 读 JSON → dict|None 委托 read_json_dict(与 store._read_meta 共享,去重:
    # 避免两处并行扩 except 的漂移)。
    data = read_json_dict(path)
    if data is None:
        return None
    try:
        return SubagentMeta.model_validate(data)
    except Exception:
        log.warning("subagent meta 校验失败,忽略: %s", str(data)[:80])
        return None


def list_subagent_metas(subagents_dir: Path) -> list[SubagentMeta]:
    """扫 subagents/*.meta.json → list[SubagentMeta](供 /agents)。坏文件跳过。"""
    if not subagents_dir.exists():
        return []
    out: list[SubagentMeta] = []
    # sorted:glob 顺序随文件系统而异(ext4 哈希序、非确定),排序保证 reminder 行 / UI
    # 列表 / 测试断言顺序确定(否则同批 agent 在不同刷新里列序漂移,LLM 上下文也不稳)。
    for p in sorted(subagents_dir.glob("agent-*.meta.json")):
        m = read_subagent_meta(p)
        if m is not None:
            out.append(m)
    return out
