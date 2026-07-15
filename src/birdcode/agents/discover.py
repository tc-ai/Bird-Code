"""续跑发现层:扫子 agent meta,返回可续跑(非终态 ∪ cancelled)集合。"""
from __future__ import annotations

from pathlib import Path

from birdcode.session.models import SubagentMeta
from birdcode.session.subagent_meta import list_subagent_metas

# 非终态(launched/running/idle = 断电/关终端中断)+ cancelled(ESC 中断)。
# 终态 completed/error/cancelled 里只有 cancelled 可续(用户显式中断、非自然终态)。
RESUMABLE_STATUSES = frozenset({"launched", "running", "idle", "cancelled"})


def find_resumable_subagents(subagents_dir: Path) -> list[SubagentMeta]:
    """扫 subagents/*.meta.json → 过滤可续跑 status。

    坏/缺文件由 list_subagent_metas 跳过(read_subagent_meta 返回 None)。
    """
    return [m for m in list_subagent_metas(subagents_dir) if m.status in RESUMABLE_STATUSES]
