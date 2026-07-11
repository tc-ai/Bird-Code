# src/birdcode/session/subagent_store.py
"""SubagentStore:子 agent 侧链会话日志写入器。

复用 codec.encode_user/encode_assistant,但每行强制 isSidechain=true + agentId,
写到 <sid>/subagents/agent-<agentId>.jsonl。首行 parentUuid=null(子 agent 链根,
不挂主会话)。不更新主 <sid>.meta sidecar、不做 resume/compaction(MVP 子 agent
不独立 resume)。IO 失败只 log,不杀主循环(与 SessionStore 一致)。
"""
from __future__ import annotations

import json
import uuid as _uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from birdcode.conversation import Message
from birdcode.session import codec, paths
from birdcode.session.models import SessionContext
from birdcode.session.store import write_jsonl_line
from birdcode.session.timeutil import utc_iso
from birdcode.utils.logging import get_logger
from birdcode.utils.worktree import worktree_store_name

if TYPE_CHECKING:
    from birdcode.agent.provider import TokenUsage
    from birdcode.conversation import Turn

log = get_logger("birdcode.session.subagent")


class SubagentStore:
    def __init__(
        self,
        ctx: SessionContext,
        project_root: Path,
        agent_id: str,
        *,
        root: Path | None = None,
    ) -> None:
        self._ctx = ctx
        self._agent_id = agent_id
        self._root = root if root is not None else paths.default_root()
        wt = worktree_store_name(ctx.cwd)  # 与主会话 jsonl 落同一 worktree/<name>/
        self._jsonl = paths.subagent_jsonl_path(
            self._root, ctx.session_id, project_root, agent_id, worktree_name=wt
        )
        self._jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._jsonl.open("a", encoding="utf-8")
        self._last_uuid: str | None = None  # None → 首行 parentUuid=null

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl

    async def append(
        self,
        msg: Message,
        *,
        is_assistant: bool = False,
        usage: TokenUsage | None = None,
        stop_reason: str | None = None,
    ) -> str:
        """追加一行侧链行。返回本行 uuid;链自动挂(首行 parentUuid=null)。IO 失败只 log。"""
        new_uuid = str(_uuid.uuid4())
        parent_uuid = self._last_uuid
        timestamp = utc_iso()
        if is_assistant:
            line = codec.encode_assistant(
                msg, uuid=new_uuid, parent_uuid=parent_uuid, ctx=self._ctx,
                timestamp=timestamp, usage=usage, stop_reason=stop_reason,
            )
        else:
            line = codec.encode_user(
                msg, uuid=new_uuid, parent_uuid=parent_uuid, ctx=self._ctx, timestamp=timestamp,
            )
        # 强制侧链标记(每行)。model_dump 已含 isSidechain=false 默认,这里覆盖为 true + 注 agentId。
        line["isSidechain"] = True
        line["agentId"] = self._agent_id
        if not write_jsonl_line(self._fh, line, label="subagent append"):
            return new_uuid
        self._last_uuid = new_uuid
        return new_uuid

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def read_subagent_transcript(path: Path) -> list[Turn]:
    """读 sidechain jsonl → list[Turn](完整对话,给 /agents UI 看 teammate transcript)。

    镜像 app._read_sidechain_final_text 的读循环,但返完整 codec.decode_lines(非只末条文本)。
    坏行/缺文件 → 空或跳过(韧性,不抛)。decode_lines 不过滤 isSidechain,直接适用 sidechain 行。
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return codec.decode_lines(rows)
