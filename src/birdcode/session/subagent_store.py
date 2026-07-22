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
                msg,
                uuid=new_uuid,
                parent_uuid=parent_uuid,
                ctx=self._ctx,
                timestamp=timestamp,
                usage=usage,
                stop_reason=stop_reason,
            )
        else:
            line = codec.encode_user(
                msg,
                uuid=new_uuid,
                parent_uuid=parent_uuid,
                ctx=self._ctx,
                timestamp=timestamp,
            )
        # 强制侧链标记(每行)。model_dump 已含 isSidechain=false 默认,这里覆盖为 true + 注 agentId。
        line["isSidechain"] = True
        line["agentId"] = self._agent_id
        if not write_jsonl_line(self._fh, line, label="subagent append"):
            return new_uuid
        self._last_uuid = new_uuid
        return new_uuid

    async def append_compaction_event(self, *, trigger: str, summary: str, fell_back: bool) -> None:
        """记录一次压缩事件(纯观测,不含摘要内容/尾段)。

        type=system + subtype=compact_event:decode_lines 跳过(type 非 user/assistant)、
        find_last_boundary_idx 不认(非 compact_boundary)→ resume 照常重载全量 + 首轮自愈重压,
        事件行只供 read_history/排错可见。不更新 _last_uuid(不进消息 parentUuid 链,保持旁观)。
        IO 失败只 log(write_jsonl_line 内部记录),不杀子 agent。
        """
        line: dict[str, Any] = {
            "type": "system",
            "subtype": "compact_event",
            "isSidechain": True,
            "agentId": self._agent_id,
            "uuid": str(_uuid.uuid4()),
            "parentUuid": self._last_uuid,
            "timestamp": utc_iso(),
            "compact": {"trigger": trigger, "summary": summary, "fellBack": fell_back},
        }
        write_jsonl_line(self._fh, line, label="subagent compact_event")

    async def load_and_repair_sidechain(self, path: Path | None = None) -> list[Turn]:
        """读侧链 + 末尾修复 + 【持久化合成消息】(镜像 SessionStore.load_mainline)。

        续跑前由 runner 调:把 repair 补的合成 tool_result / 收尾 assistant 落盘,使再次中断的
        侧链不再残留 orphan tool_use(否则 direction 追加后成中段 orphan → 二次中断后下轮 API 400
        或 tool_use 被静默剥离)。返回 turns 末尾恒以 assistant 收尾、tool_use 全配对,可直接喂
        run_agent_loop。文件不存在/全坏行 → [](无侧链 = 空历史)。

        path 默认 None → 读 self._jsonl(本 store 写的侧链);runner 续跑传 self.resume_from(生产中
        与 self._jsonl 同路径,二者由同组 ctx/project_root/agent_id 推导)。显式传 path 保留「从
        resume_from 读」的语义(旧代码 load_sidechain_turns(self.resume_from) 的契约 + 回归测试)。
        _last_uuid 先设为所读文件的叶子,使随后 append(合成消息 + direction)正确挂链。decode 按
        文件序重放,parentUuid 不影响顺序。与 codec.load_sidechain_turns(只读,供 resume.py/app.py
        探测)的区别:本方法持久化,只在「即将向侧链追加」的 runner 续跑路径调一次。
        """
        src = path if path is not None else self._jsonl
        rows = codec._read_jsonl_rows(src)  # noqa: SLF001 - 同包内复用容错读
        live = codec.split_live_segment(rows)
        self._last_uuid = next((r.get("uuid") for r in reversed(live) if r.get("uuid")), None)
        turns = codec.decode_lines(live)
        synthetics = codec.repair_trailing_edge(
            turns, error_message_fn=codec.conservative_sidechain_error_message
        )
        for msg in synthetics:
            try:
                await self.append(msg, is_assistant=(msg.role == "assistant"))
            except Exception:  # noqa: BLE001 - 持久化失败不杀续跑,内存 turns 仍可用
                log.exception("持久化侧链合成消息失败(内存 turns 仍可用)")
                break
        if not turns and rows:
            log.warning(
                "load_and_repair_sidechain: 侧链 jsonl 非空但 turns 为空,疑似加载截断(#15837 防御)"
            )
        return turns

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def read_subagent_transcript(path: Path) -> list[Turn]:
    """读 sidechain jsonl → list[Turn](完整对话,供子 agent transcript 查看)。

    复用侧链读取法(同 resume._read_sidechain_final_text),但返完整 codec.decode_lines(非只末条文本)。
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
