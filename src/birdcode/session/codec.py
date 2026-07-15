# src/birdcode/session/codec.py
"""翻译层:内存 Message/ContentBlock(dataclass) ↔ 持久化行(Pydantic) 双向无损转换。

block 用 Anthropic 原生名,未来 provider 层可把历史行 message.content 原样塞进 API 请求。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from birdcode.agent.provider import TokenUsage
from birdcode.blocks import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from birdcode.conversation import Message, Turn
from birdcode.session.models import (
    AssistantLine,
    QueueOperationLine,
    SessionContext,
    SystemLine,
    UserLine,
)
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.session.codec")

# 待续跑 sync agent 的合成 tool_result 占位文案(覆盖 repair 默认「请重新发起」)。
# 主会话 resume 时,sync agent tool_use 尚无真结果(中断期未落盘)→ 用此占位;
# 真结果经 resume_agent inline 返回(见 agents.resume.resume_subagent)。
# 含「中断/续跑」:与 repair 默认文案(含「中断/请重新发起」)区分——「续跑」为占位专属关键词。
_SYNC_PENDING_PLACEHOLDER = "子 agent 因会话中断未完成,待续跑(结果经续跑 inline 返回)"

# ---- block ↔ dict (Anthropic 原生名) ----


def block_to_dict(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "text": block.text, "signature": block.signature}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    # 兜底返回占位 TextBlock(保护 live agent loop 不崩;未知 block 不杀主循环)。
    return {"type": "text", "text": f"[unknown block: {type(block).__name__}]"}


def block_from_dict(d: dict[str, Any]) -> ContentBlock | None:
    """未知 type 或关键字段残缺 → None(不抛),由 message_from_dict 过滤+告警。

    残缺判定:text/thinking 缺 text;tool_use 缺 id 或 name;tool_result 缺
    tool_use_id 或 content(非 str)。任一不满足 → None,使单块损坏不杀整条 resume。
    """
    if not isinstance(d, dict):
        return None
    t = d.get("type")
    if t == "text":
        text = d.get("text")
        return TextBlock(text=text) if isinstance(text, str) else None
    if t == "thinking":
        text = d.get("text")
        if not isinstance(text, str):
            return None
        sig = d.get("signature")
        return ThinkingBlock(text=text, signature=sig if isinstance(sig, str) else "")
    if t == "tool_use":
        tid = d.get("id")
        name = d.get("name")
        if not isinstance(tid, str) or not isinstance(name, str):
            return None
        inp = d.get("input")
        return ToolUseBlock(id=tid, name=name, input=inp if isinstance(inp, dict) else {})
    if t == "tool_result":
        tuid = d.get("tool_use_id")
        content = d.get("content")
        if not isinstance(tuid, str) or not isinstance(content, str):
            return None
        return ToolResultBlock(
            tool_use_id=tuid, content=content, is_error=bool(d.get("is_error", False))
        )
    return None


# ---- message ↔ dict ----


def message_to_dict(msg: Message) -> dict[str, Any]:
    return {"role": msg.role, "content": [block_to_dict(b) for b in msg.content]}


def message_from_dict(d: dict[str, Any]) -> Message | None:
    """非 dict / role 非法 / content 非列表 / 过滤后空 → None(跳过整条)。

    空消息(所有 block 都残缺)也返回 None:避免空 content 触发 API 400。
    残缺 block / 整条丢弃均告警,便于发现持久化侧 schema 漂移。
    """
    if not isinstance(d, dict):
        log.warning("message_from_dict 跳过:非 dict message")
        return None
    role = d.get("role")
    if role not in ("user", "assistant"):
        log.warning("message_from_dict 跳过:非法 role=%r", role)
        return None
    raw = d.get("content")
    if not isinstance(raw, list):
        log.warning("message_from_dict 跳过:content 非列表(role=%s)", role)
        return None
    blocks: list[ContentBlock] = []
    dropped = 0
    for x in raw:
        b = block_from_dict(x)
        if b is None:
            dropped += 1
        else:
            blocks.append(b)
    if dropped:
        log.warning("message_from_dict 丢弃 %d 个残缺/未知 block(role=%s)", dropped, role)
    if not blocks:
        log.warning("message_from_dict 跳过:整条无可用 block(role=%s)", role)
        return None
    return Message(role=role, content=blocks)


# ---- Message → 行(camelCase by_alias) ----


def encode_user(
    msg: Message,
    *,
    uuid: str,
    parent_uuid: str | None,
    ctx: SessionContext,
    timestamp: str,
    tool_use_results: dict[str, Any] | None = None,
    source_tool_assistant_uuid: str | None = None,
    is_task_notification: bool = False,
    agent_id: str | None = None,
) -> dict[str, Any]:
    # 用 model_validate(dict) 而非关键字构造:populate_by_name=True 让 dict 的
    # snake_case key 被接受,同时规避 mypy pydantic 对 alias 字段 call-arg 的误报。
    line = UserLine.model_validate(
        {
            "uuid": uuid,
            "parent_uuid": parent_uuid,
            "session_id": ctx.session_id,
            "timestamp": timestamp,
            "cwd": ctx.cwd,
            "version": ctx.version,
            "git_branch": ctx.git_branch,
            "message": message_to_dict(msg),
            "synthetic": msg.synthetic,
        }
    )
    out = line.model_dump(by_alias=True, exclude_none=False)
    # 新字段条件注入(仿 isCompactSummary 先例):仅非默认时写,避免普通 user 行的 null 噪声。
    if tool_use_results:
        out["toolUseResult"] = tool_use_results  # dict 按 tool_use_id 索引
    if source_tool_assistant_uuid is not None:
        out["sourceToolAssistantUUID"] = source_tool_assistant_uuid
    if is_task_notification:
        out["isTaskNotification"] = True
    if agent_id is not None:
        out["agentId"] = agent_id
    return out


def encode_assistant(
    msg: Message,
    *,
    uuid: str,
    parent_uuid: str | None,
    ctx: SessionContext,
    timestamp: str,
    usage: TokenUsage | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    m = message_to_dict(msg)
    if usage is not None:
        m["usage"] = usage.model_dump()
    if stop_reason is not None:
        m["stop_reason"] = stop_reason
    line = AssistantLine.model_validate(
        {
            "uuid": uuid,
            "parent_uuid": parent_uuid,
            "session_id": ctx.session_id,
            "timestamp": timestamp,
            "cwd": ctx.cwd,
            "version": ctx.version,
            "git_branch": ctx.git_branch,
            "message": m,
            "synthetic": msg.synthetic,
        }
    )
    return line.model_dump(by_alias=True, exclude_none=False)


def encode_system_compact(
    *,
    subtype: str,
    uuid: str,
    parent_uuid: str | None,
    ctx: SessionContext,
    timestamp: str,
    logical_parent_uuid: str | None = None,
    content: str = "",
    compact_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """编码 system 边界行(compact_boundary)。

    parentUuid=null 主线树在此重启 + logicalParentUuid 续逻辑链。
    compact_metadata 落 trigger/preTokens/postTokens/preservedMessages 等。
    返回 by_alias 的 dict 供 jsonl 直接写。parentUuid 强制传 None(边界行语义)。
    """
    line = SystemLine.model_validate(
        {
            "uuid": uuid,
            "parent_uuid": parent_uuid,
            "logical_parent_uuid": logical_parent_uuid,
            "session_id": ctx.session_id,
            "timestamp": timestamp,
            "cwd": ctx.cwd,
            "version": ctx.version,
            "git_branch": ctx.git_branch,
            "subtype": subtype,
            "content": content,
            "compact_metadata": compact_metadata or {},
        }
    )
    return line.model_dump(by_alias=True, exclude_none=False)


def encode_queue_operation(
    *,
    ctx: SessionContext,
    timestamp: str,
    operation: str,
    agent_id: str,
    tool_use_id: str,
    status: str,
) -> dict[str, Any]:
    """编码 queue-operation 行。by_alias dict 供 jsonl 直接写。

    事件行:不挂 uuid/parentUuid(不进消息 DAG)、不带 content(通知正文由紧随的
    isTaskNotification user 行承载)。
    """
    line = QueueOperationLine.model_validate(
        {
            "session_id": ctx.session_id,
            "timestamp": timestamp,
            "cwd": ctx.cwd,
            "version": ctx.version,
            "git_branch": ctx.git_branch,
            "operation": operation,
            "agent_id": agent_id,
            "tool_use_id": tool_use_id,
            "status": status,
        }
    )
    return line.model_dump(by_alias=True, exclude_none=False)


# ---- 行序列 → list[Turn] ----


def _line_to_message(obj: dict[str, Any]) -> Message | None:
    """obj.message 残缺/非法 → None(decode 跳过,不杀 resume)。

    synthetic 标记从行级字段回填到 Message(占位/收尾合成行可被 UI 跳过、计费识别)。
    """
    raw = obj.get("message")
    msg = message_from_dict(raw) if isinstance(raw, dict) else None
    if msg is not None:
        if obj.get("synthetic"):
            msg.synthetic = True
        # 行级 isTaskNotification 回填到 Message(resume 重放据此跳过,不当用户气泡)。
        if obj.get("isTaskNotification"):
            msg.is_task_notification = True
    return msg


def find_last_boundary_idx(rows: list[dict[str, Any]]) -> int:
    """最后一条 compact_boundary 行的索引;无则 -1。

    边界判定(type=system + subtype=compact_boundary + parentUuid is None)的【单一来源】:
    split_live_segment(resume 取 live 段)与 read_history(取压缩前历史)共用,避免两处
    谓词分叉漂移。parentUuid 非 null 的 system 行不算边界(防御,正常路径不会出现)。
    """
    last = -1
    for i, obj in enumerate(rows):
        if (
            obj.get("type") == "system"
            and obj.get("subtype") == "compact_boundary"
            and obj.get("parentUuid") is None
        ):
            last = i
    return last


def split_live_segment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """resume 用:返回当前活上下文行 =【最后一条 compact_boundary 起(含)→ 末尾】。

    多次 compaction 只留最后一条边界起的段;更早的边界与被摘要替代的 bulk 全丢(它们
    已被那次摘要覆盖,保留只会重复)。无边界 → 返回原 rows(现状兼容)。
    """
    last_boundary_idx = find_last_boundary_idx(rows)
    if last_boundary_idx < 0:
        return rows
    return rows[last_boundary_idx:]


def decode_lines(lines: list[dict[str, Any]]) -> list[Turn]:
    """纯解码(防御):行序列 → list[Turn]。不做末尾修复(见 repair_trailing_edge)。

    Turn 边界:含 TextBlock 的 user 行为起点;tool_result 回填的 user 行归入当前 Turn。
    Turn.usage / Turn.stop_reason 取 Turn 内最后一条 assistant 行的对应字段。残缺行/消息/块
    跳过(只告警),单行损坏不致命;usage schema drift 容忍。
    """
    turns: list[Turn] = []
    current: Turn | None = None
    last_usage: TokenUsage | None = None
    last_stop_reason: str | None = None

    def _is_user_question(msg: Message) -> bool:
        return msg.role == "user" and any(isinstance(b, TextBlock) for b in msg.content)

    for obj in lines:
        if obj.get("type") not in ("user", "assistant"):
            continue
        msg = _line_to_message(obj)
        if msg is None:
            continue  # 残缺消息跳过
        if _is_user_question(msg):
            if current is not None:
                current.usage = last_usage
                current.stop_reason = last_stop_reason
                turns.append(current)
            current = Turn(messages=[msg])
        else:
            if current is None:
                # 首行不是 user 提问(异常但容错):自起一个 Turn
                current = Turn(messages=[msg])
            else:
                current.messages.append(msg)
        if msg.role == "assistant":
            m = obj.get("message")
            if isinstance(m, dict):
                u = m.get("usage")
                if isinstance(u, dict):
                    try:
                        # model_validate 容忍 extra 字段;try 兜住类型漂移/null 不杀解码。
                        last_usage = TokenUsage.model_validate(u)
                    except Exception:
                        log.warning("跳过坏 usage(类型漂移): %s", str(u)[:80])
                sr = m.get("stop_reason")
                if isinstance(sr, str):
                    last_stop_reason = sr
    if current is not None:
        current.usage = last_usage
        current.stop_reason = last_stop_reason
        turns.append(current)
    return turns


def decode_lines_to_turns(lines: list[dict[str, Any]]) -> list[Turn]:
    """便捷入口:纯解码 + 末尾修复(内存,不持久化)。返回 turns。

    持久化合成消息由 SessionStore.load_mainline 负责(再 resume 稳定);此函数仅供
    不解 store 的调用方(如单测)拿到「已修复」的 turns。
    """
    turns = decode_lines(lines)
    repair_trailing_edge(turns)
    return turns


def repair_trailing_edge(
    turns: list[Turn],
    *,
    error_message_fn: Callable[[ToolUseBlock], str] | None = None,
) -> list[Message]:
    """补全末尾 Turn,使主线以 role=assistant 收尾、所有 tool_use 配对。

    原地修改 turns[-1];返回追加的合成 Message(供 store 持久化,使再 resume 稳定)。

    error_message_fn:悬空 tool_use 的合成 error 文案生成器。不传(默认 None)→ 用原
    mainline 默认文案(向后兼容,行为严格不变);侧链续跑传 conservative_sidechain_error_message
    (决策 B:带工具名 + 保守措辞,不催盲目重发)。

    取代旧实现(只补 tool_result / 丢 orphan user)——统一处理三类中断尾部:
      - 悬空 tool_use(assistant 落盘、tool_result 没落盘)→ 补 error tool_result(user)。
      - orphan user 提问(无 assistant 跟随)→ 不丢弃,补合成 assistant 完成它
        (丢弃在 append-only 下会被后续 append 埋成「中段 orphan user」,下轮 API 仍 400)。
      - 末尾 user(tool_result)无后续 assistant(工具回填后崩溃)→ 补合成 assistant。
    步骤 2 统一兜底:末尾若为 user(以上任一),补一条合成 assistant → 主线恒以 assistant
    收尾,下轮 submit 不再出现「连续两条 user」(Anthropic 400)。持久化后 orphan+
    dangling 组合不再复现。
    """
    if not turns:
        return []
    last_turn = turns[-1]
    if not last_turn.messages:
        return []
    synthetics: list[Message] = []

    # 1) 悬空 tool_use → 为每个补 error tool_result(user role,Anthropic 惯例)。
    tool_use_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for m in last_turn.messages:
        for b in m.content:
            if isinstance(b, ToolUseBlock):
                tool_use_ids.add(b.id)
            elif isinstance(b, ToolResultBlock):
                tool_result_ids.add(b.tool_use_id)
    unpaired = tool_use_ids - tool_result_ids
    # 仅当末尾是 assistant(含 tool_use)时补 result。末尾若已 user
    # (罕见残缺)→ 不在此处理,交 base_llm.normalize_messages_for_api 在 flatten 时合并,
    # 避免持久化分歧(旧实现无此守卫,会在 user 后再追加 user 造成中段违规)。
    if unpaired and last_turn.messages[-1].role == "assistant":
        # 按 id 反查 ToolUseBlock,供 error_message_fn 带工具名(决策 B)。
        tu_by_id: dict[str, ToolUseBlock] = {
            b.id: b for m in last_turn.messages for b in m.content if isinstance(b, ToolUseBlock)
        }
        repair_blocks: list[ContentBlock] = [
            ToolResultBlock(
                tool_use_id=tid,
                content=(
                    error_message_fn(tu_by_id[tid])
                    if error_message_fn is not None and tid in tu_by_id
                    else "工具调用因会话中断未完成,请重新发起"
                ),
                is_error=True,
            )
            for tid in sorted(unpaired)
        ]
        m = Message(role="user", content=repair_blocks, synthetic=True)
        last_turn.messages.append(m)
        synthetics.append(m)
        log.warning("repair:%d 个悬空 tool_use,已补 error tool_result", len(unpaired))

    # 2) 末尾若为 user(悬空补完后 / orphan 提问 / 工具回填后崩溃)→ 补合成 assistant,
    #    恢复角色交替(主线恒以 assistant 收尾)。
    if last_turn.messages[-1].role == "user":
        m = Message(
            role="assistant",
            content=[TextBlock(text="(上一轮因会话中断未完成,已自动补全收尾)")],
            synthetic=True,
        )
        last_turn.messages.append(m)
        synthetics.append(m)
        log.warning("repair:末尾以 user 收尾,已补合成 assistant 恢复角色交替")

    return synthetics


def _mark_pending_sync_agent_tooluses(
    turns: list[Turn], sync_agent_types: set[str],
) -> int:
    """把「待续跑 sync agent tool_use」的合成 tool_result content 改为占位文案。

    前置:turns 已过 ``repair_trailing_edge``(末尾 unpaired tool_use 已补合成 error
    tool_result,synthetic=True)。本函数在 repair 之后跑:对识别为「待续跑 sync agent」
    的合成 tool_result,把 content 覆盖为 ``_SYNC_PENDING_PLACEHOLDER``(含「中断/续跑」),
    覆盖 repair 默认「请重新发起」。真结果由 ``resume_agent`` inline 返回(resume 路径),
    故占位只是历史中的临时脚手架。

    识别(关键):主会话 sync agent tool_use 的 id 是 LLM 生成(toolu_xxx),runner 内部
    tool_use_id="(sync-agent)" 只进子 agent meta/queue-op、不进主 jsonl;且中断期真
    tool_result / toolUseResult 均未落盘。故靠 **tool_use.name == 非终态 sync
    meta.agent_type** 配对(sync 阻塞,同时刻仅一个在跑):sync_agent_types 即当前会话
    非终态 sync 子 agent 的 agent_type 集合(由 store.load_mainline 经
    find_resumable_subagents 过滤 is_async=False 算出后传入)。

    只改 synthetic=True 的 tool_result(repair 补的);真实 tool_result 绝不动。
    返回改写条数(0=未命中,观测/日志用)。空 sync_agent_types → 直接返 0(无非终态
    sync meta 时零成本短路)。
    """
    if not turns or not sync_agent_types:
        return 0
    last_turn = turns[-1]
    if not last_turn.messages:
        return 0
    # tool_use_id → name 映射:用 tool_use.name 配对 sync agent 类型。
    tu_name_by_id: dict[str, str] = {
        b.id: b.name
        for m in last_turn.messages
        for b in m.content
        if isinstance(b, ToolUseBlock)
    }
    changed = 0
    for m in last_turn.messages:
        if not m.synthetic or m.role != "user":
            continue
        for b in m.content:
            if not isinstance(b, ToolResultBlock):
                continue
            name = tu_name_by_id.get(b.tool_use_id)
            if name is not None and name in sync_agent_types:
                b.content = _SYNC_PENDING_PLACEHOLDER
                changed += 1
    if changed:
        log.info(
            "sync-agent 占位:%d 个待续跑 tool_use 的合成 tool_result 改为占位文案", changed,
        )
    return changed


def find_pending_notifications(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """识别 pending 的异步子 agent 完成通知(resume 规则)。

    pending = queue-operation(operation=enqueue) 且无匹配 agentId 的【注入 user 行
    或 dequeue/remove】(resolved)。resume 时对这些重新注入模型上下文。坏行(非 dict /
    缺 agentId 的 enqueue)安全跳过——返回项必带 truthy agentId,调用方 obj["agentId"] 安全。

    把 dequeue(消费标记)/ remove(取消)也并入 resolved —— 否则已消费/取消的 enqueue
    仍被当 pending,在 resume 时误重构(重复注入)。find_unresponded_notifications 处理
    「已投递未响应」轴,与本函数(投递补全轴)正交。

    injected/resolved 均为全局集合(非「后续 enqueue」):靠 agentId 每 spawn 唯一保证等价;
    若未来放宽 agentId 复用,需改为顺序扫描。
    """
    injected_agent_ids: set[str] = {
        obj["agentId"]
        for obj in rows
        if isinstance(obj, dict)
        and obj.get("type") == "user"
        and obj.get("isTaskNotification")
        and obj.get("agentId")
    }
    resolved_agent_ids: set[str] = {
        obj["agentId"]
        for obj in rows
        if isinstance(obj, dict)
        and obj.get("type") == "queue-operation"
        and obj.get("operation") in ("dequeue", "remove")
        and obj.get("agentId")
    }
    return [
        obj
        for obj in rows
        if isinstance(obj, dict)
        and obj.get("type") == "queue-operation"
        and obj.get("operation") == "enqueue"
        and obj.get("agentId")  # 过滤坏行(缺/None agentId):docstring 承诺「安全跳过」
        and obj.get("agentId") not in injected_agent_ids
        and obj.get("agentId") not in resolved_agent_ids
    ]


# ---- Phase2: find_unresponded_notifications(已投递但未消费)----


@dataclass(slots=True)
class UnrespondedNotification:
    """一条待主 agent 响应的异步完成通知(已投递、未消费)。"""

    agent_id: str
    status: str  # completed/error/cancelled,取自 enqueue(供 dequeue 复用)
    message: Message  # 解码后的 isTaskNotification user 行


def find_unresponded_notifications(rows: list[dict[str, Any]]) -> list[UnrespondedNotification]:
    """已投递(isTaskNotification user 行)但无匹配 dequeue 的通知 → 待主 agent 响应。

    与 find_pending_notifications(投递补全:enqueue 无 user 行)拆清两轴:
    - find_pending:crash 在 enqueue↔user 行之间,补投。
    - find_unresponded:已投递但主 agent 未响应(consume 标记 = dequeue)。
    统一运行时 wake 与 resume 重注入(两条路都调它)。status 取自同 agentId 的 enqueue 行
    (dequeue 复用,QueueOperationLine.status 必填)。
    """
    # 已消费/已取消的 agentId:dequeue(消费标记)或 remove(取消)。
    # 与 find_pending_notifications 的 resolved 集合对齐(两者都算 resolved)。
    resolved = {
        obj["agentId"]
        for obj in rows
        if isinstance(obj, dict)
        and obj.get("type") == "queue-operation"
        and obj.get("operation") in ("dequeue", "remove")
        and obj.get("agentId")
    }
    enqueue_status: dict[str, str] = {
        # `or "completed"` 兜底:status 显式为 null(坏数据/老会话)时不穿透 None(违 status:str 契约)
        obj["agentId"]: obj.get("status") or "completed"
        for obj in rows
        if isinstance(obj, dict)
        and obj.get("type") == "queue-operation"
        and obj.get("operation") == "enqueue"
        and obj.get("agentId")
    }
    out: list[UnrespondedNotification] = []
    for obj in rows:
        if not (
            isinstance(obj, dict)
            and obj.get("type") == "user"
            and obj.get("isTaskNotification")
            and obj.get("agentId")
            and obj["agentId"] not in resolved
        ):
            continue
        msg = _line_to_message(obj)
        if msg is None:
            continue
        out.append(
            UnrespondedNotification(
                agent_id=obj["agentId"],
                status=enqueue_status.get(obj["agentId"], "completed"),
                message=msg,
            )
        )
    return out


# ---- 子 agent 侧链续跑:保守 error 文案 + 侧链加载 ----


def conservative_sidechain_error_message(tu: ToolUseBlock) -> str:
    """决策 B:侧链续跑时,末尾 unpaired tool_use 的合成 error 文案。

    保守:不催"重新发起"(续跑是无人监督自动接着跑,盲目重发对非幂等 bash 危险);
    改为"结果未知 + 先验状态 + 勿盲目重试副作用操作",并带工具名让子 agent 知道验什么。
    """
    name = tu.name or "工具"
    return (
        f"{name} 调用因进程中断而结果未知(可能已执行、可能未执行)。"
        f"请先确认相关状态后再决定是否重试——对有副作用的操作"
        f"(如 bash 写入/提交/网络请求)勿盲目重复。"
    )


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """逐行 jsonl → list[dict]:跳空行/坏 JSON/非 dict/半截行(断电 mid-flush)。

    与 store._read_mainline_rows 不同:此处不按 isSidechain 过滤(本就是读单条侧链文件)。
    容错半截行:json.loads 抛错即跳过,不抛给调用方(对齐主会话行读取策略)。
    """
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # 半截行(断电 mid-flush)跳过,不抛
            if isinstance(obj, dict):
                out.append(obj)
    return out


def load_sidechain_turns(sidechain_path: Path) -> list[Turn]:
    """读子 agent 侧链 jsonl → split_live_segment + decode_lines + repair(决策 B 文案)。

    与主会话 load_mainline 同款管线,但 repair 用保守文案(续跑场景)。
    返回的 turns 末尾恒以 assistant 收尾、tool_use 全配对,可直接作 history 喂 run_agent_loop。
    文件不存在/全坏行 → [](供调用方容错:无侧链 = 空历史)。

    #15837 防御:侧链 jsonl 非空(raw_rows 有行)但 decode 后 turns 空(矛盾状态)→ 只记
    warning 不抛。合法空(文件缺/全坏行)不触发——raw_rows 为空即非矛盾。
    """
    raw_rows = _read_jsonl_rows(sidechain_path)
    rows = split_live_segment(raw_rows)
    turns = decode_lines(rows)
    repair_trailing_edge(turns, error_message_fn=conservative_sidechain_error_message)
    if not turns and raw_rows:
        log.warning(
            "load_sidechain_turns: 侧链 jsonl 非空但 turns 为空,疑似加载截断(#15837 防御)"
        )
    return turns
