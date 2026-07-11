# src/birdcode/session/models.py
"""持久化行模型(Pydantic v2):jsonl 每行一个 JSON 对象的强类型表示。

内存热路径用 dataclass(blocks.py/conversation.py),持久化层用 Pydantic。
翻译层(codec.py)在两者间转换。jsonl 用 CC 的 camelCase 字段名(by_alias)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _LineBase(BaseModel):
    """每行公共字段。extra='allow' 前向兼容(未来加字段不破坏旧读取)。"""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["user", "assistant", "system"]
    uuid: str
    parent_uuid: str | None = Field(None, alias="parentUuid")
    session_id: str = Field(..., alias="sessionId")
    timestamp: str
    cwd: str = ""
    version: str = ""
    git_branch: str | None = Field(None, alias="gitBranch")
    is_sidechain: bool = Field(False, alias="isSidechain")
    # compaction 边界行专用:parentUuid=null(树重启)时,靠 logical_parent_uuid 续逻辑链。
    # 仅 system 边界行会填,其余行保持 None。
    logical_parent_uuid: str | None = Field(None, alias="logicalParentUuid")
    # resume 末尾修复补的合成行(占位 tool_result / 收尾 assistant)。BirdCode 自有字段
    # (CC 格式无此字段),无 alias——字段名即 snake_case,by_alias dump 仍为 "synthetic"。
    synthetic: bool = False


class UserLine(_LineBase):
    type: Literal["user"] = "user"
    message: dict[str, Any]


class AssistantLine(_LineBase):
    type: Literal["assistant"] = "assistant"
    message: dict[str, Any]


class SystemLine(_LineBase):
    """system 行。Phase 1 只用 subtype=compact_boundary(压缩边界);extra='allow' 兼容未来 subtype。

    compact_boundary 行:parentUuid=null、logicalParentUuid 指向压缩前最后一条 uuid、
    compact_metadata 记 trigger/preTokens/postTokens/preservedSegment/preservedMessages。
    """

    type: Literal["system"] = "system"
    subtype: str = ""
    content: str = ""
    level: str = "info"
    compact_metadata: dict[str, Any] = Field(default_factory=dict, alias="compactMetadata")


class AgentToolUseResult(BaseModel):
    """Agent 工具的顶层 toolUseResult 形状。通用 toolUseResult 机制的首个实例。

    is_async=false(同步):content=最终结果、output_file=None。
    is_async=true(异步回执):content=None、output_file=结果路径。
    完成统计(同步完成/异步完成)填 total_*。不进 message.content / 不发 LLM。
    """

    model_config = ConfigDict(populate_by_name=True)

    agent_id: str = Field(..., alias="agentId")
    agent_type: str = Field("", alias="agentType")
    status: Literal["launched", "running", "completed", "error", "cancelled"]
    is_async: bool = Field(False, alias="isAsync")
    prompt: str = ""
    resolved_model: str | None = Field(None, alias="resolvedModel")
    content: str | None = None
    is_completed: bool = Field(True, alias="isCompleted")
    output_file: str | None = Field(None, alias="outputFile")
    can_read_output_file: bool = Field(False, alias="canReadOutputFile")
    total_duration_ms: int | None = Field(None, alias="totalDurationMs")
    total_tokens: int | None = Field(None, alias="totalTokens")
    total_tool_use_count: int | None = Field(None, alias="totalToolUseCount")
    usage: dict[str, Any] | None = None
    tool_stats: dict[str, Any] | None = Field(None, alias="toolStats")


class QueueOperationLine(BaseModel):
    """queue-operation 行:异步子 agent 完成事件。

    这是【事件行,非对话内容】:不挂 uuid/parentUuid(不进消息 DAG)、不带 content
    (通知正文由紧随的 isTaskNotification user 行承载,二者不再重复)。每条自带
    agentId+toolUseId+status,不依赖外部上下文即可辨识(多异步)。
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["queue-operation"] = "queue-operation"
    session_id: str = Field(..., alias="sessionId")
    timestamp: str
    cwd: str = ""
    version: str = ""
    git_branch: str | None = Field(None, alias="gitBranch")
    operation: Literal["enqueue", "dequeue", "remove"] = "enqueue"
    agent_id: str = Field(..., alias="agentId")
    tool_use_id: str = Field(..., alias="toolUseId")
    status: Literal["completed", "error", "cancelled"] = "completed"


class SubagentMeta(BaseModel):
    """子 agent 元数据(enriched)。agent-<id>.meta.json,唯一可变文件
    (状态/计时随生命周期更新;jsonl 严格 append-only)。/agents 扫此文件即得状态。"""

    # extra='allow':前向兼容——这是唯一可变文件(读→改状态→写),未来加字段时旧构建
    # model_validate 会丢未知字段、write(exclude_none)再擦除。allow 保留未知字段(与
    # QueueOperationLine 一致),读-改-写不丢未来字段。
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    agent_id: str = Field(..., alias="agentId")
    agent_type: str = Field("general-purpose", alias="agentType")
    description: str = ""
    tool_use_id: str = Field(..., alias="toolUseId")
    spawn_depth: int = Field(1, alias="spawnDepth")
    is_async: bool = Field(False, alias="isAsync")
    model: str = ""
    resolved_model: str | None = Field(None, alias="resolvedModel")
    status: Literal["launched", "running", "completed", "error", "cancelled"] = "launched"
    spawned_at: str = Field("", alias="spawnedAt")
    completed_at: str | None = Field(None, alias="completedAt")
    total_duration_ms: int | None = Field(None, alias="totalDurationMs")
    total_tokens: int | None = Field(None, alias="totalTokens")


@dataclass
class SessionContext:
    """一次会话的公共字段(append 时复用)。"""

    session_id: str
    cwd: str
    version: str
    git_branch: str | None


class PersistInfo(BaseModel):
    """persist_tool_output 返回:给 LLM 的占位文本 + 落盘路径/大小。"""

    llm_content: str
    path: str
    size: int


class SessionSummary(BaseModel):
    """list_sessions 返回:单条历史会话条目。

    first_user_summary / turn_count 来自 sidecar(<sid>.meta,append 时增量写),
    list_sessions 不扫 jsonl → O(文件数) 而非 O(总字节)。sidecar 缺/坏时 lazy
    backfill(扫一次补建,见 store._read_or_backfill_meta)。mtime 仍用 jsonl 的 stat。
    """

    session_id: str
    mtime: float
    first_user_summary: str
    turn_count: int
