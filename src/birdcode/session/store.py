# src/birdcode/session/store.py
"""SessionStore:append-only jsonl 读写 + 双轨外存落盘 + 历史会话列表。

uuid 链状态封装在 store 内部(_last_uuid):append 自动挂链,load 设为叶子。
"""
from __future__ import annotations

import json
import uuid as _uuid
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from birdcode.blocks import TextBlock
from birdcode.conversation import Message, Turn
from birdcode.session import codec, paths
from birdcode.session.jsonutil import read_json_dict
from birdcode.session.models import PersistInfo, SessionContext, SessionSummary
from birdcode.session.timeutil import utc_iso
from birdcode.utils.logging import get_logger
from birdcode.utils.worktree import worktree_store_name

if TYPE_CHECKING:
    from birdcode.agent.provider import TokenUsage

log = get_logger("birdcode.session")

_PREVIEW_CHARS = 2048
_META_SUFFIX = ".meta"  # sidecar 索引,与 <sid>.jsonl 同目录同 stem
_FIRST_USER_MAX = 60  # first_user 截断长度(与旧 _summarize_jsonl 一致)


def write_jsonl_line(fh: IO[str], line: dict[str, Any], *, label: str) -> bool:
    """写一行 jsonl(write+flush)。IO/序列化失败只 log、返回 False(不杀主循环)。

    统一 SessionStore / SubagentStore 的写入 + 错误处理(原 5 处重复块):
    - OSError:磁盘满 / 权限 / 句柄已关闭。
    - ValueError:含 Pydantic ValidationError 子类(理论 encode 阶段已校验;防御)。
    - TypeError:line 含不可 JSON 序列化对象(如调用方误传 Pydantic model 未 model_dump)
      ——原 except (OSError, ValueError) 漏掉,会杀主循环。
    成功返回 True;调用方据此决定是否推进 _last_uuid。
    """
    try:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        fh.flush()
    except (OSError, ValueError, TypeError):
        log.exception("%s 失败(不杀主循环)", label)
        return False
    return True


def _persisted_output_placeholder(path: str, size: int, full_output: str) -> str:
    kb = size / 1024
    preview = full_output[:_PREVIEW_CHARS]
    return (
        f"<persisted-output>\n"
        f"Output too large ({kb:.1f}KB). Full output saved to: {path}\n\n"
        f"Preview (first {_PREVIEW_CHARS} chars):\n{preview}"
    )


# ---- sidecar 索引(<sid>.meta):/sessions 快 + 可读,不扫 jsonl ----


def _meta_path(jsonl_path: Path) -> Path:
    """<sid>.jsonl → <sid>.meta(sidecar 索引,同目录同 stem)。"""
    return jsonl_path.with_suffix(_META_SUFFIX)


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    """读 sidecar;缺/坏(非 JSON / 非 dict)→ None(调用方 lazy backfill)。"""
    return read_json_dict(meta_path)


def _first_text(msg: Message) -> str:
    """取首条非空 TextBlock 文本;无则 ""(tool_result 回填等→"",不计入提问)。"""
    for b in msg.content:
        if isinstance(b, TextBlock) and b.text:
            return b.text
    return ""


def _count_first_user_and_turns_from_rows(rows: list[dict[str, Any]]) -> tuple[str, int]:
    """从行列表算 (first_user, turn_count):user+text 行算一轮。

    排除系统注入行:isTaskNotification/isCompactSummary(压缩摘要)——
    它们是 user+text 但非真实提问。判定与运行时 _first_text 一致。
    """
    first_user = ""
    turn_count = 0
    for obj in rows:
        if not isinstance(obj, dict) or obj.get("type") != "user":
            continue
        if (
            obj.get("isSidechain")
            or obj.get("isTaskNotification")
            or obj.get("isCompactSummary")
        ):
            continue
        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue  # content=null/非列表(坏行):防御(对齐旧 message_from_dict)
        has_text = any(
            isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            for b in content
        )
        if not has_text:
            continue
        turn_count += 1
        if not first_user:
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                    first_user = str(b["text"])[:_FIRST_USER_MAX]
                    break
    return first_user, turn_count


def _scan_first_user_and_turns(jf: Path) -> tuple[str, int]:
    """扫 jsonl 算 (first_user, turn_count)——仅 lazy backfill 用(sidecar 缺/坏时)。

    委托 _count_first_user_and_turns_from_rows(排除 isTaskNotification/isCompactSummary)。
    """
    rows: list[dict[str, Any]] = []
    with jf.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return _count_first_user_and_turns_from_rows(rows)


def _read_or_backfill_meta(jf: Path) -> tuple[str, int]:
    """读 <sid>.meta;缺/坏 → 扫一次 jsonl 补建(老会话的一次性 lazy backfill)。

    返回 (first_user, turn_count)。backfill 顺带落盘 meta,下次 O(1)。写失败只 log,
    返回值仍正确(下次 list 再补)。
    """
    data = _read_meta(_meta_path(jf))
    if data is None:
        first_user, turn_count = _scan_first_user_and_turns(jf)
        try:
            payload = json.dumps(
                {"first_user": first_user, "turn_count": turn_count}, ensure_ascii=False
            )
            _meta_path(jf).write_text(payload, encoding="utf-8")
        except OSError:
            log.debug("backfill meta 写失败(返回值仍正确,下次再试)", exc_info=True)
        return first_user, turn_count
    return str(data.get("first_user", "")), int(data.get("turn_count", 0))


class OutputSink:
    """executor 持有的轻协议对象,闭包绑定 store。

    executor._maybe_persist 调 sink.persist(tool_use_id, full_output);
    超阈落盘返回 PersistInfo(含给 LLM 的占位),不超阈返回 None。
    """

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def persist(self, tool_use_id: str, full_output: str) -> PersistInfo | None:
        return await self._store.persist_tool_output(tool_use_id, full_output)


class SessionStore:
    def __init__(
        self,
        ctx: SessionContext,
        project_root: Path,
        *,
        root: Path | None = None,
    ) -> None:
        self._ctx = ctx
        self._root = root if root is not None else paths.default_root()
        self._project_root = project_root
        # per-worktree 隔离:从 ctx.cwd 推 worktree 分段名(worktree/<name>/ 或主仓目录)。
        # 子 agent jsonl 经 SubagentStore 同样推导 → 与会话 jsonl 落同一 worktree/<name>/。
        self._worktree_name = worktree_store_name(ctx.cwd)
        self._jsonl = paths.session_jsonl(
            self._root, ctx.session_id, project_root, worktree_name=self._worktree_name
        )
        self._tool_results = paths.tool_results_dir(
            self._root, ctx.session_id, project_root, worktree_name=self._worktree_name
        )
        # 惰性打开:不在构造时建目录/文件 → 空会话(启动即 Ctrl+Q、无任何内容)不落 0kb jsonl。
        # 首次写入经 _fh property 才 mkdir+open('a')。close()/各 append 均经 self._fh 取句柄。
        self._fh_io: IO[str] | None = None
        self._last_uuid: str | None = None
        # 启动时清扫本项目目录下遗留的 0kb 空会话(旧版 __init__ 曾 open('a') 占位创建)。
        # 0kb 文件无内容、list_sessions 本就忽略,删除安全(含并发会话:0kb 即未写入,无可丢内容)。
        self._purge_empty_sessions()

    def as_output_sink(self) -> OutputSink:
        return OutputSink(self)

    @property
    def _fh(self) -> IO[str]:
        """惰性打开 jsonl 句柄:首次写入才 mkdir + open('a')。

        空会话(从未 append)→ 永不建文件,不留 0kb jsonl。各 append/append_compaction
        均经 self._fh 取句柄;close() 直读 _fh_io(避免在此触发建文件)。
        """
        if self._fh_io is None:
            self._jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._fh_io = self._jsonl.open("a", encoding="utf-8")
        return self._fh_io

    def _purge_empty_sessions(self) -> None:
        """清扫本项目会话目录下的 0kb jsonl(遗留空会话)。best-effort,失败只 log。"""
        d = paths.sessions_dir(
            self._root, self._project_root, worktree_name=self._worktree_name
        )
        if not d.exists():
            return
        try:
            for jf in d.glob("*.jsonl"):
                try:
                    if jf.stat().st_size <= 0:
                        jf.unlink()
                except OSError:
                    log.debug("删除空会话 %s 失败", jf, exc_info=True)
        except OSError:
            log.debug("purge empty sessions 失败", exc_info=True)

    @property
    def tool_results_dir(self) -> Path:
        """当前 session 的工具结果落盘目录(供 gate extra_roots / executor 双轨外存)。

        公开访问器:避免 ui 层直读 store._tool_results 私有属性。
        """
        return self._tool_results

    @property
    def jsonl_path(self) -> Path:
        """当前 session 的主线 jsonl 路径(供 read_history 等只读当前会话记录)。

        公开访问器:避免工具层直读 store._jsonl 私有属性。read_history 仅读此文件,
        不扫会话目录、不接受路径参数 —— 结构上保证只读当前会话。
        """
        return self._jsonl

    @property
    def ctx(self) -> SessionContext:
        """当前会话的 SessionContext(供 AgentTool 派生子 agent 时复用 sessionId/cwd 等)。

        公开访问器:避免 tools 层直读 store._ctx 私有属性。
        """
        return self._ctx

    @property
    def root(self) -> Path:
        """会话存储根(<root>/<encoded-cwd>/...)。公开访问器:供侧链路径派生等只读用途。"""
        return self._root

    @property
    def project_root(self) -> Path:
        """项目根(用于派生 encoded-cwd 子目录)。公开访问器:避免直读 store._project_root。"""
        return self._project_root

    @property
    def worktree_name(self) -> str | None:
        """worktree 存储分段名(None=主仓目录,'<name>'=worktree/<name>/)。公开访问器。"""
        return self._worktree_name

    def reopen(self, new_session_id: str) -> SessionStore:
        """开新 sessionId(复用 root/project_root/cwd/version/git_branch),旧文件保留。/clear 用。

        异常安全——先构造新 store,成功后再 close 旧。若新 __init__ 抛异常,
        旧 fh 仍开,调用方持有的旧 store 仍可 append(不变成“已关壳”)。
        """
        new_ctx = SessionContext(
            session_id=new_session_id,
            cwd=self._ctx.cwd,
            version=self._ctx.version,
            git_branch=self._ctx.git_branch,
        )
        new = SessionStore(new_ctx, self._project_root, root=self._root)
        self.close()
        return new

    async def append(
        self,
        msg: Message,
        *,
        is_assistant: bool = False,
        usage: TokenUsage | None = None,
        stop_reason: str | None = None,
        tool_use_results: dict[str, Any] | None = None,
        source_tool_assistant_uuid: str | None = None,
        is_task_notification: bool = False,
        agent_id: str | None = None,
    ) -> str:
        """追加一行。返回本行 uuid;链自动挂(内部 _last_uuid)。IO 失败只 log。

        新参:tool_use_results(顶层 toolUseResult dict-by-id)、
        source_tool_assistant_uuid、is_task_notification(+agent_id)。is_task_notification
        行不计入 sidecar turn_count(系统注入,非真实提问)。
        """
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
                msg, uuid=new_uuid, parent_uuid=parent_uuid, ctx=self._ctx,
                timestamp=timestamp, tool_use_results=tool_use_results,
                source_tool_assistant_uuid=source_tool_assistant_uuid,
                is_task_notification=is_task_notification, agent_id=agent_id,
            )
        if not write_jsonl_line(self._fh, line, label="session append"):
            return new_uuid
        self._last_uuid = new_uuid
        # sidecar 增量:user 文本提问(非 tool_result 回填、非 task-notification 注入)
        # → 更新 <sid>.meta。
        if not is_assistant and not is_task_notification:
            text = _first_text(msg)
            if text:
                self._update_meta_on_prompt(text)
        return new_uuid

    async def append_queue_operation(
        self,
        *,
        operation: str,
        agent_id: str,
        tool_use_id: str,
        status: str,
    ) -> None:
        """追加 queue-operation 行。IO 失败只 log,不杀主循环。

        事件行:不生成 uuid、不挂/不推进 _last_uuid(对消息 DAG 透明)——下一条对话行
        仍链到本事件行之前的最后对话节点。content 不再落此行(通知正文由紧随的
        isTaskNotification user 行承载,避免重复)。
        """
        timestamp = utc_iso()
        try:
            line = codec.encode_queue_operation(
                ctx=self._ctx, timestamp=timestamp,
                operation=operation, agent_id=agent_id, tool_use_id=tool_use_id,
                status=status,
            )
        except ValueError:
            # encode 的 QueueOperationLine.model_validate 抛 ValidationError(ValueError 子类,
            # 非法 operation/status)——原 encode 在 try 外,会杀主循环。
            log.exception("append_queue_operation encode 失败(不杀主循环)")
            return
        write_jsonl_line(self._fh, line, label="append_queue_operation")

    async def append_system_compact(
        self,
        *,
        subtype: str,
        logical_parent_uuid: str | None,
        content: str,
        compact_metadata: dict[str, Any],
    ) -> str:
        """追加 system 边界行(compact_boundary)。

        parentUuid=null 重启主线树、logicalParentUuid 续逻辑链(跨边界可追溯)。
        链不挂 _last_uuid 的常规父——边界行本身就是新链根;但推进 _last_uuid 到本行,
        使紧随的摘要行 append 时 parent = 本边界行 uuid。IO 失败只 log,不杀主循环。
        """
        new_uuid = str(_uuid.uuid4())
        timestamp = utc_iso()
        line = codec.encode_system_compact(
            subtype=subtype, uuid=new_uuid, parent_uuid=None, ctx=self._ctx,
            timestamp=timestamp, logical_parent_uuid=logical_parent_uuid,
            content=content, compact_metadata=compact_metadata,
        )
        if not write_jsonl_line(self._fh, line, label="append_system_compact"):
            return new_uuid
        self._last_uuid = new_uuid
        return new_uuid

    async def append_compact_summary(self, msg: Message) -> str:
        """追加摘要 user 行(parentUuid = 当前 _last_uuid = 边界行),带 isCompactSummary=True。

        复用 encode_user 的 message 结构,额外置 isCompactSummary 标记(CC 式,供 UI/解码识别)。
        IO 失败只 log,不杀主循环。
        """
        new_uuid = str(_uuid.uuid4())
        parent_uuid = self._last_uuid  # = 刚 append 的边界行 uuid
        timestamp = utc_iso()
        line = codec.encode_user(
            msg, uuid=new_uuid, parent_uuid=parent_uuid, ctx=self._ctx, timestamp=timestamp,
        )
        line["isCompactSummary"] = True
        if not write_jsonl_line(self._fh, line, label="append_compact_summary"):
            return new_uuid
        self._last_uuid = new_uuid
        # 不更新 sidecar:摘要是系统注入的 user 行,不是真实用户提问;first_user/turn_count
        # 应反映真实 prompts(load_mainline 的 backfill 同理,Phase 2 再校准 isCompactSummary 过滤)。
        return new_uuid

    async def append_compaction(
        self,
        *,
        subtype: str,
        logical_parent_uuid: str | None,
        content: str,
        compact_metadata: dict[str, Any],
        summary_msg: Message,
        tail_turns: list[Turn],
    ) -> tuple[str, str] | None:
        """原子写【边界 + 摘要 + 重写尾段】,一次 write+flush。

        返回 (boundary_uuid, summary_uuid)。一次 compaction 的全部产物拼成一段、单次 flush,
        修三类问题:
        - **尾段重写**:尾段 turns 原样重写到边界之后(新 uuid、从 summary 链式挂父)→
          resume 时 split_live_segment 取「最后边界起」即含尾段,装回[摘要+尾段],
          不再丢尾段。
        - **原子性**:boundary/summary/tail 同生共死,不会「只有边界没摘要」→ resume 变空。
        - **不返幻影**:落盘失败返 None(不推进 _last_uuid),调用方放弃持久化、内存仍正确。

        每 turn 的 usage 挂到该 turn 最后一条 assistant 行(decode 据此还原 Turn.usage)。
        """
        boundary_uuid = str(_uuid.uuid4())
        summary_uuid = str(_uuid.uuid4())
        timestamp = utc_iso()
        boundary_line = codec.encode_system_compact(
            subtype=subtype, uuid=boundary_uuid, parent_uuid=None, ctx=self._ctx,
            timestamp=timestamp, logical_parent_uuid=logical_parent_uuid,
            content=content, compact_metadata=compact_metadata,
        )
        summary_line = codec.encode_user(
            summary_msg, uuid=summary_uuid, parent_uuid=boundary_uuid, ctx=self._ctx,
            timestamp=timestamp,
        )
        summary_line["isCompactSummary"] = True
        lines: list[dict[str, Any]] = [boundary_line, summary_line]
        prev_uuid: str = summary_uuid
        for turn in tail_turns:
            # 该 turn 最后一条 assistant 行的索引(挂 usage;decode 取 Turn 内末条 assistant 的 usage)
            last_asst_idx = max(
                (i for i, m in enumerate(turn.messages) if m.role == "assistant"), default=-1
            )
            for i, msg in enumerate(turn.messages):
                row_uuid = str(_uuid.uuid4())
                if msg.role == "assistant":
                    line = codec.encode_assistant(
                        msg, uuid=row_uuid, parent_uuid=prev_uuid, ctx=self._ctx,
                        timestamp=timestamp, usage=(turn.usage if i == last_asst_idx else None),
                    )
                else:
                    line = codec.encode_user(
                        msg, uuid=row_uuid, parent_uuid=prev_uuid,
                        ctx=self._ctx, timestamp=timestamp,
                    )
                lines.append(line)
                prev_uuid = row_uuid
        try:
            # blob 构造(json.dumps)与 write 都在 try 内:TypeError 兜底对
            # 序列化路径才真正生效——旧实现 blob 在 try 外,compact_metadata/尾段含不可
            # 序列化对象时 TypeError 逃出 except,经 context._persist_compaction(无 try)
            # 杀主循环。现与落盘失败同处理(返 None、不推进 _last_uuid、不写半截)。
            blob = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lines)
            self._fh.write(blob)
            self._fh.flush()
        except (OSError, ValueError, TypeError):
            log.exception("append_compaction 落盘失败(不杀主循环,本次不持久化)")
            return None
        self._last_uuid = prev_uuid  # 末行:尾段最后一条;tail 空 时 = summary_uuid
        return boundary_uuid, summary_uuid

    def _update_meta_on_prompt(self, text: str) -> None:
        """user 文本提问 → 增量更新 <sid>.meta(turn_count+1;首条记 first_user)。

        read-modify-write:prompts 低频,读写可忽略。
        """
        data = _read_meta(_meta_path(self._jsonl)) or {"first_user": "", "turn_count": 0}
        if not data.get("first_user"):
            data["first_user"] = text[:_FIRST_USER_MAX]
        data["turn_count"] = int(data.get("turn_count", 0)) + 1
        try:
            _meta_path(self._jsonl).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            log.debug("sidecar meta 写失败(不影响 jsonl)", exc_info=True)

    async def load_mainline(self) -> list[Turn]:
        """读 jsonl 主线 + 末尾修复 + 持久化合成消息。

        本轮单链(无 fork/子agent),文件序 = 主线序(反向追溯遇 parentUuid 缺失会
        丢弃更早行;文件序兜底等价但不丢)。叶子 = 文件最后一条主线行;_last_uuid 设为
        叶子,使 resume 后续 append 接续。

        末尾修复的合成消息(error tool_result / 合成 assistant)写回 jsonl,使再次
        resume 稳定。否则悬空 tool_use / orphan user 会被后续 append 埋到中段,下次 resume
        仅查末尾时漏掉 → API 400。修复逻辑见 codec.repair_trailing_edge。
        """
        all_rows = self._read_mainline_rows()
        rows = codec.split_live_segment(all_rows)  # compaction:只装最后边界起的[摘要+尾段]
        if not rows:
            return []
        # 叶子 = 最后一条【带 uuid 的对话节点】;跳过无 uuid 的事件行(queue-operation:
        # 事件行不进 DAG,见 append_queue_operation)。否则 trailing dequeue(唤醒消费标记,
        # 每次异步完成后都是末行)会让 _last_uuid 取到 None → 下行 parentUuid=null 误断链。
        self._last_uuid = next((r.get("uuid") for r in reversed(rows) if r.get("uuid")), None)
        turns = codec.decode_lines(rows)
        synthetics = codec.repair_trailing_edge(turns)
        # sync 中断占位:待续跑 sync agent tool_use 的合成 tool_result content 改为
        # 「中断,待续跑」(真结果经 resume_agent inline 返回)。在持久化前跑,使占位写入
        # jsonl(再 resume 稳定)。仅命中「非终态 sync meta 的 agent_type」对应的 tool_use,
        # 其他(正常完成/async/无关 tool_use)零影响。
        codec._mark_pending_sync_agent_tooluses(turns, self._pending_sync_agent_types())
        # 持久化合成消息(使再 resume 稳定)。best-effort:append 自身已吞
        # OSError/ValueError;此 try 兜意外异常——放弃持久化自愈,但内存 turns 仍正确,
        # base_llm.normalize 在 flatten 时兜底,不杀 resume。下次 load 会再修(idempotent)。
        for msg in synthetics:
            try:
                await self.append(msg, is_assistant=(msg.role == "assistant"))
            except Exception:
                log.exception("持久化合成消息失败,放弃(内存 turns 仍可用)")
                break
        # sidecar backfill:仅 sidecar 缺/坏时补——不覆盖已有正确值。否则压缩后
        # resume 会用 live 段计数覆盖掉压缩前 sidecar 的完整计数,/sessions 永久少计。
        # 用 all_rows(完整主线)计数,而非 split 后的 live 段(只剩[摘要+尾段],少计前段)。
        if _read_meta(_meta_path(self._jsonl)) is None:
            first_user, turn_count = _count_first_user_and_turns_from_rows(all_rows)
            try:
                _meta_path(self._jsonl).write_text(
                    json.dumps(
                        {"first_user": first_user, "turn_count": turn_count},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                log.debug("load_mainline backfill meta 写失败(不影响 resume)", exc_info=True)
        return turns

    def _pending_sync_agent_types(self) -> set[str]:
        """当前会话【非终态 sync 子 agent】的 agent_type 集合(供 sync 占位配对)。

        复用 agents.discover.find_resumable_subagents(状态非终态 ∪ cancelled),再过滤
        is_async=False(sync)。函数级 import:避免会话层模块加载期反向依赖 agents 层。
        无 subagents 目录 /无非终态 sync meta → 空集(占位短路,零影响)。

        配对依据:主会话 sync agent tool_use 的 id 是 LLM 生成,无法直联 agent_id;
        但 tool_use.name == meta.agent_type(_AgentTool 以 defn.name 注册),故按类型名配对。
        """
        from birdcode.agents.discover import find_resumable_subagents

        sub_dir = paths.subagents_dir(
            self._root, self._ctx.session_id, self._project_root,
            worktree_name=self._worktree_name,
        )
        return {
            m.agent_type
            for m in find_resumable_subagents(sub_dir)
            if not m.is_async
        }

    def _read_mainline_rows(self) -> list[dict[str, Any]]:
        """读 jsonl 主线行:跳过空/坏 JSON/非 dict/isSidechain 行。"""
        if not self._jsonl.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self._jsonl.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("跳过半截坏行: %s", raw[:80])
                    continue
                # 有效 JSON 但非 dict(如 list/str/int)→ 跳过(否则后续 .get 抛 AttributeError)。
                if not isinstance(obj, dict):
                    continue
                if obj.get("isSidechain"):
                    continue
                rows.append(obj)
        return rows

    def mainline_rows(self) -> list[dict[str, Any]]:
        """主线行(跳过 sidechain)的只读副本,供 _process_wake/resume 喂 codec 辅助。

        包装 _read_mainline_rows(每次重读文件、构造新 list,天然是只读副本)。
        Phase2:为 codec.find_pending/find_unresponded 等辅助函数提供行级视图,
        避免调用方直读 store._jsonl 私有文件。
        """
        return self._read_mainline_rows()

    @staticmethod
    def list_sessions(
        root: Path, project_root: Path, *, worktree_name: str | None = None,
    ) -> list[SessionSummary]:
        """列 encoded-cwd 目录下所有非空 .jsonl,按 mtime 倒序。

        first_user / turn_count 来自 sidecar <sid>.meta(append 时增量写,不扫 jsonl);
        sidecar 缺/坏 → lazy backfill 扫一次该 jsonl 补建(老会话一次性代价,之后 O(1))。
        mtime 用 jsonl 的 stat。空文件(st_size<=0,__init__ 占位)排除——stat 不读内容,
        故 O(文件数) 而非 O(总字节)。
        """
        d = paths.sessions_dir(root, project_root, worktree_name=worktree_name)
        if not d.exists():
            return []
        out: list[SessionSummary] = []
        for jf in d.glob("*.jsonl"):
            try:
                st = jf.stat()
            except OSError:
                log.debug("list_sessions 跳过 %s", jf, exc_info=True)
                continue
            if st.st_size <= 0:
                continue  # 空会话(__init__ 占位文件,从未写入)
            first_user, turn_count = _read_or_backfill_meta(jf)
            out.append(
                SessionSummary(
                    session_id=jf.stem,
                    mtime=st.st_mtime,
                    first_user_summary=first_user,
                    turn_count=turn_count,
                )
            )
        out.sort(key=lambda s: s.mtime, reverse=True)
        return out

    async def persist_tool_output(self, tool_use_id: str, full_output: str) -> PersistInfo | None:
        """落盘到 tool-results/<id>.txt,返回 PersistInfo;IO 失败返回 None。

        dumb sink:阈值决策全在 executor(per-tool max_result_chars),store 只管写文件。
        不再做 size 守卫——否则低阈值工具(grep 20K)在 20K~32K 之间会被 executor 决定
        落盘、却被这里的旧 32K 守卫拒绝,致模型拿不到占位也拿不到全文。
        """
        try:
            self._tool_results.mkdir(parents=True, exist_ok=True)
            target = self._tool_results / f"{tool_use_id}.txt"
            target.write_text(full_output, encoding="utf-8")
        except OSError:
            log.exception("persist_tool_output 落盘失败,降级 inline")
            return None
        # size 用字节(UTF-8 编码后),使占位 'KB' 展示准确——CJK 字符 UTF-8 占 3 字节,
        # 用字符数会低估 ~3 倍。阈值(max_result_chars)仍是字符,在 executor 比较 len(str)。
        size = len(full_output.encode("utf-8"))
        return PersistInfo(
            llm_content=_persisted_output_placeholder(str(target), size, full_output),
            path=str(target),
            size=size,
        )

    def close(self) -> None:
        # 直读 _fh_io(而非 self._fh property)→ 空会话不会因 close 触发建文件。
        if self._fh_io is not None:
            try:
                self._fh_io.close()
            except Exception:
                pass
            self._fh_io = None
        # 兜底:本会话 jsonl 若仍为空(从未写入,或异常遗留)→ 删掉,不留 0kb。
        try:
            if self._jsonl.exists() and self._jsonl.stat().st_size <= 0:
                self._jsonl.unlink()
        except OSError:
            log.debug("close 删除空 jsonl 失败", exc_info=True)
