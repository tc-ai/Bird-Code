"""记忆治理(memory governance / autoDream):后台定期回顾/合并/删除/修正记忆。

仿 MemoryManager 的轻量范式:扫记忆+会话 → 一次性 provider.complete → JSON 操作 →
代码应用 → rebuild_index。LLM 只返回意图,永不直接写文件。详见
docs/superpowers/specs/2026-07-14-memory-governance-design.md(路径 A)。

锁:空内容 lock 文件 + mtime 双用(锁状态 + 上次治理时间),O_EXCL 原子创建 +
mtime CAS,不用 PID(路径 A 下治理是主进程内 asyncio task,PID 多余且有复用陷阱)。
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, Field, ValidationError

from birdcode.memory.paths import (
    project_memory_dir,
    type_to_dir,
    user_memory_dir,
)
from birdcode.memory.store import (
    MemoryMeta,
    delete_memory,
    list_memories,
    parse_memory,
    rebuild_index,
    write_memory,
)
from birdcode.session.models import SessionSummary
from birdcode.session.store import SessionStore


class _LockBusy:
    """sentinel 类型:锁被占/抢输。独占类型使 `prev is _LOCK_BUSY` 后 mypy 可窄化。"""

    __slots__ = ()


_LOCK_BUSY: Final[_LockBusy] = _LockBusy()  # sentinel:锁被占/抢输(Final 使 is 可窄化)


def _retry_create(lock_path: Path) -> _LockBusy | None:
    """stat 发现文件刚被删时重试创建一次。返回 None(成功)或 _LOCK_BUSY(被抢)。"""
    try:
        lock_path.touch(exist_ok=False)
        return None
    except FileExistsError:
        return _LOCK_BUSY


def acquire_lock(lock_path: Path, *, now: float) -> _LockBusy | float | None:
    """O_EXCL 原子获取治理锁。

    返回 _LOCK_BUSY(被占/抢输) 或 acquire 前 mtime(None=首次,float=过期回收)。
    一文件两用:lock 空内容,mtime 承担全部判据——<1h 占用/冷却,>1h 过期可回收。
    详见 spec §4.2。
    """
    # 确保父目录存在:用户从未写过 user/feedback 记忆时 ~/.birdcode/memory/ 可能缺失,
    # touch 会抛 FileNotFoundError(非 FileExistsError)→ 外层 except 吞掉 → 治理静默失败。
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # 1. 原子创建:成功=拿到锁(首次)
    try:
        lock_path.touch(exist_ok=False)  # O_CREAT|O_EXCL
        return None
    except FileExistsError:
        pass
    # 2. 已存在:判占用/过期
    try:
        st = lock_path.stat()
    except FileNotFoundError:
        return _retry_create(lock_path)
    if (now - st.st_mtime) < 3600:
        return _LOCK_BUSY  # 1h 内:占用/冷却
    # 3. 过期:回收(CAS 验证 mtime 未变,再 unlink+touch)
    try:
        st2 = lock_path.stat()
    except FileNotFoundError:
        return _retry_create(lock_path)
    if st2.st_mtime != st.st_mtime:  # CAS:别人已更新
        return _LOCK_BUSY
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    try:
        lock_path.touch(exist_ok=False)
        return st.st_mtime  # 回收成功,记旧 mtime 供失败回滚
    except FileExistsError:
        return _LOCK_BUSY  # 被别人抢先


def restore_lock_mtime(lock_path: Path, prev_mtime: float | None) -> None:
    """整理失败时恢复锁 mtime,使时间门下次仍可通过。prev None(首次)→删文件。"""
    if prev_mtime is None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        return
    try:
        os.utime(lock_path, (prev_mtime, prev_mtime))
    except FileNotFoundError:
        pass


# ====== 治理操作模型与解析 ======

log = logging.getLogger(__name__)

_GOVERN_ACTION = Literal["merge", "delete", "update", "create"]
_MEMORY_TYPE = Literal["user", "feedback", "project", "reference"]


class _GovernTarget(BaseModel):
    """治理目标:记忆条目。"""

    name: str
    description: str
    type: _MEMORY_TYPE
    body: str


class _GovernOp(BaseModel):
    """单条治理操作(LLM 输出)。

    - merge: source_names(要合并的旧 name)+ target(新记忆)
    - delete: name + type(定位目录)
    - update: name + type(旧目录) + target(新内容,name 可改/可换目录)
    - create: target
    """

    action: _GOVERN_ACTION
    source_names: list[str] = Field(default_factory=list)
    name: str = ""
    type: _MEMORY_TYPE | None = None
    target: _GovernTarget | None = None


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*", re.MULTILINE)


def parse_operations(raw: str) -> list[_GovernOp]:
    """防御性解析 LLM 输出。剥围栏/取首末花括号/逐条 pydantic 校验/坏条跳过。"""
    text = _FENCE_RE.sub("", raw).strip().rstrip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    items = data.get("operations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    ops: list[_GovernOp] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            ops.append(_GovernOp.model_validate(it))
        except ValidationError:
            continue
    return ops


def _delete_any_scope(name: str, project_root: Path) -> None:
    """两目录都尝试删 name(幂等)——merge 的 source 可能跨目录,两删最稳。"""
    delete_memory(user_memory_dir(), name)
    delete_memory(project_memory_dir(project_root), name)


def apply_operation(op: _GovernOp, project_root: Path) -> None:
    """应用单条治理操作。单条失败吞+log,不波及整批(仿 extract _apply)。"""
    try:
        if op.action == "delete":
            if op.type is None:
                return
            delete_memory(type_to_dir(op.type, project_root), op.name)
        elif op.action == "create":
            if op.target is None:
                return
            t = op.target
            write_memory(
                type_to_dir(t.type, project_root),
                name=t.name,
                description=t.description,
                type_=t.type,
                body=t.body,
            )
        elif op.action == "update":
            if op.target is None or not op.name or op.type is None:
                return
            t = op.target
            delete_memory(type_to_dir(op.type, project_root), op.name)
            write_memory(
                type_to_dir(t.type, project_root),
                name=t.name,
                description=t.description,
                type_=t.type,
                body=t.body,
            )
        elif op.action == "merge":
            if op.target is None:
                return
            for src in op.source_names:
                _delete_any_scope(src, project_root)
            t = op.target
            write_memory(
                type_to_dir(t.type, project_root),
                name=t.name,
                description=t.description,
                type_=t.type,
                body=t.body,
            )
    except Exception:  # noqa: BLE001 - 单条失败不波及整批
        log.exception("治理操作应用失败: action=%s", op.action)


# ====== 治理 prompt 拼装 ======


_GOVERNANCE_SYSTEM = """你是 BirdCode 的记忆治理助手。下面给你当前的现有记忆(含正文)和最近会话主题。
请输出整理操作,严格 JSON: {"operations": [...]}。

操作类型:
- merge:把内容重复的几条合并成一条。source_names=要合并的旧 name 列表,target=新记忆。
- delete:删除已过时/被证伪/行为已变/指向已不存在文件/纯冗余的记忆。带 name + type。
- update:修正记忆。name=原 name + type=原目录,target=新内容(name 可改、可换 type 目录)。
  两条矛盾时改错的那条,并在 body 注明为何。
- create:最近会话浮现的、值得长期保留的新主题。target=新记忆。

原则:
1. 保守:拿不准就不动,宁可少改。
2. 合并优于新增:重复的合并,别 create 已有主题。
3. 日期:把"昨天/上周/最近"等相对日期转成绝对(YYYY-MM-DD)。
4. type 必须正确:user=用户偏好/feedback=用户反馈指导/project=项目进行中工作/reference=外部资源指针。
5. 只输出 JSON,无多余解释。

target 字段:name(kebab-case), description(一句话), type, body(markdown 正文)。
"""


def build_govern_prompt(
    user_mems: list[MemoryMeta],
    proj_mems: list[MemoryMeta],
    sessions: list[SessionSummary],
    *,
    today: str,
) -> str:
    """拼治理 user_prompt:今日日期 + 两目录记忆(含 body)+ 最近会话主题。"""
    lines: list[str] = [f"今日日期: {today}", ""]

    def _section(title: str, mems: list[MemoryMeta]) -> None:
        lines.append(title)
        if not mems:
            lines.append("(无)")
            return
        for m in mems:
            body = _read_body(m.path)
            lines.append(f"- [{m.name}] ({m.type}): {m.description}")
            if body.strip():
                lines.append(f"  {body.strip()}")

    _section("## 现有记忆(用户级 user/feedback)", user_mems)
    lines.append("")
    _section("## 现有记忆(项目级 project/reference)", proj_mems)
    lines.append("")
    lines.append("## 最近会话主题")
    if sessions:
        for s in sessions:
            lines.append(f"- {s.first_user_summary}({s.turn_count} 轮)")
    else:
        lines.append("(无)")
    return "\n".join(lines)


def _read_body(path: Path) -> str:
    """读记忆文件的 body(剥 frontmatter)。读失败→空。"""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, ValueError):  # ValueError 含 UnicodeDecodeError(非 OSError 子类)
        return ""
    parsed = parse_memory(text)
    return parsed[1] if parsed is not None else ""


# ====== Task 5: GovernanceManager 编排 ======

_MAX_SIGNAL_SESSIONS = 8
_GOVERN_MAX_TOKENS = 4096
_LOCK_FILENAME = ".consolidate-lock"


class GovernanceManager:
    """记忆治理:后台定期回顾/合并/删除/修正记忆。仿 MemoryManager 三件套。

    绝不阻塞主 agent(调用方 fire-and-forget)、绝不抛(吞异常)、绝不杀主循环。
    详见 spec §4。
    """

    def __init__(
        self, provider: Any, *, project_root: Path, govern_model: str, session_store: Any
    ) -> None:
        self._provider = provider
        self._project_root = project_root
        self._govern_model = govern_model
        self._session_store = session_store
        self._lock = asyncio.Lock()
        self._last_scan_ts: float = 0.0
        self._lock_path = user_memory_dir() / _LOCK_FILENAME

    def set_provider(self, provider: Any, *, govern_model: str) -> None:
        """/profile 切换后重连(同 MemoryManager.set_provider)。"""
        self._provider = provider
        self._govern_model = govern_model

    def _list_sessions(self) -> list[Any]:
        return SessionStore.list_sessions(
            self._session_store.root,
            self._session_store.project_root,
            worktree_name=self._session_store.worktree_name,
        )

    def _early_gates(self) -> bool:
        """门 1-3:目录存在 / 时间门 24h / 扫描节流 10min。轻量,先于会话扫描。"""
        # 门 1:记忆目录存在
        if not (user_memory_dir().is_dir() or project_memory_dir(self._project_root).is_dir()):
            return False
        # 门 2:时间门——锁 mtime 距今 > 24h(无锁=从未治理=通过)
        try:
            age = time.time() - self._lock_path.stat().st_mtime
        except FileNotFoundError:
            age = float("inf")
        if age <= 86400:
            return False
        # 门 3:扫描节流——距上次扫描 > 10min
        if time.time() - self._last_scan_ts <= 600:
            return False
        self._last_scan_ts = time.time()
        return True

    async def govern(self) -> None:
        """门控 + 锁 + 整理。绝不抛,绝不阻塞。"""
        async with self._lock:
            try:
                if not self._early_gates():  # 门 1-3
                    return
                sessions = self._list_sessions()
                if len(sessions) < 5:  # 门 4
                    return
                prev = acquire_lock(self._lock_path, now=time.time())  # 门 5
                # isinstance 窄化(mypy 不对 `is _LOCK_BUSY` 类实例做窄化);
                # _LockBusy 仅一个单例,故等价于 `is _LOCK_BUSY`。
                if isinstance(prev, _LockBusy):
                    return
                # 过 busy 守卫后 prev 必为 float|None;显式绑定让 except 内不依赖
                # 跨 try/except 的类型窄化(mypy 不延续窄化到 except 块)。
                prev_mtime: float | None = prev
                try:
                    await self._govern_inner(sessions)
                except Exception:  # noqa: BLE001 - 整理失败:回滚锁 mtime,下次可重试
                    log.exception("记忆治理失败")
                    restore_lock_mtime(self._lock_path, prev_mtime)
            except Exception:  # noqa: BLE001 - 入口异常绝不杀主循环
                log.exception("记忆治理入口异常")

    async def _govern_inner(self, sessions: list[Any]) -> None:
        user_mems = list_memories(user_memory_dir())
        proj_mems = list_memories(project_memory_dir(self._project_root))
        today = datetime.date.today().isoformat()
        user_prompt = build_govern_prompt(
            user_mems,
            proj_mems,
            sessions[:_MAX_SIGNAL_SESSIONS],
            today=today,
        )
        raw = await self._provider.complete(
            _GOVERNANCE_SYSTEM,
            user_prompt,
            max_tokens=_GOVERN_MAX_TOKENS,
            model=self._govern_model,
        )
        for op in parse_operations(raw):
            apply_operation(op, self._project_root)
        rebuild_index(user_memory_dir())
        rebuild_index(project_memory_dir(self._project_root))
