"""记忆治理测试:锁/parse/apply/门控/govern_inner。仿 test_memory_extractor.py。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from birdcode.memory.governance import (
    _GOVERNANCE_SYSTEM,
    _LOCK_BUSY,
    GovernanceManager,
    _GovernOp,
    acquire_lock,
    apply_operation,
    build_govern_prompt,
    parse_operations,
    restore_lock_mtime,
)
from birdcode.memory.paths import project_memory_dir, user_memory_dir
from birdcode.memory.store import list_memories
from birdcode.session.models import SessionSummary
from birdcode.session.paths import default_root, sessions_dir


def _ss(n: int) -> list[SessionSummary]:
    """造 n 个真 SessionSummary(替代裸字符串 mock,匹配 build_govern_prompt 类型契约)。"""
    return [
        SessionSummary(session_id="s", mtime=1.0, first_user_summary="x", turn_count=1)
        for _ in range(n)
    ]


@pytest.fixture
def home(monkeypatch, tmp_path):
    """重定向 Path.home() 到 tmp(锁文件+用户级记忆都在此)。"""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", lambda: h)
    return h


def test_acquire_first_creates_returns_none(tmp_path):
    lock = tmp_path / "lock"
    assert acquire_lock(lock, now=1_000_000.0) is None
    assert lock.exists()


def test_acquire_busy_when_recent(tmp_path):
    lock = tmp_path / "lock"
    lock.touch()
    now = time.time()
    os.utime(lock, (now - 60, now - 60))  # 1 分钟前,<1h
    assert acquire_lock(lock, now=now) is _LOCK_BUSY


def test_acquire_recycles_when_expired(tmp_path):
    """mtime>1h 过期:回收(unlink+touch)成功,返回旧 mtime 供失败回滚。"""
    lock = tmp_path / "lock"
    lock.touch()
    now = time.time()
    old = now - 7200
    os.utime(lock, (old, old))  # 2h 前
    prev = acquire_lock(lock, now=now)
    assert prev is not _LOCK_BUSY
    assert prev == old
    assert lock.exists()  # 已重建


def test_acquire_cas_aborts_if_mtime_changed(tmp_path, monkeypatch):
    """过期回收前 CAS 再 stat,若 mtime 已被别人更新→放弃。"""
    lock = tmp_path / "lock"
    lock.touch()
    now = time.time()
    os.utime(lock, (now - 7200, now - 7200))  # 过期
    real_stat = Path.stat
    count = {"n": 0}

    def counted(self, *a, **k):
        count["n"] += 1
        # 第二次 stat(CAS 检查)时模拟别人已 touch(mtime 变新)
        if count["n"] == 2 and self == lock:
            os.utime(lock, (now - 10, now - 10))
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", counted)
    assert acquire_lock(lock, now=now) is _LOCK_BUSY


def test_acquire_concurrent_only_one_wins(tmp_path):
    """两进程并发(顺序模拟):首个 touch 成功,第二个看到存在+mtime<1h→放弃。"""
    lock = tmp_path / "lock"
    now = time.time()
    first = acquire_lock(lock, now=now)
    second = acquire_lock(lock, now=now)
    assert first is None
    assert second is _LOCK_BUSY


def test_restore_none_unlinks(tmp_path):
    lock = tmp_path / "lock"
    lock.touch()
    restore_lock_mtime(lock, None)
    assert not lock.exists()


def test_restore_mtime_sets_utime(tmp_path):
    lock = tmp_path / "lock"
    lock.touch()
    restore_lock_mtime(lock, 12345.0)
    assert lock.stat().st_mtime == 12345.0


def test_restore_missing_file_noop(tmp_path):
    restore_lock_mtime(tmp_path / "nope", 12345.0)  # 不抛


# ====== parse_operations 测试 ======


def _ops_json(ops):
    return json.dumps({"operations": ops}, ensure_ascii=False)


def test_parse_normal():
    raw = _ops_json(
        [
            {"action": "delete", "name": "old", "type": "project"},
            {
                "action": "create",
                "target": {"name": "x", "description": "d", "type": "user", "body": "b"},
            },
        ]
    )
    ops = parse_operations(raw)
    assert len(ops) == 2
    assert ops[0].action == "delete" and ops[0].name == "old"
    assert ops[1].action == "create" and ops[1].target.name == "x"


def test_parse_strips_code_fence():
    raw = "```json\n" + _ops_json([{"action": "delete", "name": "a", "type": "user"}]) + "\n```"
    ops = parse_operations(raw)
    assert len(ops) == 1


def test_parse_malformed_json_returns_empty():
    assert parse_operations("这不是 JSON") == []


def test_parse_skips_bad_keeps_good():
    raw = _ops_json(
        [
            {
                "action": "create",
                "target": {"name": "good", "description": "d", "type": "feedback", "body": "b"},
            },
            {"action": "craete", "target": {"name": "bad"}},  # action 拼错+缺字段
        ]
    )
    ops = parse_operations(raw)
    assert [o.action for o in ops] == ["create"]
    assert ops[0].target.name == "good"


def test_parse_merge_op():
    raw = _ops_json(
        [
            {
                "action": "merge",
                "source_names": ["a", "b"],
                "target": {"name": "c", "description": "d", "type": "user", "body": "b"},
            }
        ]
    )
    ops = parse_operations(raw)
    assert ops[0].action == "merge"
    assert ops[0].source_names == ["a", "b"]


def test_parse_bad_action_type_rejected():
    raw = _ops_json([{"action": "explode", "name": "x", "type": "user"}])
    assert parse_operations(raw) == []


def test_govern_op_target_optional_by_default():
    op = _GovernOp(action="delete", name="x", type="user")
    assert op.target is None
    assert op.source_names == []


# ====== apply_operation 测试 ======


def _target(**kw):
    from birdcode.memory.governance import _GovernTarget

    base = {"name": "t", "description": "d", "type": "user", "body": "b"}
    base.update(kw)
    return _GovernTarget(**base)


def test_apply_create_routes_by_type(tmp_path, home):
    op = _GovernOp(action="create", target=_target(name="pref", type="feedback"))
    apply_operation(op, tmp_path)
    assert [m.name for m in list_memories(user_memory_dir())] == ["pref"]
    assert list_memories(project_memory_dir(tmp_path)) == []


def test_apply_create_project_type(tmp_path, home):
    op = _GovernOp(action="create", target=_target(name="ci", type="project"))
    apply_operation(op, tmp_path)
    assert [m.name for m in list_memories(project_memory_dir(tmp_path))] == ["ci"]


def test_apply_delete_removes(tmp_path, home):
    from birdcode.memory.store import write_memory

    write_memory(user_memory_dir(), name="n", description="d", type_="user", body="b")
    op = _GovernOp(action="delete", name="n", type="user")
    apply_operation(op, tmp_path)
    assert list_memories(user_memory_dir()) == []


def test_apply_update_rewrites_same_dir(tmp_path, home):
    from birdcode.memory.store import write_memory

    write_memory(user_memory_dir(), name="n", description="old", type_="user", body="b")
    op = _GovernOp(
        action="update",
        name="n",
        type="user",
        target=_target(name="n", description="new", type="user"),
    )
    apply_operation(op, tmp_path)
    metas = list_memories(user_memory_dir())
    assert len(metas) == 1 and metas[0].description == "new"


def test_apply_update_rename_removes_old(tmp_path, home):
    from birdcode.memory.store import write_memory

    write_memory(
        project_memory_dir(tmp_path), name="old", description="d", type_="project", body="b"
    )
    op = _GovernOp(
        action="update",
        name="old",
        type="project",
        target=_target(name="new", description="d", type="project"),
    )
    apply_operation(op, tmp_path)
    names = [m.name for m in list_memories(project_memory_dir(tmp_path))]
    assert names == ["new"]


def test_apply_merge_deletes_sources_writes_target(tmp_path, home):
    from birdcode.memory.store import write_memory

    d = user_memory_dir()
    write_memory(d, name="a", description="d", type_="user", body="b")
    write_memory(d, name="b", description="d", type_="user", body="b")
    op = _GovernOp(
        action="merge",
        source_names=["a", "b"],
        target=_target(name="merged", type="user"),
    )
    apply_operation(op, tmp_path)
    assert [m.name for m in list_memories(d)] == ["merged"]


def test_apply_missing_target_noop(tmp_path, home):
    op = _GovernOp(action="create")  # target=None
    apply_operation(op, tmp_path)  # 不抛


def test_apply_delete_missing_type_noop(tmp_path, home):
    op = _GovernOp(action="delete", name="x")  # type=None
    apply_operation(op, tmp_path)  # 不抛


# ====== Task 4: _GOVERNANCE_SYSTEM + build_govern_prompt 测试 ======


def test_governance_system_mentions_actions_and_principles():
    assert "merge" in _GOVERNANCE_SYSTEM
    assert "delete" in _GOVERNANCE_SYSTEM
    assert "保守" in _GOVERNANCE_SYSTEM
    assert "JSON" in _GOVERNANCE_SYSTEM


def test_build_prompt_includes_today_and_mems_and_sessions(tmp_path, home):
    from birdcode.memory.store import write_memory

    write_memory(
        user_memory_dir(), name="pref", description="用户偏好 X", type_="user", body="正文"
    )
    umems = list_memories(user_memory_dir())
    sess = [SessionSummary(session_id="s1", mtime=1.0, first_user_summary="聊聊 Y", turn_count=3)]
    prompt = build_govern_prompt(umems, [], sess, today="2026-07-14")
    assert "2026-07-14" in prompt
    assert "pref" in prompt and "用户偏好 X" in prompt
    assert "正文" in prompt  # body 也喂(找重复/矛盾需要)
    assert "聊聊 Y" in prompt and "3 轮" in prompt


def test_build_prompt_empty_mems(tmp_path, home):
    prompt = build_govern_prompt([], [], [], today="2026-07-14")
    assert "(无)" in prompt


# ====== Task 5: GovernanceManager 门控/编排 测试 ======


class FakeProvider:
    def __init__(self, payload=""):
        self._payload = payload
        self.calls: list[dict] = []

    async def complete(self, system, user, *, max_tokens, model=None):
        self.calls.append({"model": model})
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeStore:
    """给 GovernanceManager 提供 root/project_root/worktree_name(list_sessions 是 staticmethod)。"""

    def __init__(self, root, project_root):
        self.root = root
        self.project_root = project_root
        self.worktree_name = None


def _mgr(tmp_path, monkeypatch, payload=""):
    monkeypatch.setattr("birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: [])
    return GovernanceManager(
        FakeProvider(payload),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(tmp_path, tmp_path),
    )


def test_govern_no_dir_returns_early(tmp_path, home, monkeypatch):
    # home 下无 .birdcode/memory,project 下也无→门1不过
    mgr = _mgr(tmp_path, payload=_ops_json([]), monkeypatch=monkeypatch)
    import asyncio

    asyncio.run(mgr.govern())
    assert mgr._provider.calls == []


def test_govern_time_gate_blocks(tmp_path, home, monkeypatch):
    # 建目录让门1过,但 lock 文件 mtime<24h→门2不过
    user_memory_dir().mkdir(parents=True)
    lock = user_memory_dir() / ".consolidate-lock"
    lock.touch()
    now = time.time()
    os.utime(lock, (now - 60, now - 60))
    monkeypatch.setattr("birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: [])
    mgr = GovernanceManager(
        FakeProvider(_ops_json([])),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(tmp_path, tmp_path),
    )
    import asyncio

    asyncio.run(mgr.govern())
    assert mgr._provider.calls == []


def test_govern_insufficient_sessions_blocks(tmp_path, home, monkeypatch):
    user_memory_dir().mkdir(parents=True)  # 门1过;无 lock→门2过(从未治理)
    monkeypatch.setattr(
        "birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: _ss(4)
    )  # 4<5
    mgr = GovernanceManager(
        FakeProvider(_ops_json([])),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(tmp_path, tmp_path),
    )
    import asyncio

    asyncio.run(mgr.govern())
    assert mgr._provider.calls == []


def test_govern_runs_and_applies(tmp_path, home, monkeypatch):
    user_memory_dir().mkdir(parents=True)
    monkeypatch.setattr(
        "birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: _ss(5)
    )
    payload = _ops_json(
        [
            {
                "action": "create",
                "target": {"name": "new", "description": "d", "type": "feedback", "body": "b"},
            }
        ]
    )
    mgr = GovernanceManager(
        FakeProvider(payload),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(tmp_path, tmp_path),
    )
    import asyncio

    asyncio.run(mgr.govern())
    assert mgr._provider.calls != []
    assert [m.name for m in list_memories(user_memory_dir())] == ["new"]


def test_govern_passes_govern_model(tmp_path, home, monkeypatch):
    user_memory_dir().mkdir(parents=True)
    monkeypatch.setattr(
        "birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: _ss(5)
    )
    p = FakeProvider(_ops_json([]))
    mgr = GovernanceManager(
        p,
        project_root=tmp_path,
        govern_model="haiku-mini",
        session_store=FakeStore(tmp_path, tmp_path),
    )
    import asyncio

    asyncio.run(mgr.govern())
    assert p.calls[0]["model"] == "haiku-mini"


def test_govern_failure_restores_lock_and_swallows(tmp_path, home, monkeypatch):
    user_memory_dir().mkdir(parents=True)
    monkeypatch.setattr(
        "birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: _ss(5)
    )
    lock = user_memory_dir() / ".consolidate-lock"
    mgr = GovernanceManager(
        FakeProvider(RuntimeError("boom")),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(tmp_path, tmp_path),
    )
    import asyncio

    asyncio.run(mgr.govern())  # 不抛
    # 失败回滚:锁被删(prev=None,首次)
    assert not lock.exists()


def test_govern_serializes_via_lock(tmp_path, home, monkeypatch):
    user_memory_dir().mkdir(parents=True)
    monkeypatch.setattr(
        "birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: _ss(5)
    )
    payload = _ops_json(
        [
            {
                "action": "create",
                "target": {"name": "n", "description": "d", "type": "user", "body": "b"},
            }
        ]
    )
    mgr = GovernanceManager(
        FakeProvider(payload),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(tmp_path, tmp_path),
    )
    import asyncio

    async def _both() -> None:
        await asyncio.gather(mgr.govern(), mgr.govern())

    asyncio.run(_both())
    assert len(list_memories(user_memory_dir())) == 1


def test_set_provider_reroutes(tmp_path, home, monkeypatch):
    monkeypatch.setattr("birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: [])
    p2 = FakeProvider(_ops_json([]))
    mgr = _mgr(tmp_path, payload=_ops_json([]), monkeypatch=monkeypatch)
    mgr.set_provider(p2, govern_model="new")
    assert mgr._govern_model == "new" and mgr._provider is p2


@pytest.mark.asyncio
async def test_maybe_govern_skips_when_none(tmp_path):
    import asyncio  # noqa: F401

    from birdcode.conversation import Turn, TurnController

    ctrl = object.__new__(TurnController)
    ctrl._governance = None
    ctrl._governance_tasks = set()
    ctrl._maybe_govern(Turn(messages=[]))  # 不抛


@pytest.mark.asyncio
async def test_maybe_govern_skips_interrupted(tmp_path):
    import asyncio

    from birdcode.conversation import Turn, TurnController

    class _Gov:
        called = False

        async def govern(self):
            _Gov.called = True

    ctrl = object.__new__(TurnController)
    ctrl._governance = _Gov()
    ctrl._governance_tasks = set()
    turn = Turn(messages=[])
    turn.interrupted = True
    ctrl._maybe_govern(turn)
    await asyncio.sleep(0)
    assert _Gov.called is False


@pytest.mark.asyncio
async def test_maybe_govern_fires_task(tmp_path):
    import asyncio

    from birdcode.conversation import Turn, TurnController

    class _Gov:
        def __init__(self):
            self.done = asyncio.Event()

        async def govern(self):
            self.done.set()

    ctrl = object.__new__(TurnController)
    gov = _Gov()
    ctrl._governance = gov
    ctrl._governance_tasks = set()
    ctrl._maybe_govern(Turn(messages=[]))
    await asyncio.wait_for(gov.done.wait(), timeout=2)
    # done.set() 先唤醒 wait,task done_callback(discard)排在其后,需再让一步。
    await asyncio.sleep(0)
    assert not ctrl._governance_tasks  # done 回调已 discard


# ====== Task 7: 端到端集成测试(真实 list_sessions + 文件落盘) ======


def _make_sessions(tmp_path, home, n=5):
    """在 ~/.birdcode/projects/<encoded-cwd>/ 下造 n 个非空会话 jsonl。

    home fixture 重定向 Path.home → tmp,故 default_root() 指向 tmp 内。
    e2e 测试中 FakeStore(default_root(), ...)让 list_sessions 读同一 root。
    """
    d = sessions_dir(default_root(), tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"s{i}.jsonl").write_text(
            json.dumps({"type": "user", "content": f"会话 {i}"}) + "\n",
            encoding="utf-8",
        )


@pytest.mark.asyncio
async def test_e2e_merge_delete_fixdate(tmp_path, home):
    from birdcode.memory.store import write_memory

    _make_sessions(tmp_path, home, 5)

    d = user_memory_dir()
    d.mkdir(parents=True, exist_ok=True)
    write_memory(d, name="no-push-1", description="不要push", type_="feedback", body="别自动push")
    write_memory(d, name="no-push-2", description="禁止push", type_="feedback", body="不要git push")
    # old-framework 是 project 型:须落在项目目录,delete 按 type 路由才命中。
    write_memory(
        project_memory_dir(tmp_path),
        name="old-framework",
        description="用LangChain",
        type_="project",
        body="旧",
    )

    payload = _ops_json(
        [
            {
                "action": "merge",
                "source_names": ["no-push-1", "no-push-2"],
                "target": {
                    "name": "no-auto-push",
                    "description": "禁止自动 git push",
                    "type": "feedback",
                    "body": "用户明确禁止自动 git commit/push(2026-07-14 确认)",
                },
            },
            {"action": "delete", "name": "old-framework", "type": "project"},
        ]
    )
    mgr = GovernanceManager(
        FakeProvider(payload),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(default_root(), tmp_path),
    )
    await mgr.govern()

    names = sorted(m.name for m in list_memories(d))
    assert names == ["no-auto-push"]  # 合并重复 + 删过时
    assert "no-auto-push" in (d / "MEMORY.md").read_text(encoding="utf-8")  # rebuild 同步
    assert list_memories(project_memory_dir(tmp_path)) == []


@pytest.mark.asyncio
async def test_e2e_provider_error_does_not_throw(tmp_path, home):
    _make_sessions(tmp_path, home, 5)
    # 门1:记忆目录须存在,方可穿过早门到达 provider 触发错误 → 验证锁回滚。
    user_memory_dir().mkdir(parents=True)
    mgr = GovernanceManager(
        FakeProvider(RuntimeError("网络挂了")),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(default_root(), tmp_path),
    )
    await mgr.govern()  # 不抛,不阻塞
    # 锁被回滚(首次失败→删锁),时间门下次可过
    assert not (user_memory_dir() / ".consolidate-lock").exists()


def test_govern_runs_when_only_project_mem_dir_exists(tmp_path, home, monkeypatch):
    """I1 回归:只有 project 记忆目录、user 记忆目录缺失时,govern 不应因 acquire_lock 崩。

    场景:用户只有 project/reference 记忆(<project>/.birdcode/memory/ 存在),从未写过
    user/feedback 记忆(~/.birdcode/memory/ 不存在)。门1(任一目录存在)过 → 进 acquire_lock,
    旧实现 touch 在缺失父目录上抛 FileNotFoundError(非 FileExistsError,未被捕获)→ 治理静默失败。
    修复后 acquire_lock 开头 mkdir,自动建 user_memory_dir。
    """
    # 只建 project 记忆目录;故意不建 user_memory_dir(验证 acquire_lock 自动建)
    project_memory_dir(tmp_path).mkdir(parents=True)
    assert not user_memory_dir().exists()  # 前置断言:user 目录确实不存在
    monkeypatch.setattr(
        "birdcode.memory.governance.SessionStore.list_sessions", lambda *a, **k: _ss(5)
    )
    payload = _ops_json(
        [
            {
                "action": "create",
                "target": {"name": "new", "description": "d", "type": "feedback", "body": "b"},
            }
        ]
    )
    mgr = GovernanceManager(
        FakeProvider(payload),
        project_root=tmp_path,
        govern_model="haiku",
        session_store=FakeStore(tmp_path, tmp_path),
    )
    import asyncio

    asyncio.run(mgr.govern())  # 不抛
    assert mgr._provider.calls != []  # provider 被调用 → 穿过了 acquire_lock
    assert user_memory_dir().is_dir()  # acquire_lock 自动建了父目录
