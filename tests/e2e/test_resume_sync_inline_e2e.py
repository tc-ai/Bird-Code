# tests/e2e/test_resume_sync_inline_e2e.py
"""Task 24 E2E:sync 子 agent 中断续跑——阻塞 inline 返回报告。

串联真实组件(非单点 mock):discover(T2)→ ResumeAgentTool(T9)→ resume_subagent(T5
sync 分支:`await runner.run()` 阻塞返回 inline,非 launch_async 后台)→ 真实 SessionStore
主 jsonl 文件 IO + 真实 SubagentManager(只走 has_live 幂等守卫,sync 不进 launch_async/
_supervise/_inject)+ T6 sync 占位(load_mainline 把悬空 sync agent tool_use 的合成
tool_result content 改「中断,待续跑」)。

场景(brief Step 1):
  ① 造现场——主 jsonl 末尾有 sync agent tool_use(name=general-purpose、id=toolu_1)
     unpaired(无 tool_result);meta sub-A status=running、is_async=False、toolUseId
     占位"(sync-agent)";侧链有种子 user + 一轮 assistant(已读 a.txt)。
  ② load_mainline 触发 T6 占位:toolu_1 的合成 tool_result content 含「中断/续跑」。
  ③ resume→find_resumable_subagents 发现 sub-A→授权假定→主 agent 调
     ResumeAgentTool.execute(agent_id="sub-A", direction="继续")。
  ④ 断言——sync 分支:runner.run() 被调、launch_async 全程未被调、resume_subagent 阻塞
     等 run() 完成(非 fire-and-forget)、result == report.text(inline,非 async ack)。
     主 jsonl 无 enqueue / 无 task-notification(sync 不经 _inject)。

真实 vs mock 边界(据实划定):
  - 真实路径:discover / resume_subagent(sync 分支 runner.run)/ ResumeAgentTool /
    SubagentManager.has_live / 真实 meta + 侧链 + 主 jsonl 文件 IO / T6 占位
    (load_mainline → _mark_pending_sync_agent_tooluses)。
  - mock:resume_mod.SubagentRunner(其 run() 经 class-level asyncio.Event 阻塞到测试放行,
    既避免真实 LLM,又借此证明 resume_subagent 同步 await run() 而非后台)。
  - 不验证:LLM 智力、真实工具副作用(均 mock)。聚焦「sync 中断 → 发现 → 阻塞续跑 →
    inline 报告返回」这条闭环的组件串联,以及与 async 路径的分支差异(不进 launch_async)。

mock 范式:抄 tests/e2e/test_resume_async_e2e.py(_FakeRunner + 真实 meta/侧链/paths +
monkeypatch resume_mod.SubagentRunner),改成 sync 场景(is_async=False)+ 阻塞探针。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from birdcode.agents import resume as resume_mod
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.discover import find_resumable_subagents
from birdcode.agents.manager import SubagentManager
from birdcode.agents.registry import AgentRegistry
from birdcode.agents.resume import ResumeDeps
from birdcode.agents.runner import SubagentReport
from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.conversation import Message, Turn, TurnController
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.paths import (
    subagent_jsonl_path,
    subagent_meta_path,
    subagents_dir,
)
from birdcode.session.store import SessionStore
from birdcode.session.subagent_meta import write_subagent_meta
from birdcode.tools.resume_agent_tool import ResumeAgentTool

# ---- mock 主 agent provider:sync 路径 resume_subagent 不经 notify_wake 驱动主 ----
# agent 消费(结果 inline 返回),parent_provider 仅作 ResumeDeps 真实构造入参,不被流式调用。


class _MainProvider:
    """主 agent 桩:sync 续跑结果 inline 返回,不经主 agent 流式消费,故 stream 不会被调。

    保留 stream 接口仅为满足 ResumeDeps 构造(parent_provider 字段);若被调即说明 sync
    路径误走了 notify_wake→主 agent 消费(不该发生),断言里会捕获。
    """

    def __init__(self, profile: ProviderProfile) -> None:
        self.profile = profile
        self.stream_called = False

    def stream(self, messages, *, history):  # noqa: ARG002
        self.stream_called = True

        async def _gen():  # pragma: no cover - sync 路径不应走到主 agent 流式
            yield ""

        return _gen()


# ---- mock SubagentRunner:阻塞探针 + 固定 run() 返回,验证 sync 分支阻塞 inline ----


class _FakeRunner:
    """SubagentRunner 替身:经 class-level asyncio.Event 阻塞 run() 直到测试放行。

    - run() 先 set `started`(通知测试:resume_subagent 已到 sync 分支并开始 await run());
      再 await `proceed`(阻塞,证明 resume_subagent 同步等待 run() 而非 fire-and-forget);
      最后 `finished=True` 返固定 completed SubagentReport(避免真实 LLM)。
    - constructed 类级列表验「复用 id」「is_async=False」「构造次数」「resume_from 指向侧链」。
    """

    constructed: list[_FakeRunner] = []
    # 类级阻塞探针(在测试中、running loop 内重新赋值新 Event):
    started: asyncio.Event | None = None
    proceed: asyncio.Event | None = None
    finished: bool = False

    def __init__(self, **kw: object) -> None:
        self.captured_kw = kw
        self.defn = kw.get("defn")
        self.agent_id = kw.get("agent_id")
        self.prompt = kw.get("prompt")
        self.tool_use_id = kw.get("tool_use_id")
        self.isolation = kw.get("isolation")
        self.description = kw.get("description")
        self.run_called = False
        type(self).constructed.append(self)

    async def run(self) -> SubagentReport:
        self.run_called = True
        cls = type(self)
        # 通知测试:sync 分支已 await runner.run()(started 在 running loop 内 set)
        if cls.started is not None:
            cls.started.set()
        # 阻塞直到测试放行 → 证明 resume_subagent 同步 await run()(非后台 launch_async)
        if cls.proceed is not None:
            await cls.proceed.wait()
        cls.finished = True
        return SubagentReport(
            is_completed=True,
            text="续跑完成:已读 a.txt 并改成 X",
            status="completed",
            duration_ms=12,
            tokens=8,
            tool_use_count=2,
        )


# ---- 辅助 ----


async def _noop_event(_ev: object) -> None: ...


async def _noop_status() -> None: ...


def _cfg() -> AppConfig:
    return AppConfig(
        providers={
            "p": ProviderProfile(
                name="p", protocol="anthropic", model="m", base_url="http://x", api_key="k"
            )
        },
        default="p",
    )


def _make_store(tmp_path: Path, session_id: str) -> SessionStore:
    ctx = SessionContext(
        session_id=session_id, cwd=str(tmp_path), version="0.1.0", git_branch=None
    )
    return SessionStore(ctx, tmp_path, root=tmp_path)


def _registry() -> AgentRegistry:
    r = AgentRegistry()
    # defn 注册名 general-purpose 与 meta.agent_type / 主 jsonl tool_use.name 一致
    # (T6 占位按 tool_use.name == agent_type 配对);run_in_background=False = sync agent。
    r.register(
        AgentDefinition(
            name="general-purpose",
            description="通用子 agent",
            system_prompt="SP",
            run_in_background=False,
        )
    )
    return r


def _write_sub_a_meta(tmp_path: Path, session_id: str, *, status: str = "running") -> None:
    """写真实 sub-A meta.json(status=running、is_async=False、toolUseId 占位"(sync-agent)")。

    is_async=False → resume_subagent 走 sync 分支(runner.run 阻塞 inline)。agent_type=
    general-purpose 与主 jsonl tool_use.name 一致 → T6 占位配对命中。
    """
    meta_path = subagent_meta_path(tmp_path, session_id, tmp_path, "sub-A")
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId="sub-A",
            agentType="general-purpose",
            description="旧任务:处理 a.txt",
            toolUseId="(sync-agent)",  # sync 占位 id:只进子 agent meta/queue-op,不进主 jsonl
            isAsync=False,
            status=status,
        ),
    )


def _write_sub_a_sidechain(tmp_path: Path, session_id: str) -> Path:
    """写 sub-A 侧链:种子 user + 一轮 assistant(已读 a.txt)。

    assistant 文本用「## 待办清单」头(runner.py 完成判据的反向标志)→ T17 交叉校验判
    未完成、不误注入完成,使续跑路径抵达 runner.run()。有真实(非 synthetic)assistant
    产出 → T18 空侧链检查放行。
    """
    p = subagent_jsonl_path(tmp_path, session_id, tmp_path, "sub-A")
    user_line = (
        '{"type":"user","message":{"role":"user","content":'
        '[{"type":"text","text":"读 a.txt 并改成 X"}]}}'
    )
    asst_line = (
        '{"type":"assistant","message":{"role":"assistant","content":'
        '[{"type":"text","text":"## 待办清单\\n- [x] 已读 a.txt\\n- [ ] 改成 X"}]}}'
    )
    p.write_text("\n".join([user_line, asst_line]) + "\n", encoding="utf-8")
    return p


def _queue_ops(rows: list[dict]) -> list[dict]:
    return [r for r in rows if isinstance(r, dict) and r.get("type") == "queue-operation"]


def _notif_rows(rows: list[dict], agent_id: str | None = None) -> list[dict]:
    out = [
        r
        for r in rows
        if isinstance(r, dict) and r.get("type") == "user" and r.get("isTaskNotification")
    ]
    if agent_id is not None:
        out = [r for r in out if r.get("agentId") == agent_id]
    return out


def _find_tool_result(turns: list[Turn], tool_use_id: str) -> ToolResultBlock:
    return next(
        b for t in turns for m in t.messages for b in m.content
        if isinstance(b, ToolResultBlock) and b.tool_use_id == tool_use_id
    )


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch, tmp_path):
    """侧链/subagent-store 落 tmp(paths.default_root → tmp_path),不污染 ~/.birdcode。"""
    from birdcode.session import paths

    monkeypatch.setattr(paths, "default_root", lambda: tmp_path)


# ---- E2E:sync 中断续跑阻塞 inline 闭环 ----


@pytest.mark.asyncio
async def test_sync_interrupted_resumes_blocking_inline(tmp_path: Path, monkeypatch):
    """brief Step 1:造现场 → T6 占位 → resume → 断言 sync 阻塞 inline 闭环。

    真实串联:find_resumable_subagents → ResumeAgentTool.execute → resume_subagent(sync
    分支 `await runner.run()`)→ FakeRunner.run()(阻塞到测试放行)→ ResumeResult.text
    inline 回传。全程 launch_async / _inject / notify_wake 不触发(sync 不走后台)。
    """
    _FakeRunner.constructed.clear()
    # 类级阻塞探针重置(必须在 running loop 内构造 Event):
    _FakeRunner.started = asyncio.Event()
    _FakeRunner.proceed = asyncio.Event()
    _FakeRunner.finished = False

    session_id = "e2e-sync"

    # ===== ① 造现场:meta(running/sync)+ 侧链(种子 user + 已读 a.txt 的 assistant)=====
    _write_sub_a_meta(tmp_path, session_id, status="running")
    expected_sidechain = _write_sub_a_sidechain(tmp_path, session_id)

    cfg = _cfg()

    # 主 jsonl:主 agent 调 sync general-purpose 工具(toolu_1)未拿到 result 就中断(unpaired)。
    store = _make_store(tmp_path, session_id)
    try:
        await store.append(Message(role="user", content=[TextBlock(text="go")]))
        await store.append(
            Message(
                role="assistant",
                content=[ToolUseBlock(id="toolu_1", name="general-purpose",
                                      input={"prompt": "do x"})],
            ),
            is_assistant=True,
        )
    finally:
        store.close()

    # ===== ② T6 占位:reload → load_mainline 把 toolu_1 悬空 tool_use 的合成 tool_result
    #       content 改「中断...续跑」(sub-A running + is_async=False + agent_type 命中)。=====
    store_pl = _make_store(tmp_path, session_id)
    try:
        turns_pl = await store_pl.load_mainline()
    finally:
        store_pl.close()
    pl_tr = _find_tool_result(turns_pl, "toolu_1")
    assert "中断" in pl_tr.content and "续跑" in pl_tr.content  # T6 占位生效
    # 注:占位对应【原 sync agent tool_use(toolu_1)】,续跑结果 inline 进【新 resume_agent
    # 工具的 tool_result】,不直接替换占位(brief:据实划定 → 占位保留 + inline 报告返回)。

    # ===== ③ resume:真实 discover → ResumeAgentTool.execute → resume_subagent(sync 分支)=====
    store2 = _make_store(tmp_path, session_id)
    try:
        ctrl = TurnController(
            _MainProvider(cfg.providers["p"]),
            on_event=_noop_event,
            on_status=_noop_status,
            store=store2,
        )
        mgr = SubagentManager(store=store2, controller=ctrl)

        # launch_async 间谍:sync 路径全程不应被调(阻塞 inline,非后台派发)。
        launch_async_calls: list[str] = []
        _orig_launch_async = mgr.launch_async

        async def _spy_launch_async(runner):  # noqa: ANN001
            launch_async_calls.append(runner.agent_id)
            return await _orig_launch_async(runner)

        monkeypatch.setattr(mgr, "launch_async", _spy_launch_async)

        # 真实 discover(T2):sub-A meta running ∈ RESUMABLE_STATUSES → 命中
        sub_dir = subagents_dir(tmp_path, session_id, tmp_path)
        resumable = find_resumable_subagents(sub_dir)
        resumable_ids = {m.agent_id for m in resumable}
        assert "sub-A" in resumable_ids  # discover 真实发现 sub-A
        # 且是 sync(meta.is_async=False)
        sub_a_meta = next(m for m in resumable if m.agent_id == "sub-A")
        assert sub_a_meta.is_async is False

        # 唯一 mock:SubagentRunner(避免真实 LLM);其余 resume/manager/store/ctrl 全真实
        monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

        deps = ResumeDeps(
            manager=mgr,
            root=tmp_path,
            session_id=session_id,
            project_root=tmp_path,
            worktree_name=None,
            agent_registry=_registry(),
            parent_provider=_MainProvider(cfg.providers["p"]),
            parent_registry=None,
            parent_gate=None,
            cfg=cfg,
            app=None,
            ctx=store2.ctx,
            spawn_depth=1,
            progress_cb=None,
        )
        tool = ResumeAgentTool(deps=deps)

        # 启动 execute(后台任务):sync 分支会阻塞在 runner.run() 等 proceed 放行
        task = asyncio.create_task(
            tool.execute(agent_id="sub-A", direction="继续把 a.txt 改成 X")
        )

        # ===== ④ 断言:阻塞证明 + sync 分支 + inline 返回 =====
        # (a) run() 已启动(resume_subagent 抵达 sync 分支 await runner.run()),但 execute
        #     未返回(proceed 未放行)→ 证明同步阻塞,非 fire-and-forget 后台派发。
        await asyncio.wait_for(_FakeRunner.started.wait(), timeout=1.0)  # type: ignore[union-attr]
        await asyncio.sleep(0)  # 让调度回 task 一次,确保它已进入 run() 的 await proceed
        assert not task.done()  # execute 仍阻塞在 await runner.run()
        assert not _FakeRunner.finished
        # 阻塞期间 launch_async 未被调(否则就是误走 async 后台分支)
        assert launch_async_calls == []

        # 放行 run() → 阻塞结束 → inline 返回报告
        _FakeRunner.proceed.set()  # type: ignore[union-attr]
        result_text = await asyncio.wait_for(task, timeout=1.0)

        # (b) inline 返回 report.text(sync 阻塞跑完的真结果),非 async ack 文本
        assert result_text == "续跑完成:已读 a.txt 并改成 X"
        assert "已派发" not in result_text  # 不是 async ack(manager.launch_async 回执文案)

        # (c) sync 分支:runner.run() 被调;is_async=False;复用 sub-A(非新 spawn id);
        #     resume_from 指向侧链(侧链当 history 加载)
        assert len(_FakeRunner.constructed) == 1
        fake_runner = _FakeRunner.constructed[0]
        assert fake_runner.run_called is True
        assert fake_runner.captured_kw["is_async"] is False
        assert fake_runner.agent_id == "sub-A"  # 复用原 id,非新 spawn
        assert fake_runner.captured_kw["resume_from"] == expected_sidechain

        # (d) launch_async 全程未被调(sync 走阻塞 inline,非后台)
        assert launch_async_calls == []

        # (e) 主 jsonl 无 enqueue / 无 task-notification(sync 不经 manager._inject)
        rows = store2.mainline_rows()
        assert _queue_ops(rows) == []
        assert _notif_rows(rows, "sub-A") == []

        # (f) 主 agent 未被流式消费(sync 结果 inline 回传,不经 notify_wake→主 agent)
        assert ctrl._provider.stream_called is False  # type: ignore[attr-defined]

        # (g) 阻塞已结束(run() 完成)
        assert _FakeRunner.finished is True
    finally:
        store2.close()
