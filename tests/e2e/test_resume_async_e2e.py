# tests/e2e/test_resume_async_e2e.py
"""Task 22 E2E:async 子 agent 断电续跑完整闭环。

串联真实组件(非单点 mock):discover(T2)→ ResumeAgentTool(T9)→ resume_subagent(T5)
→ SubagentManager.launch_async + _supervise + _inject(T6)→ 真实 SessionStore 主 jsonl
文件 IO + 真实 TurnController(notify_wake→_process_wake→dequeue 消费)。

场景(brief Step 1):
  ① 造现场——主 jsonl 有 launched ack(agentId=sub-A)、无 enqueue;meta sub-A
     status=running、is_async=True;侧链有种子 user + 一轮 assistant(已读某文件)。
  ② resume→find_resumable_subagents 发现 sub-A→授权假定→主 agent 调
     ResumeAgentTool.execute(agent_id="sub-A", direction="继续")。
  ③ 断言——复用 sub-A(非新 id)、launch_async 被调、侧链被当 history(resume_from 指向
     侧链路径)、主 jsonl 最终有 enqueue + isTaskNotification。

真实 vs mock 边界(据实划定):
  - 真实路径:discover / resume_subagent / ResumeAgentTool / SubagentManager.launch_async
    + _supervise + _inject / 真实 meta + 侧链 + 主 jsonl 文件 IO / TurnController
    notify_wake→_process_wake→run_agent_loop(主 agent 消费→dequeue)。
  - mock:resume_mod.SubagentRunner(其 run() 返固定 SubagentReport,避免真实 LLM 调用);
    主 agent provider(返固定文本 + Done,避免 LLM 智力与真实工具执行)。
  - 不验证:LLM 智力、真实工具副作用(均 mock)。聚焦「断电→发现→续跑→完成注入→主 jsonl
    有 enqueue+notification」这个闭环的组件串联。

mock 范式:抄 tests/agents/test_resume_idempotency.py(_FakeRunner + 真实 meta/侧链/paths,
monkeypatch resume_mod.SubagentRunner)+ tests/agents/test_phase2_e2e.py(真实
SubagentManager + TurnController + SessionStore + mock 主 provider + _wait_until_* 轮询)。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from birdcode.agent.provider import Done, TextDelta, TokenUsage
from birdcode.agents import resume as resume_mod
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.discover import find_resumable_subagents
from birdcode.agents.manager import SubagentManager
from birdcode.agents.registry import AgentRegistry
from birdcode.agents.resume import ResumeDeps
from birdcode.agents.runner import SubagentReport
from birdcode.blocks import TextBlock
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.conversation import Message, TurnController
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.paths import (
    subagent_jsonl_path,
    subagent_meta_path,
    subagents_dir,
)
from birdcode.session.store import SessionStore
from birdcode.session.subagent_meta import write_subagent_meta
from birdcode.tools.resume_agent_tool import ResumeAgentTool

# ---- mock 主 agent provider:固定文本 + Done,避免真实 LLM(只验闭环串联)----


class _MainProvider:
    """主 agent 流式桩:notify_wake→_process_wake 跑一轮时 yield 固定回复 + Done。

    不发工具调用 → 不需 executor;run_agent_loop 只追加一条 assistant 行后正常收尾。
    """

    def __init__(self, profile: ProviderProfile, *, text: str = "已收到续跑完成通知。") -> None:
        self.profile = profile
        self._text = text

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            yield TextDelta(text=self._text)
            yield Done(usage=TokenUsage(input_tokens=5, output_tokens=3), stop_reason="end_turn")

        return _gen()


# ---- mock SubagentRunner:捕获构造 kw(验证复用 id + 侧链 history)+ 固定 run() ----


class _FakeRunner:
    """SubagentRunner 替身:真实 _supervise / _inject 会访问 .defn / .agent_id / .prompt /
    .tool_use_id / .isolation / .description,故把这些从构造 kw 取出挂实例。

    run() 返固定 completed SubagentReport(避免真实 LLM)。constructed 类级列表验「复用 id」
    与「构造次数」。
    """

    constructed: list[_FakeRunner] = []

    def __init__(self, **kw: object) -> None:
        self.captured_kw = kw
        # _supervise / _inject 在 manager.py 内访问的真实属性:
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
    ctx = SessionContext(session_id=session_id, cwd=str(tmp_path), version="0.1.0", git_branch=None)
    return SessionStore(ctx, tmp_path, root=tmp_path)


def _registry() -> AgentRegistry:
    r = AgentRegistry()
    r.register(
        AgentDefinition(
            name="general-purpose",
            description="通用子 agent",
            system_prompt="SP",
            run_in_background=True,
        )
    )
    return r


def _write_sub_a_meta(tmp_path: Path, session_id: str, *, status: str = "running") -> None:
    """写真实 sub-A meta.json(status=running、is_async=True)→ discover/resume 走真实路径。"""
    meta_path = subagent_meta_path(tmp_path, session_id, tmp_path, "sub-A")
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId="sub-A",
            agentType="general-purpose",
            description="旧任务:处理 a.txt",
            toolUseId="(async-agent)",
            isAsync=True,
            status=status,
        ),
    )


def _write_sub_a_sidechain(tmp_path: Path, session_id: str) -> Path:
    """写 sub-A 侧链:种子 user + 一轮 assistant(已读 a.txt)。

    assistant 文本用「## 待办清单」头(runner.py 完成判据的反向标志)→ T17 交叉校验判
    未完成、不误注入完成,使续跑派发路径抵达 launch_async。有真实(非 synthetic)assistant
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


def _launched_ack_rows(rows: list[dict], agent_id: str) -> list[dict]:
    """主 jsonl 里 launched ack 行:_AgentTool 异步派发回执的 tool_result user 行
    (toolUseResult 内含 agentId == sub-A、status=launched)。

    encode_user 把 tool_use_results(dict-by-tool_use_id)整体写到行级 toolUseResult 字段,
    即 toolUseResult = {toolu_x: {agentId, status, isAsync, ...}};本辅助遍历内层 dict
    匹配(容忍扁平或嵌套结构)。
    """
    out = []
    for r in rows:
        if not isinstance(r, dict) or r.get("type") != "user":
            continue
        tur = r.get("toolUseResult")
        if not isinstance(tur, dict):
            continue
        # 嵌套(dict-by-id):值都是 dict → 取内层;否则视作扁平(容忍)。
        inner = list(tur.values()) if all(isinstance(v, dict) for v in tur.values()) else [tur]
        for v in inner:
            if (
                isinstance(v, dict)
                and v.get("agentId") == agent_id
                and v.get("status") == "launched"
            ):
                out.append(r)
    return out


async def _wait_until_live_drained(mgr: SubagentManager, *, timeout: float = 2.0) -> None:
    """轮询 manager._live 直到空(_supervise 收尾 = run() + _inject 都跑完)。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while mgr._live:  # noqa: SLF001
        if loop.time() > deadline:
            raise TimeoutError("SubagentManager._live 未在限时内清空")
        await asyncio.sleep(0.005)


async def _wait_until_idle(ctrl: TurnController, *, timeout: float = 2.0) -> None:
    """notify_wake 的 _drain 是后台 fire-and-forget;轮询 busy 直到 False。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while ctrl.busy:
        if loop.time() > deadline:
            raise TimeoutError("TurnController 未在限时内回到空闲")
        await asyncio.sleep(0.005)


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch, tmp_path):
    """侧链/subagent-store 落 tmp(paths.default_root → tmp_path),不污染 ~/.birdcode。"""
    from birdcode.session import paths

    monkeypatch.setattr(paths, "default_root", lambda: tmp_path)


# ---- E2E:async 断电续跑闭环 ----


async def test_async_interrupted_resumes_and_closes_loop(tmp_path: Path, monkeypatch):
    """brief Step 1:造现场 → resume → 断言闭环(复用 id / launch / 侧链 history /
    主 jsonl enqueue+notification)。

    真实串联:find_resumable_subagents → ResumeAgentTool.execute → resume_subagent →
    SubagentManager.launch_async → _supervise → FakeRunner.run() → _inject(enqueue +
    isTaskNotification user + notify_wake)→ TurnController _process_wake 消费 → dequeue。
    """
    _FakeRunner.constructed.clear()
    session_id = "e2e-async"

    # ===== ① 造现场:meta(running/async)+ 侧链(种子 user + 已读 a.txt 的 assistant)=====
    _write_sub_a_meta(tmp_path, session_id, status="running")
    expected_sidechain = _write_sub_a_sidechain(tmp_path, session_id)

    # 真实 SessionStore(主 jsonl)+ 真实 SubagentManager + 真实 TurnController
    cfg = _cfg()
    main_provider = _MainProvider(cfg.providers["p"])
    store = _make_store(tmp_path, session_id)
    try:
        ctrl = TurnController(
            main_provider, on_event=_noop_event, on_status=_noop_status, store=store
        )
        mgr = SubagentManager(store=store, controller=ctrl)

        # 主 jsonl 的 launched ack:_AgentTool 异步派发回执的 tool_result(有 agentId、无 enqueue)
        await store.append(
            Message(role="user", content=[TextBlock(text="")]),
            tool_use_results={
                "toolu_launch": {
                    "agentId": "sub-A",
                    "status": "launched",
                    "isAsync": True,
                }
            },
        )

        # ----- 断电现场断言:有 launched ack、无 enqueue、无 notification -----
        rows_before = store.mainline_rows()
        assert len(_launched_ack_rows(rows_before, "sub-A")) == 1
        assert _queue_ops(rows_before) == []
        assert _notif_rows(rows_before) == []

        # ===== ② resume:发现 sub-A(真实 discover)→ 授权假定 → ResumeAgentTool.execute =====
        # 真实 discover(T2):sub-A meta status=running ∈ RESUMABLE_STATUSES → 命中
        sub_dir = subagents_dir(tmp_path, session_id, tmp_path)
        resumable = find_resumable_subagents(sub_dir)
        resumable_ids = {m.agent_id for m in resumable}
        assert "sub-A" in resumable_ids  # discover 真实发现 sub-A

        # 唯一 mock:SubagentRunner(避免真实 LLM);其余 resume/manager/store/ctrl 全真实
        monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

        deps = ResumeDeps(
            manager=mgr,
            root=tmp_path,
            session_id=session_id,
            project_root=tmp_path,
            worktree_name=None,
            agent_registry=_registry(),
            parent_provider=main_provider,
            parent_registry=None,
            parent_gate=None,
            cfg=cfg,
            app=None,
            ctx=store.ctx,
            spawn_depth=1,
            progress_cb=None,
        )
        tool = ResumeAgentTool(deps=deps)

        # 主 agent 调 resume_agent(sub-A, "继续")——授权假定(brief 明示)
        result_text = await tool.execute(agent_id="sub-A", direction="继续把 a.txt 改成 X")

        # ===== ③ 断言闭环 =====
        # (a) launch_async 被调:ResumeAgentTool 透传 resume_subagent 的 async ack 文本
        #     (来自真实 manager.launch_async 的 LaunchAck.ack_text,含复用的 sub-A)
        assert "已派发" in result_text
        assert "sub-A" in result_text

        # (b) 复用 sub-A(非新生成 sub-{uuid})+ resume_from 指向侧链(侧链当 history 加载)
        assert len(_FakeRunner.constructed) == 1
        fake_runner = _FakeRunner.constructed[0]
        assert fake_runner.agent_id == "sub-A"  # 复用原 id,非新 spawn
        assert fake_runner.captured_kw["resume_from"] == expected_sidechain
        assert fake_runner.captured_kw["is_async"] is True

        # (c) 真实 _supervise 跑完(run() + _inject):_live 清空、run() 被调
        await _wait_until_live_drained(mgr)
        assert fake_runner.run_called is True

        # (d) 主 jsonl 落盘 enqueue + isTaskNotification(真实 _inject 写入)
        rows_after_inject = store.mainline_rows()
        enq = [o for o in _queue_ops(rows_after_inject) if o.get("operation") == "enqueue"]
        assert len(enq) == 1
        assert enq[0].get("agentId") == "sub-A"
        assert enq[0].get("status") == "completed"  # FakeRunner.run() 报 completed
        assert len(_notif_rows(rows_after_inject, "sub-A")) == 1

        # (e) bonus:notify_wake 真实驱动主 agent 消费 → assistant 响应 + dequeue
        await _wait_until_idle(ctrl)
        rows_final = store.mainline_rows()
        assert any(r.get("type") == "assistant" for r in rows_final)  # 主 agent 回复
        deq = [o for o in _queue_ops(rows_final) if o.get("operation") == "dequeue"]
        assert any(o.get("agentId") == "sub-A" for o in deq)  # 已消费标记
        # launched ack 仍在(未被破坏)
        assert len(_launched_ack_rows(rows_final, "sub-A")) == 1
    finally:
        store.close()
