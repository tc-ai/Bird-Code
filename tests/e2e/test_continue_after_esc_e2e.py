# tests/e2e/test_continue_after_esc_e2e.py
"""Task 23 E2E:ESC 中断(cancelled)的 async 子 agent,用户输入"继续"自动发现并续跑。

串联真实组件(非单点 mock):discover(T2, cancelled ∈ RESUMABLE_STATUSES)→"继续"路由
(T12 _maybe_route_continue,真方法绑 SimpleNamespace fake_self,真 store 扫描)→
ResumeAgentTool(T9)→ resume_subagent(T5)→ SubagentManager.launch_async + _supervise +
_inject(T6)→ 真实 SessionStore 主 jsonl 文件 IO + 真实 TurnController(notify_wake→
_process_wake→dequeue 消费)。

场景(brief Step 1):
  ① 造现场——主 jsonl 有 launched ack(agentId=sub-B)、无 enqueue;meta sub-B
     status=cancelled(ESC 中断)、is_async=True;侧链有种子 user + 一轮 assistant(待办
     清单头 = 进行中标志,过 T18 真实产出检查、且 cancelled 绕过 T17 交叉校验进续跑)。
  ② "继续"路由:T12 _maybe_route_continue("继续")→ 真 find_resumable_subagents 发现 sub-B
     (cancelled ∈ RESUMABLE_STATUSES 印证)→ 返回 True(拦截 submit)+ 触发 inject 回调
     (代表橙色 ResumePrompt + reminder,UI 副作用由 T10/T11 单测覆盖)。
  ③ 续跑闭环:授权假定 → 主 agent 调 ResumeAgentTool.execute(agent_id="sub-B",
     direction="继续")→ resume_subagent(cancelled 绕过 T17 交叉校验 → 进 launch 路径)→
     launch_async → FakeRunner.run()(completed)→ _inject(enqueue + isTaskNotification
     user + notify_wake)→ TurnController 消费 → dequeue。
  ④ 断言——cancelled 真被发现(both find_resumable & 路由)、"继续"路由真拦截、复用 sub-B
     (非新 id)、launch_async 被调、侧链当 history、主 jsonl 有 enqueue+notification、
     主 agent 回复 + dequeue。

与 T22(test_resume_async_e2e.py)的区别:
  - meta status=cancelled(ESC)而非 running(断电);侧链文案同样用「## 待办清单」头
    (cancelled 不走 T17 交叉校验,status=cancelled 已在 `meta.status not in
    ("completed","error","cancelled")` 的排除集 → 条件 False,绕过注入-as-completed,
    必进 launch 路径)。
  - 新增「继续」路由验证:用 T12 单测的 SimpleNamespace + types.MethodType 绑真实
    _maybe_route_continue 的范式,直接对真 store + 真 cancelled meta 调用 —— 验证路由
    真发现 cancelled(非 mock)。"继续"路由本身的单元语义(strip 匹配/dedup)由 T12 覆盖。

真实 vs mock 边界(据实划定):
  - 真实路径:find_resumable_subagents(cancelled 发现)/ _maybe_route_continue(真方法
    绑 fake_self,真 store 扫描,只 mock inject 回调代表 UI 副作用)/ resume_subagent /
    ResumeAgentTool / SubagentManager.launch_async + _supervise + _inject / 真实 meta +
    侧链 + 主 jsonl 文件 IO / TurnController notify_wake→_process_wake→run_agent_loop。
  - mock:resume_mod.SubagentRunner(_FakeRunner,其 run() 返固定 completed 报告,避免真实
    LLM 调用);主 agent provider(_MainProvider,固定文本 + Done,避免 LLM 智力与真实
    工具执行);_maybe_route_continue 的 _refresh_resume_prompt 回调(代表橙色 widget
    + reminder Turn 的 UI 副作用,具体验证属 T10/T11)。
  - 不验证:LLM 智力、真实工具副作用、Textual pilot UI 接线(均 mock/简化)。聚焦
    「cancelled → 发现(两层)→ 路由拦截 → 续跑闭环 → 完成注入 → 主 jsonl 有
    enqueue+notification+dequeue」这个端到端串联。
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from birdcode.agent.provider import Done, TextDelta, TokenUsage
from birdcode.agents import resume as resume_mod
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.discover import RESUMABLE_STATUSES, find_resumable_subagents
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
from birdcode.ui.app import BirdApp

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
            yield Done(
                usage=TokenUsage(input_tokens=5, output_tokens=3), stop_reason="end_turn"
            )

        return _gen()


# ---- mock SubagentRunner:捕获构造 kw(验证复用 id + 侧链 history)+ 固定 run() ----


class _FakeRunner:
    """SubagentRunner 替身:真实 _supervise / _inject 会访问 .defn / .agent_id / .prompt /
    .tool_use_id / .isolation / .description,故把这些从构造 kw 取出挂实例。

    run() 返固定 completed SubagentReport(避免真实 LLM)。constructed 类类列表验「复用 id」
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
            text="续跑完成:已按新方向把 b.txt 改成 Y",
            status="completed",
            duration_ms=9,
            tokens=7,
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
    r.register(
        AgentDefinition(
            name="general-purpose",
            description="通用子 agent",
            system_prompt="SP",
            run_in_background=True,
        )
    )
    return r


def _write_sub_b_meta(tmp_path: Path, session_id: str) -> None:
    """写真实 sub-B meta.json(status=cancelled = ESC 中断、is_async=True)。

    cancelled ∈ RESUMABLE_STATUSES → discover/resume 真实路径发现;且 resume_subagent
    T17 交叉校验排除集含 cancelled → 绕过注入-as-completed,必进续跑 launch 路径。
    """
    meta_path = subagent_meta_path(tmp_path, session_id, tmp_path, "sub-B")
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId="sub-B",
            agentType="general-purpose",
            description="被 ESC 中断的旧任务:处理 b.txt",
            toolUseId="(async-agent)",
            isAsync=True,
            status="cancelled",
        ),
    )


def _write_sub_b_sidechain(tmp_path: Path, session_id: str) -> Path:
    """写 sub-B 侧链(ESC 中断前的部分产出):种子 user + 一轮 assistant(待办清单头)。

    assistant 文本用「## 待办清单」头(runner.py 完成判据的反向标志 = 进行中)→ T17 交叉
    校验判未完成(即便读到也判续跑;而 cancelled 本就绕过 T17)。有真实(非 synthetic)
    assistant 产出 → T18 空侧链检查放行(_has_assistant_output=True)。
    """
    p = subagent_jsonl_path(tmp_path, session_id, tmp_path, "sub-B")
    user_line = (
        '{"type":"user","message":{"role":"user","content":'
        '[{"type":"text","text":"读 b.txt 并改成 Y"}]}}'
    )
    asst_line = (
        '{"type":"assistant","message":{"role":"assistant","content":'
        '[{"type":"text","text":"## 待办清单\\n- [x] 已读 b.txt\\n- [ ] 改成 Y"}]}}'
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
    (toolUseResult 内含 agentId == sub-B、status=launched)。

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


def _build_route_fake_self(store: SessionStore) -> SimpleNamespace:
    """构造覆盖 _maybe_route_continue 全部 self 属性的 fake(范式同 T12 单测)。

    Task 7 后:_maybe_route_continue 命中"继续"调 self._refresh_resume_prompt(force_show=True)
    (代表橙色 ResumePrompt + reminder Turn 的 UI 副作用,具体验证属 T10/T11)。
    返回的 ns 暴露 refresh_calls(记录 force_show 值)+ 绑真实 _maybe_route_continue
    方法(经 types.MethodType 绑定,使 self 正确传递)。
    """
    refresh_calls: list[bool] = []

    async def _fake_refresh(*, force_show: bool = False) -> None:
        refresh_calls.append(force_show)

    ns = SimpleNamespace(
        _store=store,
        _controller=object(),  # 非 None 即可(_maybe_route_continue 只校验 is None)
        _bg_tasks=set(),
    )
    ns.refresh_calls = refresh_calls
    ns._refresh_resume_prompt = _fake_refresh
    # 绑真实方法(未绑形式 → 用 types.MethodType 绑到 fake_self,使 self 正确传递)
    ns._maybe_route_continue = types.MethodType(BirdApp._maybe_route_continue, ns)
    return ns


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch, tmp_path):
    """侧链/subagent-store 落 tmp(paths.default_root → tmp_path),不污染 ~/.birdcode。"""
    from birdcode.session import paths

    monkeypatch.setattr(paths, "default_root", lambda: tmp_path)


# ---- E2E:ESC(cancelled)→ 发现 → "继续"路由 → 续跑闭环 ----


async def test_esc_cancelled_then_continue_rediscovers(tmp_path: Path, monkeypatch):
    """brief Step 1:造 ESC 现场 → 发现(cancelled)→ "继续"路由拦截 → 续跑 → 通知闭环。

    真实串联:find_resumable_subagents(cancelled 命中)→ _maybe_route_continue(真方法绑
    fake_self,真 store 扫描发现 cancelled)→ ResumeAgentTool.execute → resume_subagent
    (cancelled 绕过 T17)→ launch_async → _supervise → FakeRunner.run() → _inject(enqueue
    + isTaskNotification user + notify_wake)→ TurnController _process_wake 消费 → dequeue。
    """
    _FakeRunner.constructed.clear()
    session_id = "e2e-esc-continue"

    # ===== ① 造 ESC 现场:meta(cancelled/async)+ 侧链(种子 user + 进行中 assistant)=====
    _write_sub_b_meta(tmp_path, session_id)
    expected_sidechain = _write_sub_b_sidechain(tmp_path, session_id)

    # 印证 cancelled ∈ RESUMABLE_STATUSES(可续跑集合的源头真理)
    assert "cancelled" in RESUMABLE_STATUSES

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
                    "agentId": "sub-B",
                    "status": "launched",
                    "isAsync": True,
                }
            },
        )

        # ----- ESC 现场断言:有 launched ack、无 enqueue、无 notification -----
        rows_before = store.mainline_rows()
        assert len(_launched_ack_rows(rows_before, "sub-B")) == 1
        assert _queue_ops(rows_before) == []
        assert _notif_rows(rows_before) == []

        # ===== ② 发现(cancelled)—— 真实 find_resumable_subagents 扫到 sub-B =====
        sub_dir = subagents_dir(tmp_path, session_id, tmp_path)
        resumable = find_resumable_subagents(sub_dir)
        resumable_ids = {m.agent_id for m in resumable}
        assert "sub-B" in resumable_ids  # cancelled 真被 discover 发现
        # 印证具体 meta 的 status 字段确为 cancelled(非串进来的别的 status)
        sub_b_meta = next(m for m in resumable if m.agent_id == "sub-B")
        assert sub_b_meta.status == "cancelled"

        # ===== ③ "继续"路由发现它(T12 _maybe_route_continue,真方法绑 fake_self)=====
        # 用 T12 单测的 SimpleNamespace 范式,直接调真 _maybe_route_continue:真 strip 校验 +
        # 真 find_resumable_subagents 扫描(对真 cancelled meta)→ 命中 → 调 refresh 回调 +
        # 返回 True(拦截 submit)。refresh 回调本身 UI 副作用(橙色 widget + reminder Turn)由
        # T10/T11 单测覆盖;此处只验路由判定+发现+拦截,不重测 UI。
        route_self = _build_route_fake_self(store)
        handled = await BirdApp._maybe_route_continue(route_self, "继续")
        assert handled is True  # "继续"路由拦截(不透传 submit)
        # Task 7:force_show=True 被调(代表橙色 + reminder mount)
        assert route_self.refresh_calls == [True]
        # 对照:非"继续"输入不拦截(回归 strip 严格匹配)
        route_self2 = _build_route_fake_self(store)
        handled_other = await BirdApp._maybe_route_continue(route_self2, "继续分析")
        assert handled_other is False
        assert route_self2.refresh_calls == []

        # ===== ④ 续跑闭环:授权假定 → ResumeAgentTool.execute(sub-B, "继续")=====
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

        # 主 agent 调 resume_agent(sub-B, "继续")——授权假定(brief 明示)
        result_text = await tool.execute(agent_id="sub-B", direction="继续把 b.txt 改成 Y")

        # ===== ⑤ 断言闭环 =====
        # (a) launch_async 被调:ResumeAgentTool 透传 resume_subagent 的 async ack 文本
        #     (来自真实 manager.launch_async 的 LaunchAck.ack_text,含复用的 sub-B)
        assert "已派发" in result_text
        assert "sub-B" in result_text

        # (b) 复用 sub-B(非新生成 sub-{uuid})+ resume_from 指向侧链(侧链当 history 加载)
        #     关键:cancelled 经 resume_subagent 绕过 T17 交叉校验、T18 放行 → 抵达 launch 路径
        assert len(_FakeRunner.constructed) == 1
        fake_runner = _FakeRunner.constructed[0]
        assert fake_runner.agent_id == "sub-B"  # 复用原 id,非新 spawn
        assert fake_runner.captured_kw["resume_from"] == expected_sidechain
        assert fake_runner.captured_kw["is_async"] is True

        # (c) 真实 _supervise 跑完(run() + _inject):_live 清空、run() 被调
        await _wait_until_live_drained(mgr)
        assert fake_runner.run_called is True

        # (d) 主 jsonl 落盘 enqueue + isTaskNotification(真实 _inject 写入)
        rows_after_inject = store.mainline_rows()
        enq = [o for o in _queue_ops(rows_after_inject) if o.get("operation") == "enqueue"]
        assert len(enq) == 1
        assert enq[0].get("agentId") == "sub-B"
        assert enq[0].get("status") == "completed"  # FakeRunner.run() 报 completed(续跑完成)
        assert len(_notif_rows(rows_after_inject, "sub-B")) == 1

        # (e) bonus:notify_wake 真实驱动主 agent 消费 → assistant 响应 + dequeue
        await _wait_until_idle(ctrl)
        rows_final = store.mainline_rows()
        assert any(r.get("type") == "assistant" for r in rows_final)  # 主 agent 回复
        deq = [o for o in _queue_ops(rows_final) if o.get("operation") == "dequeue"]
        assert any(o.get("agentId") == "sub-B" for o in deq)  # 已消费标记
        # launched ack 仍在(未被破坏)
        assert len(_launched_ack_rows(rows_final, "sub-B")) == 1
    finally:
        store.close()
