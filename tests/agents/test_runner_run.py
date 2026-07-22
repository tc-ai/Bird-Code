# tests/agents/test_runner_run.py
import json

import pytest

from birdcode.agent.provider import Done, Error, TextDelta, TokenUsage, ToolCallStart
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.runner import SubagentRunner
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.session.models import SessionContext
from birdcode.session.paths import subagent_meta_path
from birdcode.session.subagent_meta import read_subagent_meta


class _FakeProvider:
    """实现 StreamingProvider 协议:yield 一段文本 + Done,无工具。"""

    def __init__(self, profile, text="子 agent 完成了任务"):
        self.profile = profile
        self._text = text

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            yield TextDelta(text=self._text)
            yield Done(usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="end_turn")

        return _gen()


class _ToolFakeProvider:
    """yield 一个 ToolCallStart + Done → 验证 tool_use_count 计数。"""

    def __init__(self, profile):
        self.profile = profile

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            yield ToolCallStart(id="call_x", name="Read", args='{"path": "a.py"}')
            yield Done(usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="end_turn")

        return _gen()


class _OneToolThenTextProvider:
    """第一轮发 ToolCallStart(失败一次),其后发文本 → 正常完成,计 1 次。"""

    def __init__(self, profile):
        self.profile = profile
        self._round = 0

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            if self._round == 0:
                self._round += 1
                yield ToolCallStart(id="c1", name="Read", args='{"path": "a.py"}')
                yield Done(
                    usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="tool_use"
                )
            else:
                yield TextDelta(text="完成")
                yield Done(
                    usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="end_turn"
                )

        return _gen()


class _ErrorThenRecoverProvider:
    """第 1 轮 yield ToolCallStart + Error(模拟 base_llm 已 yield token 后不重试、Error+return);

    tool_uses 非空 → agent_loop 继续下一轮;第 2 轮 yield 文本 + Done 恢复完成。验证 Done 清除
    sticky error_msg(否则恢复成功被误报 status='error',#2 的反向 false-negative)。
    """

    def __init__(self, profile):
        self.profile = profile
        self._round = 0

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            if self._round == 0:
                self._round += 1
                yield ToolCallStart(id="c1", name="Read", args='{"path": "a.py"}')
                yield Error(message="流中途出错")
            else:
                yield TextDelta(text="恢复完成")
                yield Done(
                    usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="end_turn"
                )

        return _gen()


class _CacheBearingProvider:
    """yield Done 携带归一后的全量 input_tokens + 缓存字段(复刻 anthropic_provider 修复后输出)。

    全量 1333 = miss 629 + cache_read 704;用于验证进度卡 ↓ tokens 去缓存显示 629(非 1333)。
    """

    def __init__(self, profile):
        self.profile = profile

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            yield TextDelta(text="完成")
            yield Done(
                usage=TokenUsage(
                    input_tokens=1333,
                    output_tokens=5,
                    cache_read_tokens=704,
                    cache_creation_tokens=0,
                ),
                stop_reason="end_turn",
            )

        return _gen()


def _cfg():
    return AppConfig(
        providers={
            "p": ProviderProfile(
                name="p", protocol="anthropic", model="m", base_url="http://x", api_key="k"
            )
        },
        default="p",
    )


def _ctx():
    return SessionContext(session_id="s1", cwd=".", version="", git_branch=None)


def _defn():
    return AgentDefinition(name="general-purpose", description="d", system_prompt="SP")


@pytest.fixture(autouse=True)
def _patch_build_provider(monkeypatch):
    """避免真实 API:把 runner.build_provider 换成 fake。可变 holder 改写子 agent 文本/工具。"""
    from birdcode.agents import runner

    state = {"text": "子 agent 完成了任务", "use_tool": False}  # 默认子 agent 输出

    def _fake(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        if state.get("recover"):
            return _ErrorThenRecoverProvider(profile=profile)
        if state.get("one_tool"):
            return _OneToolThenTextProvider(profile=profile)
        if state.get("use_tool"):
            return _ToolFakeProvider(profile=profile)
        if state.get("cache_usage"):
            return _CacheBearingProvider(profile=profile)
        return _FakeProvider(profile=profile, text=state["text"])

    monkeypatch.setattr(runner, "build_provider", _fake)
    return state


async def test_run_completes_and_writes_sidechain(tmp_path):
    parent_provider = _FakeProvider(profile=_cfg().providers["p"])
    r = SubagentRunner(
        defn=_defn(),
        prompt="做某事",
        description="d",
        tool_use_id="call_1",
        model_override="",
        spawn_depth=1,
        is_async=False,
        parent_provider=parent_provider,
        parent_registry=None,  # None → 空子 registry
        parent_gate=None,
        cfg=_cfg(),
        app=None,
        ctx=_ctx(),
        project_root=tmp_path,
        root=tmp_path,
    )
    report = await r.run()
    assert report.status == "completed"
    assert report.is_completed is True
    assert "完成了任务" in report.text
    # 侧链 jsonl 已写(至少 2 行:user prompt + assistant)
    sidechain = r.sidechain_path
    assert sidechain.exists()
    lines = [ln for ln in sidechain.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 2
    # meta 已 completed
    meta = read_subagent_meta(subagent_meta_path(tmp_path, "s1", tmp_path, r.agent_id))
    assert meta is not None and meta.status == "completed"
    assert meta.is_async is False


async def test_run_checklist_marks_not_completed(tmp_path, _patch_build_provider):
    _patch_build_provider["text"] = "## 待办清单\n- 主 agent 接手 X"
    parent_provider = _FakeProvider(profile=_cfg().providers["p"])
    r = SubagentRunner(
        defn=_defn(),
        prompt="p",
        description="d",
        tool_use_id="call_2",
        model_override="",
        spawn_depth=1,
        is_async=False,
        parent_provider=parent_provider,
        parent_registry=None,
        parent_gate=None,
        cfg=_cfg(),
        app=None,
        ctx=_ctx(),
        project_root=tmp_path,
        root=tmp_path,
    )
    report = await r.run()
    assert report.status == "completed"  # 跑到底(生命周期)
    assert report.is_completed is False  # 任务语义:未完成,交清单
    assert "待办清单" in report.text


async def test_run_guardrail_failure_reports_error_with_count(tmp_path, _patch_build_provider):
    """连续同失败熔断(_MAX_SAME_FAILURES=3)→ run_agent_loop emit(Error)+正常 return(不抛)。

    #2:修复后报 status='error'(旧实现误报 completed);#10:error 报告保留真实 tool_use_count=3
    (旧实现硬编码 0)。_ToolFakeProvider 每轮发同一 ToolCallStart(Read)+空 registry → 同签名
    失败 3 次触发熔断,恰好计 3 次。
    """
    _patch_build_provider["use_tool"] = True
    parent_provider = _FakeProvider(profile=_cfg().providers["p"])
    r = SubagentRunner(
        defn=_defn(),
        prompt="p",
        description="d",
        tool_use_id="call_3",
        model_override="",
        spawn_depth=1,
        is_async=False,
        parent_provider=parent_provider,
        parent_registry=None,
        parent_gate=None,
        cfg=_cfg(),
        app=None,
        ctx=_ctx(),
        project_root=tmp_path,
        root=tmp_path,
    )
    report = await r.run()
    assert report.status == "error"  # 熔断 → error(非 completed)
    assert report.is_completed is False
    assert report.tool_use_count == 3  # 出错前累计保留(非 0)
    assert "出错" in report.text  # 护栏消息经「子 agent 出错:」回传


async def test_run_counts_tool_use_on_clean_completion(tmp_path, _patch_build_provider):
    """单次工具调用后正常完成(非熔断)→ tool_use_count=1,正常计数仍工作。"""
    _patch_build_provider["one_tool"] = True
    parent_provider = _FakeProvider(profile=_cfg().providers["p"])
    r = SubagentRunner(
        defn=_defn(),
        prompt="p",
        description="d",
        tool_use_id="call_3b",
        model_override="",
        spawn_depth=1,
        is_async=False,
        parent_provider=parent_provider,
        parent_registry=None,
        parent_gate=None,
        cfg=_cfg(),
        app=None,
        ctx=_ctx(),
        project_root=tmp_path,
        root=tmp_path,
    )
    report = await r.run()
    assert report.status == "completed"
    assert report.tool_use_count == 1


async def test_run_recovers_from_midstream_error(tmp_path, _patch_build_provider):
    """流中途 Error(已 yield token,base_llm 不重试→Error+return)+ 后续轮恢复完成 → 报 completed。

    #2 回归:Done 必须清除 sticky error_msg,否则恢复成功被误报 status='error'(反向 false-negative)。
    第 1 轮发 ToolCallStart+Error(tool_uses 非空→agent_loop 不 return、继续下一轮);
    第 2 轮发文本+Done 恢复 → Done 清 error_msg → completed。
    """
    _patch_build_provider["recover"] = True
    parent_provider = _FakeProvider(profile=_cfg().providers["p"])
    r = SubagentRunner(
        defn=_defn(),
        prompt="p",
        description="d",
        tool_use_id="call_3c",
        model_override="",
        spawn_depth=1,
        is_async=False,
        parent_provider=parent_provider,
        parent_registry=None,
        parent_gate=None,
        cfg=_cfg(),
        app=None,
        ctx=_ctx(),
        project_root=tmp_path,
        root=tmp_path,
    )
    report = await r.run()
    assert report.status == "completed"  # 恢复完成(非 error)
    assert "恢复完成" in report.text


async def test_child_provider_gets_mcp_instructions(tmp_path, _patch_build_provider):
    """子 agent provider 经 build_provider 透传 mcp_instructions(spec §8.2)。

    主 provider 以懒回调(C2 = app.mcp_instructions 绑定方法)形式透传(build_provider 形参
    类型 Callable[[], dict[str, str]] | None,base_llm 在 stream 时调用)。子 provider 同契约:
    应传【绑定方法本身】而非 eager 调用结果(dict 会让 base_llm._system_text 调用时 TypeError)。
    故断言 captured 是 callable 且解析为 app.mcp_instructions() 的返回值。
    """
    from birdcode.agents import runner

    captured: dict[str, object] = {}

    def _spy(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        captured["mcp"] = mcp_instructions
        return _FakeProvider(profile=profile)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "build_provider", _spy)

        class _AppStub:
            def mcp_instructions(self) -> dict[str, str]:
                return {"srv": "hello"}

        r = SubagentRunner(
            defn=_defn(),
            prompt="p",
            description="d",
            tool_use_id="call_m",
            model_override="",
            spawn_depth=1,
            is_async=False,
            parent_provider=_FakeProvider(profile=_cfg().providers["p"]),
            parent_registry=None,
            parent_gate=None,
            cfg=_cfg(),
            app=_AppStub(),
            ctx=_ctx(),
            project_root=tmp_path,
            root=tmp_path,
        )
        await r.run()

    mcp = captured.get("mcp")
    assert callable(mcp)  # 懒回调(绑定方法),非 eager dict
    assert mcp() == {"srv": "hello"}  # 解析为 app.mcp_instructions() 的返回值


async def test_run_progress_cb_failure_isolated(tmp_path):
    """progress_cb 抛异常 → 不杀子 agent,run 正常完成(不伪装成 error)。"""
    parent_provider = _FakeProvider(profile=_cfg().providers["p"])

    async def _bad_progress(_p):
        raise RuntimeError("UI hiccup")

    r = SubagentRunner(
        defn=_defn(),
        prompt="p",
        description="d",
        tool_use_id="call_4",
        model_override="",
        spawn_depth=1,
        is_async=False,
        parent_provider=parent_provider,
        parent_registry=None,
        parent_gate=None,
        cfg=_cfg(),
        app=None,
        ctx=_ctx(),
        project_root=tmp_path,
        root=tmp_path,
        progress_cb=_bad_progress,
    )
    report = await r.run()
    assert report.status == "completed"  # 非 error
    assert report.is_completed is True


async def test_progress_card_shows_current_context(tmp_path, _patch_build_provider):
    """进度卡 ↓ = 最近一轮 full input(当前上下文占用),非累计、非去缓存。

    provider 归一后 usage.input_tokens 即全量;卡片取最近一轮 Done 的全量作「当前上下文」,
    bounded by 窗口(用户要的口径:占用/上限,不会像累计那样超过窗口)。
    """
    _patch_build_provider["cache_usage"] = True
    cfg = _cfg()
    parent_provider = _FakeProvider(profile=cfg.providers["p"])
    captured: list = []

    async def _cap(p):
        captured.append(p)

    r = SubagentRunner(
        defn=_defn(),
        prompt="p",
        description="d",
        tool_use_id="call_c",
        model_override="",
        spawn_depth=1,
        is_async=False,
        parent_provider=parent_provider,
        parent_registry=None,
        parent_gate=None,
        cfg=cfg,
        app=None,
        ctx=_ctx(),
        project_root=tmp_path,
        root=tmp_path,
        progress_cb=_cap,
    )
    report = await r.run()
    # 卡片 ↓ = 最近一轮 full = 1333(非累计、非去缓存 629)
    assert captured and captured[-1].context_tokens == 1333
    assert captured[-1].context_window == cfg.context_window
    # totalTokens 仍是累计全量 input+output = 1333 + 5(总开销,与本卡口径不同)
    assert report.tokens == 1338


async def test_subagent_wires_compaction_and_records_event(tmp_path, monkeypatch):
    """子 agent 接 ContextManager(enable_snip=False) + 压缩事件落侧链(compact_event 行)。

    monkeypatch run_agent_loop 模拟「压缩发生」(emit CompactionEnd),验证:
    ① runner 构造了非 None 的 ContextManager 且 enable_snip=False(只全量压缩、不 snip);
    ② _emit 收到 CompactionEnd → store.append_compaction_event 写 compact_event 行;
    ③ 该行被 decode_lines 跳过(type=system),不破坏侧链加载。
    """
    from birdcode.agent.context import ContextManager
    from birdcode.agent.provider import CompactionEnd
    from birdcode.agents import runner as runner_mod
    from birdcode.session.subagent_store import read_subagent_transcript

    parent_provider = _FakeProvider(profile=_cfg().providers["p"])
    captured: dict = {}

    async def _fake_loop(*, context, emit, **_):  # noqa: ANN001
        captured["context"] = context
        await emit(CompactionEnd(reason="auto", summary="已压缩 100→50 token", fell_back=False))

    monkeypatch.setattr(runner_mod, "run_agent_loop", _fake_loop)

    r = SubagentRunner(
        defn=_defn(),
        prompt="p",
        description="d",
        tool_use_id="call_z",
        model_override="",
        spawn_depth=1,
        is_async=False,
        parent_provider=parent_provider,
        parent_registry=None,
        parent_gate=None,
        cfg=_cfg(),
        app=None,
        ctx=_ctx(),
        project_root=tmp_path,
        root=tmp_path,
    )
    await r.run()

    # ① 接了 ContextManager、关了 snip(只全量压缩)
    ctx = captured.get("context")
    assert isinstance(ctx, ContextManager)
    assert ctx._enable_snip is False
    # ② 侧链有 compact_event 行
    sidechain = r.sidechain_path
    assert sidechain.exists()
    raw = sidechain.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(ln) for ln in raw if ln.strip()]
    events = [row for row in rows if row.get("subtype") == "compact_event"]
    assert len(events) == 1
    assert events[0]["compact"]["summary"] == "已压缩 100→50 token"
    assert events[0]["compact"]["trigger"] == "auto"
    assert events[0]["isSidechain"] is True
    # ③ 事件行被 decode 跳过,不破坏侧链加载(无崩溃)
    assert isinstance(read_subagent_transcript(sidechain), list)


def test_system_override_appends_worktree_addendum():
    """worktree 子 agent 的 system_prompt 追加产物报告提示;非 worktree 不加。

    抽成 _system_override 函数单测(免真 git worktree):worktree → SP+addendum;否则原 SP。
    """
    from birdcode.agents.runner import _WORKTREE_ADDENDUM, _system_override

    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")
    assert _system_override(defn, None) == "SP"
    assert _system_override(defn, "worktree") == "SP" + _WORKTREE_ADDENDUM
    over = _system_override(defn, "worktree")
    assert "worktree" in over and "报告" in over and "文件" in over
