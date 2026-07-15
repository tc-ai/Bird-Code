# tests/ui/test_resumable_reminder.py
"""Task 10: build_resumable_reminder — 可续跑子 agent 列表 → system reminder 文本。

纯函数测试(Step 1):id + 描述(≤100 截断)+ 空列表 → ""。
on_mount 注入接线另用 _inject_resumable_reminder 方法覆盖(轻量集成,见末尾)。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from birdcode.blocks import TextBlock
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.store import SessionStore
from birdcode.ui.app import BirdApp, build_resumable_reminder


def _meta(
    agent_id: str, status: str, prompt: str = "", description: str = ""
) -> SubagentMeta:
    return SubagentMeta(
        agent_id=agent_id,
        tool_use_id=f"tu-{agent_id}",
        status=status,  # type: ignore[arg-type]
        prompt=prompt,
        description=description,
    )


# —— Step 1: 纯函数 ——


def test_build_resumable_reminder_lists_ids_and_truncated_desc():
    metas = [
        _meta("sub-1", "running", prompt="分析鉴权模块的完整逻辑"),
        _meta("sub-2", "cancelled", prompt="x" * 200),
    ]
    text = build_resumable_reminder(metas)
    assert "sub-1" in text and "sub-2" in text
    assert "分析鉴权模块" in text
    # 描述截断 ≤100 + 前缀("agent_id=... status=... : " 约 34 字符)→ 整行 ≤160
    sub2_line = [ln for ln in text.splitlines() if "sub-2" in ln][0]
    assert len(sub2_line) <= 160


def test_build_resumable_reminder_empty_returns_empty_string():
    """空 metas → ""(不注入)。"""
    assert build_resumable_reminder([]) == ""


def test_build_resumable_reminder_falls_back_to_description():
    """prompt 空时回退 description。"""
    m = _meta("sub-3", "idle", prompt="", description="备份描述")
    text = build_resumable_reminder([m])
    assert "备份描述" in text


def test_build_resumable_reminder_no_desc_placeholder():
    """prompt 与 description 都空 → (无描述) 占位。"""
    m = _meta("sub-4", "launched", prompt="", description="")
    text = build_resumable_reminder([m])
    assert "无描述" in text


def test_build_resumable_reminder_truncates_to_exactly_100():
    """描述截断后恰为 100 字符(含 CJK 多字节也按字符计)。"""
    m = _meta("sub-5", "running", prompt="喵" * 150)
    text = build_resumable_reminder([m])
    line = [ln for ln in text.splitlines() if "sub-5" in ln][0]
    # 提取 ": " 之后的部分即截断后的描述
    desc = line.split(" : ", 1)[1]
    assert len(desc) == 100


# —— 注入接线:_inject_resumable_reminder(轻量集成,未绑定方法 + SimpleNamespace)——


def _make_store(tmp_path) -> SessionStore:
    ctx = SessionContext(session_id="resume", cwd="C:/p", version="0.1.0", git_branch=None)
    return SessionStore(ctx, tmp_path, root=tmp_path)


def _write_meta_file(store: SessionStore, agent_id: str, status: str, prompt: str) -> None:
    """写 agent-<id>.meta.json 到 store 的 subagents 目录。"""
    import json

    from birdcode.session import paths

    p = paths.subagent_meta_path(
        store.root, store.ctx.session_id, store.project_root, agent_id,
        worktree_name=store.worktree_name,
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "agentId": agent_id, "agentType": "general-purpose",
                "toolUseId": f"tu-{agent_id}", "status": status, "prompt": prompt,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_inject_resumable_reminder_appends_synthetic_turn(tmp_path):
    """扫到非终态 meta → 追加合成 Turn 到 controller.history(供首轮 LLM 上下文)。"""
    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "running", "分析模块")
    controller = SimpleNamespace(history=[])

    async def _noop_mount(_metas):  # T11: _inject 现还会调 _mount_resume_prompt,fake 接住
        return None

    fake_self = SimpleNamespace(
        _store=store, _controller=controller, _mount_resume_prompt=_noop_mount,
        _resume_reminder_injected=False,
    )

    await BirdApp._inject_resumable_reminder(fake_self)

    assert len(controller.history) == 1
    turn = controller.history[0]
    # 合成 user 消息,内容含 agent_id 与 <system-reminder> 包裹
    msg = turn.messages[0]
    assert msg.synthetic is True
    assert msg.role == "user"
    text = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
    assert "A1" in text and "分析模块" in text
    assert "<system-reminder>" in text


@pytest.mark.asyncio
async def test_inject_resumable_reminder_also_mounts_resume_prompt(tmp_path):
    """T11:扫到非终态 meta → 同时把 metas 传给 _mount_resume_prompt(UI 提示)。"""
    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "running", "分析模块")
    controller = SimpleNamespace(history=[])
    mounted: list = []

    async def _capture_mount(metas):
        mounted.append(metas)

    fake_self = SimpleNamespace(
        _store=store, _controller=controller, _mount_resume_prompt=_capture_mount,
        _resume_reminder_injected=False,
    )

    await BirdApp._inject_resumable_reminder(fake_self)

    assert len(controller.history) == 1  # reminder 仍注入
    assert len(mounted) == 1  # mount 调用一次
    assert mounted[0] and mounted[0][0].agent_id == "A1"


@pytest.mark.asyncio
async def test_inject_resumable_reminder_skips_when_empty(tmp_path):
    """无可续跑 meta → 不追加任何 Turn(空注入)。"""
    store = _make_store(tmp_path)  # 无 meta 文件
    controller = SimpleNamespace(history=[])
    fake_self = SimpleNamespace(
        _store=store, _controller=controller, _resume_reminder_injected=False,
    )

    await BirdApp._inject_resumable_reminder(fake_self)

    assert controller.history == []


@pytest.mark.asyncio
async def test_inject_resumable_reminder_no_store_returns_early():
    """无 store → 早返回(no-op)。"""
    controller = SimpleNamespace(history=[])
    fake_self = SimpleNamespace(_store=None, _controller=controller)
    await BirdApp._inject_resumable_reminder(fake_self)  # 不应抛
    assert controller.history == []


# —— T11:_mount_resume_prompt(挂橙色 ResumePrompt 到 #scroll)——


class _FakeScroll:
    """伪 #scroll 容器:收 mount(widget) → 记录挂载的 widget。"""

    def __init__(self) -> None:
        self.mounted: list = []

    async def mount(self, widget):  # noqa: D401
        self.mounted.append(widget)


@pytest.mark.asyncio
async def test_mount_resume_prompt_constructs_widget_with_truncated_desc():
    """非空 metas → mount ResumePrompt,描述取 prompt[:100](回退 description/占位)。"""
    from birdcode.ui.widgets.resume_prompt import ResumePrompt

    metas = [
        _meta("A1", "running", prompt="分析模块"),
        _meta("A2", "idle", prompt="", description="备用"),
        _meta("A3", "launched", prompt="x" * 200),
    ]
    scroll = _FakeScroll()
    fake_self = SimpleNamespace(
        query_one=lambda sel: scroll if sel == "#scroll" else None,
        query=lambda _sel: [],  # 去重检查:无已存在 ResumePrompt
    )

    await BirdApp._mount_resume_prompt(fake_self, metas)

    assert len(scroll.mounted) == 1
    prompt = scroll.mounted[0]
    assert isinstance(prompt, ResumePrompt)
    # 三条描述:prompt 直取 / 回退 description / 截断 100
    aids = [aid for aid, _ in prompt._desc]
    assert aids == ["A1", "A2", "A3"]
    assert prompt._desc[0] == ("A1", "分析模块")
    assert prompt._desc[1] == ("A2", "备用")
    assert len(prompt._desc[2][1]) == 100


@pytest.mark.asyncio
async def test_mount_resume_prompt_empty_metas_noop():
    """空 metas → 不挂载(no-op)。"""
    scroll = _FakeScroll()
    fake_self = SimpleNamespace(query_one=lambda sel: scroll)

    await BirdApp._mount_resume_prompt(fake_self, [])

    assert scroll.mounted == []


@pytest.mark.asyncio
async def test_mount_resume_prompt_best_effort_on_query_failure():
    """query_one 抛异常 → 不传播(best-effort,不杀 resume)。"""
    def _boom(_sel):
        raise RuntimeError("no #scroll")

    fake_self = SimpleNamespace(query_one=_boom)

    await BirdApp._mount_resume_prompt(fake_self, [_meta("A1", "running", prompt="x")])  # 不应抛


# —— T12 dedup:重复调用 _inject / _mount 不累积 reminder Turn / ResumePrompt widget ——


@pytest.mark.asyncio
async def test_inject_resumable_reminder_dedup_turn_across_repeated_calls(tmp_path):
    """T12 dedup(1):重复调 _inject_resumable_reminder → reminder Turn 只 append 一次。

    失败场景(修复前):on_mount 注入一次;用户每次"继续"再触发 _inject → 每次都 append
    相同 reminder Turn,LLM 上下文累积逐字重复的 <system-reminder> 块。
    修复后:self._resume_reminder_injected flag 首次置 True,后续调用跳过 append。
    """
    store = _make_store(tmp_path)
    _write_meta_file(store, "A1", "running", "分析模块")
    controller = SimpleNamespace(history=[])
    mount_calls: list = []

    async def _capture_mount(_metas):
        mount_calls.append(_metas)

    fake_self = SimpleNamespace(
        _store=store, _controller=controller, _mount_resume_prompt=_capture_mount,
        _resume_reminder_injected=False,
    )

    await BirdApp._inject_resumable_reminder(fake_self)
    await BirdApp._inject_resumable_reminder(fake_self)
    await BirdApp._inject_resumable_reminder(fake_self)

    # reminder Turn 不累积:3 次调用 → history 仍只 1 条
    assert len(controller.history) == 1
    # flag 置位(后续不再 append)
    assert fake_self._resume_reminder_injected is True
    # _mount_resume_prompt 仍每次被调(widget 去重在其内部处理)
    assert len(mount_calls) == 3


@pytest.mark.asyncio
async def test_mount_resume_prompt_skips_when_widget_already_present():
    """T12 dedup(2):#scroll 已有 ResumePrompt → 不重复 mount(避免橙色 widget 堆积)。

    失败场景(修复前):"继续"重复触发 _mount_resume_prompt → 每次都 mount 一个新 ResumePrompt,
    UI 橙色框堆积。
    修复后:mount 前查 query("ResumePrompt"),命中已有 → 直接 return 不重挂。
    """
    from birdcode.ui.widgets.resume_prompt import ResumePrompt

    existing = ResumePrompt(descriptions=[("A1", "旧任务")])
    scroll = _FakeScroll()
    # query 首次返回已存在的 widget(非空),应跳过 mount
    fake_self = SimpleNamespace(
        query_one=lambda sel: scroll if sel == "#scroll" else None,
        query=lambda _sel: [existing],
    )

    await BirdApp._mount_resume_prompt(fake_self, [_meta("A1", "running", prompt="x")])

    assert scroll.mounted == []  # 已存在 → 不重复 mount


@pytest.mark.asyncio
async def test_mount_resume_prompt_remounts_after_user_ignores_existing():
    """T12 dedup(2) 边界:用户按 i 忽略移除旧 widget 后(query 返回空)→ 重挂允许。

    widget 去重按"当前是否存在"判断,而非 flag:旧 widget 被 remove() 后 query 返回空,
    再次触发"继续"应重挂(UI 提示需要重新可见)。这与 reminder Turn 的 flag 去重不对称:
    Turn 永不重复 append,但 widget 允许重挂。
    """
    from birdcode.ui.widgets.resume_prompt import ResumePrompt

    scroll = _FakeScroll()
    # query 返回空列表(旧 widget 已被用户按 i 移除)
    fake_self = SimpleNamespace(
        query_one=lambda sel: scroll if sel == "#scroll" else None,
        query=lambda _sel: [],
    )

    await BirdApp._mount_resume_prompt(fake_self, [_meta("A1", "running", prompt="x")])

    assert len(scroll.mounted) == 1  # 旧 widget 已移除 → 重挂新 ResumePrompt
    assert isinstance(scroll.mounted[0], ResumePrompt)
