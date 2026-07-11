# tests/test_agent_loop.py
import logging

import pytest

from birdcode.agent.agent_loop import _summarize_tool, run_agent_loop
from birdcode.agent.provider import (
    Done,
    Error,
    TextDelta,
    TokenUsage,
    ToolCallStart,
)
from birdcode.agent.provider import (
    ToolResult as ToolResultEvent,
)
from birdcode.blocks import TextBlock, ToolResultBlock
from birdcode.conversation import Message, Turn
from birdcode.tools.executor import ToolExecutor, default_registry


class _ScriptProvider:
    """按调用次序返回预设事件序列(每次 stream 弹一个脚本)。"""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def stream(self, messages, *, history):
        self.calls += 1
        for e in self._scripts.pop(0):
            yield e


async def _run(p, turn):
    events = []

    async def emit(ev):
        events.append(ev)

    await run_agent_loop(
        provider=p,
        turn=turn,
        history=[],
        executor=ToolExecutor(default_registry()),
        emit=emit,
    )
    return events


@pytest.mark.asyncio
async def test_single_turn_no_tool_use():
    p = _ScriptProvider([[TextDelta(text="hi"), Done(usage=TokenUsage())]])
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])
    evs = await _run(p, turn)
    assert any(isinstance(e, TextDelta) for e in evs)
    assert isinstance(evs[-1], Done)
    assert len(turn.messages) == 2  # user + assistant
    assert turn.messages[-1].role == "assistant"
    assert any(isinstance(b, TextBlock) for b in turn.messages[-1].content)


@pytest.mark.asyncio
async def test_tool_use_triggers_second_round():
    """第 1 轮:tool_use;第 2 轮:最终 text。验证 tool_result 回传、续了第 2 轮。"""
    p = _ScriptProvider(
        [
            [
                ToolCallStart(id="t1", name="echo", args='{"text": "hello"}'),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), Done(usage=TokenUsage())],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])
    evs = await _run(p, turn)
    assert p.calls == 2
    # assistant(tool_use) → user(tool_result) → assistant(text)
    assert len(turn.messages) == 4
    assert any(isinstance(b, ToolResultBlock) for b in turn.messages[2].content)
    tr = next(b for b in turn.messages[2].content if isinstance(b, ToolResultBlock))
    assert tr.tool_use_id == "t1" and tr.content == "hello" and tr.is_error is False
    assert any(isinstance(e, ToolResultEvent) and e.id == "t1" for e in evs)


# ---- stage2 polish(§7): truncated 透传 + provider 类型收紧 ----


@pytest.mark.asyncio
async def test_truncated_propagated_to_tool_result_event():
    """executor 截断结果 → ToolResult 事件 truncated=True。"""
    from pydantic import BaseModel

    from birdcode.tools import Tool, ToolRegistry

    class _BigInput(BaseModel):
        text: str

    class _BigTool(Tool):
        name = "big"
        description = "返回超长文本,触发截断"
        parameters = _BigInput
        kind = "read"
        parallel_safe = True

        async def execute(self, **args):
            return "x" * 5000

    reg = ToolRegistry()
    reg.register(_BigTool())
    ex = ToolExecutor(reg, char_limit=100)

    p = _ScriptProvider(
        [
            [
                ToolCallStart(id="b1", name="big", args='{"text": "y"}'),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [TextDelta(text="ok"), Done(usage=TokenUsage())],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="z")])])
    events = []

    async def emit(ev):
        events.append(ev)

    await run_agent_loop(
        provider=p,
        turn=turn,
        history=[],
        executor=ex,
        emit=emit,
    )
    tr_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert tr_events and tr_events[0].truncated is True


# ---- stage3: ToolResult 事件透传完整正文 output(供 UI 折叠区显示)----


@pytest.mark.asyncio
async def test_tool_result_event_carries_output_and_truncated():
    """ToolResult UI 事件须携带截断后的完整正文 + truncated 标志。"""
    from pydantic import BaseModel

    from birdcode.tools import Tool, ToolRegistry

    class _BigInput(BaseModel):
        text: str

    class _BigTool(Tool):
        name = "big"
        description = "返回超长文本,触发截断"
        parameters = _BigInput
        kind = "read"
        parallel_safe = True

        async def execute(self, **args):
            return "x" * 100_000

    r = ToolRegistry()
    r.register(_BigTool())
    ex = ToolExecutor(r, char_limit=1000)

    p = _ScriptProvider(
        [
            [
                ToolCallStart(id="b1", name="big", args='{"text": "y"}'),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [TextDelta(text="ok"), Done(usage=TokenUsage())],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="z")])])
    events = []

    async def emit(ev):
        events.append(ev)

    await run_agent_loop(provider=p, turn=turn, history=[], executor=ex, emit=emit)
    tr = next(e for e in events if isinstance(e, ToolResultEvent))
    assert tr.truncated is True
    assert len(tr.output) > 0  # 完整截断后正文,非空
    assert "已截断" in tr.output


# ---- summary 摘要:不把正文塞 head(否则折叠态泄露、展开态重复) ----


def test_summarize_read_shows_filename_and_lines_not_content():
    from birdcode.blocks import ToolUseBlock

    tu = ToolUseBlock(id="r1", name="read_file", input={"file_path": "src/x.py"})
    body = "1\timport os\n2\tx = 1\n3\ty = 2\n"  # 带行号的正文(cat -n 式)
    s = _summarize_tool(tu, body, ok=True)
    assert "x.py" in s and "3 行" in s
    assert "import os" not in s  # 正文不泄露进 summary


def test_summarize_grep_shows_hit_count():
    from birdcode.blocks import ToolUseBlock

    tu = ToolUseBlock(id="g1", name="grep", input={"pattern": "x"})
    assert _summarize_tool(tu, "a\nb\nc\n", ok=True) == "3 处命中"


def test_summarize_bash_shows_command():
    from birdcode.blocks import ToolUseBlock

    tu = ToolUseBlock(id="b1", name="bash", input={"command": "ls -la"})
    assert _summarize_tool(tu, "drwxr...", ok=True) == "$ ls -la"


def test_summarize_error_shows_message_prefix():
    from birdcode.blocks import ToolUseBlock

    tu = ToolUseBlock(id="e1", name="read_file", input={"file_path": "x"})
    s = _summarize_tool(tu, "<error>文件不存在: x</error>", ok=False)
    assert "文件不存在" in s


def test_summarize_glob_shows_file_count():
    from birdcode.blocks import ToolUseBlock

    tu = ToolUseBlock(id="gl", name="glob", input={"pattern": "*.py"})
    assert _summarize_tool(tu, "a.py\nb.py\n", ok=True) == "2 个文件"
    assert _summarize_tool(tu, "", ok=True) == "无匹配"


def test_summarize_write_edit_shows_filename():
    from birdcode.blocks import ToolUseBlock

    tu_w = ToolUseBlock(
        id="w1", name="write_file", input={"file_path": "src/x.py", "content": "x"}
    )
    assert _summarize_tool(tu_w, "已写入", ok=True) == "x.py"
    tu_e = ToolUseBlock(id="e1", name="edit_file", input={"file_path": "README.md"})
    assert _summarize_tool(tu_e, "已编辑", ok=True) == "README.md"


def test_summarize_fallback_unknown_tool_uses_first_line():
    """未列举的工具(如 TodoWrite)兜底用 output 首行前 80 字。"""
    from birdcode.blocks import ToolUseBlock

    tu = ToolUseBlock(id="u1", name="TodoWrite", input={})
    assert _summarize_tool(tu, "created 3 todos\nmore lines", ok=True) == "created 3 todos"
    assert _summarize_tool(tu, "", ok=True) == "TodoWrite"  # 空输出兜底工具名


# ---- 中断正确性:ESC 期间不得留下悬空 tool_use(否则下轮 API 400) ----


@pytest.mark.asyncio
async def test_interrupt_synthesizes_tool_result_no_dangling_tool_use():
    """回归(中断正确性):ESC 落在 execute_batch 期间 → CancelledError 穿透 executor。
    agent_loop 必须为每个 tool_use 合成 error tool_result 回填,否则 assistant(tool_use)
    在 history 悬空、下一轮 API 报 400。断言:CancelledError 抛出 + 末尾是完整 user(tool_result)。"""
    import asyncio

    from pydantic import BaseModel

    from birdcode.blocks import ToolResultBlock, ToolUseBlock
    from birdcode.tools import Tool, ToolRegistry

    class _NoInput(BaseModel):
        pass

    class _SlowTool(Tool):
        name = "slow"
        description = "d"
        parameters = _NoInput
        kind = "read"
        parallel_safe = True

        async def execute(self, **args):  # type: ignore[override]
            await asyncio.sleep(10)
            return "done"

    reg = ToolRegistry()
    reg.register(_SlowTool())
    ex = ToolExecutor(reg)

    class _P:
        async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
            yield ToolCallStart(id="s1", name="slow", args="{}")
            yield Done(usage=TokenUsage(), stop_reason="tool_use")

    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])

    async def emit(ev):  # type: ignore[no-untyped-def]
        pass

    task = asyncio.create_task(
        run_agent_loop(provider=_P(), turn=turn, history=[], executor=ex, emit=emit)
    )
    await asyncio.sleep(0.2)  # 等进入 execute_batch 的 sleep(10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # assistant(tool_use) 已落,其后紧跟合成的 user(tool_result)——无悬空
    assert len(turn.messages) == 3
    assert turn.messages[1].role == "assistant"
    assert any(isinstance(b, ToolUseBlock) for b in turn.messages[1].content)
    last = turn.messages[2]
    assert last.role == "user"
    tr = next(b for b in last.content if isinstance(b, ToolResultBlock))
    assert tr.tool_use_id == "s1"
    assert tr.is_error is True
    assert "中断" in (tr.content or "")


@pytest.mark.asyncio
async def test_interrupt_keeps_real_result_for_already_executed_tool():
    """回归(中断语义):批次含「已完成的写工具 + 慢工具」,ESC 落在慢工具执行期。

    修复前 agent_loop 对全部 tool_use 回填 error → 已落盘的 write 真实结果丢失、
    模型下轮看到「调了 write_file」配「未执行」→ 重试写入造成重复。修复后已完成的工具
    保留真实 tool_result,只有未完成的标 error。
    """
    import asyncio

    from pydantic import BaseModel

    from birdcode.blocks import ToolResultBlock
    from birdcode.tools import Tool, ToolRegistry

    class _In(BaseModel):
        pass

    class _FastWrite(Tool):
        name = "fastwrite"
        description = "d"
        parameters = _In
        kind = "write"
        parallel_safe = False

        async def execute(self, **args):  # type: ignore[override]
            return "fast-output"

    class _SlowWrite(Tool):
        name = "slowwrite"
        description = "d"
        parameters = _In
        kind = "write"
        parallel_safe = False

        async def execute(self, **args):  # type: ignore[override]
            await asyncio.sleep(10)
            return "slow-output"

    reg = ToolRegistry()
    reg.register(_FastWrite())
    reg.register(_SlowWrite())
    ex = ToolExecutor(reg)

    class _P:
        async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
            yield ToolCallStart(id="f1", name="fastwrite", args="{}")
            yield ToolCallStart(id="s1", name="slowwrite", args="{}")
            yield Done(usage=TokenUsage(), stop_reason="tool_use")

    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])

    async def emit(ev):  # type: ignore[no-untyped-def]
        pass

    task = asyncio.create_task(
        run_agent_loop(provider=_P(), turn=turn, history=[], executor=ex, emit=emit)
    )
    await asyncio.sleep(0.3)  # 等 fastwrite 完成、进入 slowwrite 的 sleep(10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 末尾 user 消息含两个 tool_result:f1 真实(非 error)、s1 error(未执行)
    last = turn.messages[-1]
    assert last.role == "user"
    results = {b.tool_use_id: b for b in last.content if isinstance(b, ToolResultBlock)}
    assert results["f1"].is_error is False
    assert "fast-output" in (results["f1"].content or "")
    assert results["s1"].is_error is True
    assert "中断" in (results["s1"].content or "")


# ---- Done 时落 usage 日志(含缓存字段) ----


class _DoneOnlyProvider:
    async def stream(self, messages, *, history):
        yield Done(
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=500,
                cache_creation_tokens=20,
                cost_usd=0.01,
            )
        )


@pytest.mark.asyncio
async def test_done_usage_logged_with_cache_fields(caplog):
    emitted: list = []

    async def _emit(ev):
        emitted.append(ev)

    # get_logger 置 propagate=False(只写文件,见 utils/logging.py:22);
    # caplog 依赖向 root 传播,这里临时放开以捕获记录。
    logging.getLogger("birdcode.agent.cache").propagate = True
    caplog.set_level(logging.INFO, logger="birdcode.agent.cache")
    turn = Turn(messages=[])
    await run_agent_loop(
        provider=_DoneOnlyProvider(), turn=turn, history=[], executor=None, emit=_emit
    )
    assert any(
        "[cache]" in r.message and "read=500" in r.message and "create=20" in r.message
        for r in caplog.records
    )


# ---- 中断语义:被中断工具不得声称"未执行"(bash 非幂等,致盲目重试)----


@pytest.mark.asyncio
async def test_interrupted_tool_warns_partial_not_claims_unexecuted():
    """回归(中断语义):被中断(非已完成)的工具,result 不得声称'未执行'。

    bash 等命令非幂等,ESC 杀进程组杀在中间,已改/删的文件不会回滚。若 tool_result 标
    '未执行',模型下轮盲目重试整个命令 → rm/mv/>>/sed 等造成重复损害。须改为'可能已
    部分完成,请核查',让模型先 read_file/ls/git status 重建认知再补救。
    """
    import asyncio

    from pydantic import BaseModel

    from birdcode.blocks import ToolResultBlock
    from birdcode.tools import Tool, ToolRegistry

    class _In(BaseModel):
        pass

    class _SlowWrite(Tool):
        name = "slowwrite"
        description = "d"
        parameters = _In
        kind = "write"
        parallel_safe = False

        async def execute(self, **args):  # type: ignore[override]
            await asyncio.sleep(10)
            return "done"

    reg = ToolRegistry()
    reg.register(_SlowWrite())
    ex = ToolExecutor(reg)

    class _P:
        async def stream(self, messages, *, history):  # type: ignore[no-untyped-def]
            yield ToolCallStart(id="s1", name="slowwrite", args="{}")
            yield Done(usage=TokenUsage(), stop_reason="tool_use")

    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])

    async def emit(ev):  # type: ignore[no-untyped-def]
        pass

    task = asyncio.create_task(
        run_agent_loop(provider=_P(), turn=turn, history=[], executor=ex, emit=emit)
    )
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    tr = next(b for b in turn.messages[-1].content if isinstance(b, ToolResultBlock))
    assert tr.is_error is True
    content = tr.content or ""
    assert "未执行" not in content, "被中断工具不得声称'未执行'(bash 非幂等,致盲目重试)"
    assert "部分完成" in content or "核查" in content


# ---- Task 6: 双轨外存 — ToolResultBlock.content 用 llm_content + on_message 回调 ----


@pytest.mark.asyncio
async def test_tool_result_block_uses_llm_content_not_output():
    """双轨外存:给 LLM 的 ToolResultBlock.content 用 r.llm_content(占位),不是 r.output(UI 截断)。

    executor 产出 llm_content != output 时(超阈落盘),agent_loop 构造 block 用 llm_content。
    """
    from birdcode.blocks import ToolResultBlock
    from birdcode.tools.base import ToolResult as InternalResult

    class _FakeExec:
        """绕过真实 executor,直接返回 llm_content != output 的结果。"""

        async def execute_batch(self, tool_uses):  # type: ignore[no-untyped-def]
            return [
                InternalResult(
                    tool_use_id=tu.id,
                    ok=True,
                    output="UI截断版",
                    llm_content="<persisted-output>给LLM的占位",
                    truncated=True,
                )
                for tu in tool_uses
            ]

        def last_partial(self):
            return {}

    async def _noop_emit(ev):  # noqa: ARG001 - agent_loop 需要 awaitable emit
        return None

    p = _ScriptProvider(
        [
            [
                ToolCallStart(id="t1", name="echo", args='{"text":"x"}'),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [TextDelta(text="ok"), Done(usage=TokenUsage())],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="z")])])
    await run_agent_loop(provider=p, turn=turn, history=[], executor=_FakeExec(), emit=_noop_emit)  # type: ignore[arg-type]
    tr = next(b for b in turn.messages[2].content if isinstance(b, ToolResultBlock))
    assert tr.content == "<persisted-output>给LLM的占位"  # 不是 "UI截断版"


@pytest.mark.asyncio
async def test_on_message_called_for_each_appended_message():
    """on_message 回调:assistant 与 tool_result user 行追加时各调一次,带 usage/stop_reason。"""
    captured: list[tuple[str, object, object]] = []

    async def on_message(msg, *, usage=None, stop_reason=None):  # type: ignore[no-untyped-def]
        captured.append((msg.role, usage, stop_reason))

    async def _noop_emit(ev):  # noqa: ARG001
        return None

    p = _ScriptProvider(
        [
            [
                ToolCallStart(id="t1", name="echo", args='{"text":"x"}'),
                Done(usage=TokenUsage(input_tokens=4), stop_reason="tool_use"),
            ],
            [
                TextDelta(text="done"),
                Done(usage=TokenUsage(input_tokens=9), stop_reason="end_turn"),
            ],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="z")])])
    await run_agent_loop(
        provider=p,
        turn=turn,
        history=[],
        executor=ToolExecutor(default_registry()),
        emit=_noop_emit,
        on_message=on_message,
    )
    roles = [c[0] for c in captured]
    # assistant(tool_use) → user(tool_result) → assistant(text)
    assert roles == ["assistant", "user", "assistant"]
    # 第一个 assistant 带 stop_reason="tool_use";最后 assistant 带 usage(input=9) + end_turn
    assert captured[0][2] == "tool_use"
    assert captured[2][1] is not None and captured[2][1].input_tokens == 9
    assert captured[2][2] == "end_turn"


# ---- 护栏:轮次上限 + 连续失败熔断(防失控 / 死循环)----


@pytest.mark.asyncio
async def test_max_tool_rounds_stops_loop():
    """轮次硬上限:模型反复调工具(每次成功)不停 → 到 max_tool_rounds 强制停 + emit Error。"""
    p = _ScriptProvider(
        [
            [
                ToolCallStart(id=f"t{i}", name="echo", args='{"text":"x"}'),
                Done(usage=None, stop_reason="tool_use"),
            ]
            for i in range(3)
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])
    events: list = []

    async def emit(ev):  # type: ignore[no-untyped-def]
        events.append(ev)

    await run_agent_loop(
        provider=p, turn=turn, history=[], executor=ToolExecutor(default_registry()),
        emit=emit, max_tool_rounds=3,
    )
    assert p.calls == 3  # 第 3 轮回填后 rounds==3 → 停,不再第 4 次 stream
    errs = [e for e in events if isinstance(e, Error)]
    assert errs and "最大工具轮次" in errs[-1].message


@pytest.mark.asyncio
async def test_consecutive_same_failure_circuit_breaks():
    """连续熔断:同一工具 + 相同参数连续失败 N 次 → 停(防死循环重试)。"""
    from pydantic import BaseModel

    from birdcode.tools import Tool, ToolRegistry

    class _In(BaseModel):
        pass

    class _FailTool(Tool):
        name = "fail"
        description = "d"
        parameters = _In
        kind = "read"
        parallel_safe = True

        async def execute(self, **args):  # type: ignore[override]
            raise RuntimeError("总是失败")

    reg = ToolRegistry()
    reg.register(_FailTool())
    ex = ToolExecutor(reg)

    p = _ScriptProvider(
        [
            [
                ToolCallStart(id=f"f{i}", name="fail", args="{}"),
                Done(usage=None, stop_reason="tool_use"),
            ]
            for i in range(3)
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])
    events: list = []

    async def emit(ev):  # type: ignore[no-untyped-def]
        events.append(ev)

    await run_agent_loop(
        provider=p, turn=turn, history=[], executor=ex, emit=emit, max_same_failures=3,
    )
    assert p.calls == 3  # 第 3 次失败后 streak==3 → 熔断,不再第 4 次 stream
    errs = [e for e in events if isinstance(e, Error)]
    assert errs and "连续" in errs[-1].message and "fail" in errs[-1].message


@pytest.mark.asyncio
async def test_success_resets_failure_streak():
    """成功打破连续失败计数:fail → success → fail 不熔断(中间成功重置 streak)。"""
    from pydantic import BaseModel

    from birdcode.tools import Tool

    class _In(BaseModel):
        pass

    class _FailTool(Tool):
        name = "fail"
        description = "d"
        parameters = _In
        kind = "read"
        parallel_safe = True

        async def execute(self, **args):  # type: ignore[override]
            raise RuntimeError("失败")

    reg = default_registry()
    reg.register(_FailTool())
    ex = ToolExecutor(reg)

    p = _ScriptProvider(
        [
            [
                ToolCallStart(id="f1", name="fail", args="{}"),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [
                ToolCallStart(id="e1", name="echo", args='{"text":"x"}'),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [
                ToolCallStart(id="f2", name="fail", args="{}"),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [TextDelta(text="ok"), Done(usage=TokenUsage(), stop_reason="end_turn")],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])
    events: list = []

    async def emit(ev):  # type: ignore[no-untyped-def]
        events.append(ev)

    await run_agent_loop(provider=p, turn=turn, history=[], executor=ex, emit=emit)
    assert p.calls == 4  # 4 轮全跑完(最后一轮纯文本收尾)
    assert not any(isinstance(e, Error) for e in events)  # 未熔断


@pytest.mark.asyncio
async def test_different_failure_signature_resets_streak():
    """签名变化重置:fail(x=1) → fail(x=2) 参数不同 → 各只 streak=1,不熔断。"""
    from pydantic import BaseModel

    from birdcode.tools import Tool, ToolRegistry

    class _In(BaseModel):
        x: int = 0

    class _FailTool(Tool):
        name = "fail"
        description = "d"
        parameters = _In
        kind = "read"
        parallel_safe = True

        async def execute(self, **args):  # type: ignore[override]
            raise RuntimeError("失败")

    reg = ToolRegistry()
    reg.register(_FailTool())
    ex = ToolExecutor(reg)

    p = _ScriptProvider(
        [
            [
                ToolCallStart(id="f1", name="fail", args='{"x":1}'),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [
                ToolCallStart(id="f2", name="fail", args='{"x":2}'),
                Done(usage=None, stop_reason="tool_use"),
            ],
            [TextDelta(text="ok"), Done(usage=TokenUsage(), stop_reason="end_turn")],
        ]
    )
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="x")])])
    events: list = []

    async def emit(ev):  # type: ignore[no-untyped-def]
        events.append(ev)

    await run_agent_loop(provider=p, turn=turn, history=[], executor=ex, emit=emit)
    assert p.calls == 3  # 3 轮(fail→fail→text 收尾)
    assert not any(isinstance(e, Error) for e in events)  # 参数变 → streak 不累加 → 未熔断
