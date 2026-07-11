import pytest
from pydantic import BaseModel

from birdcode.blocks import ToolUseBlock
from birdcode.mcp.errors import ToolError
from birdcode.tools import Tool, ToolRegistry, ToolResult
from birdcode.tools.echo import EchoTool
from birdcode.tools.executor import ToolExecutor, default_registry


class _DummyTool(Tool):
    name = "dummy"
    description = "d"
    kind = "read"
    parallel_safe = True

    async def execute(self, **args: object) -> str:
        return "ok"


def test_registry_register_and_get():
    r = ToolRegistry()
    r.register(_DummyTool())
    assert r.get("dummy") is not None
    assert r.get("nope") is None
    assert r.names() == ["dummy"]


def test_registry_rejects_empty_name():
    import pytest

    class _NoName(Tool):
        name = ""

        async def execute(self, **args: object) -> str:
            return ""

    r = ToolRegistry()
    with pytest.raises(ValueError):
        r.register(_NoName())


def test_toolresult_defaults():
    tr = ToolResult(tool_use_id="t1", ok=True)
    assert tr.output == ""
    assert tr.error_message == ""
    assert tr.truncated is False


# ---- Task 7: ToolExecutor ----


class _EchoInput(BaseModel):
    text: str


class _EchoTool(Tool):
    name = "echo"
    description = "echo"
    parameters = _EchoInput
    kind = "read"
    parallel_safe = True
    max_result_chars = 100_000  # 默认阈值(测「不超阈」路径)

    async def execute(self, **args):
        return f"echo:{args['text']}"


class _LowEchoTool(_EchoTool):
    """低阈值 echo(max_result_chars=100):测「超阈落盘」路径(per-tool 阈值验证)。"""

    name = "lowecho"
    max_result_chars = 100


def _reg():
    r = ToolRegistry()
    r.register(_EchoTool())
    r.register(_LowEchoTool())
    return r


@pytest.mark.asyncio
async def test_execute_batch_runs_parallel_safe_tools():
    ex = ToolExecutor(_reg())
    tus = [
        ToolUseBlock(id="a", name="echo", input={"text": "1"}),
        ToolUseBlock(id="b", name="echo", input={"text": "2"}),
    ]
    res = await ex.execute_batch(tus)
    by_id = {r.tool_use_id: r for r in res}
    assert by_id["a"].ok and "echo:1" in by_id["a"].output
    assert by_id["b"].ok and "echo:2" in by_id["b"].output


@pytest.mark.asyncio
async def test_unknown_tool_yields_error_result():
    ex = ToolExecutor(_reg())
    res = await ex.execute_batch([ToolUseBlock(id="x", name="ghost", input={})])
    assert res[0].ok is False
    assert "ghost" in res[0].error_message


@pytest.mark.asyncio
async def test_validation_failure_yields_error_result_not_exception():
    class _Strict(BaseModel):
        text: str

    class _StrictTool(Tool):
        name = "strict"
        parameters = _Strict
        kind = "read"
        parallel_safe = True

        async def execute(self, **args):
            return "should not reach"

    r = ToolRegistry()
    r.register(_StrictTool())
    ex = ToolExecutor(r)
    res = await ex.execute_batch([ToolUseBlock(id="s", name="strict", input={})])  # 缺 text
    assert res[0].ok is False and res[0].error_message  # 校验错误回传，未抛异常


@pytest.mark.asyncio
async def test_truncates_long_output():
    ex = ToolExecutor(_reg(), char_limit=10)
    res = await ex.execute_batch([ToolUseBlock(id="a", name="echo", input={"text": "x" * 1000})])
    assert res[0].truncated is True
    assert len(res[0].output) < 1000
    assert "已截断" in res[0].output


# ---- Task 8: default_registry + EchoTool stub ----


def test_default_registry_helper():
    from birdcode.tools.executor import default_registry

    r = default_registry()
    assert r.get("echo") is not None


@pytest.mark.asyncio
async def test_echo_tool_returns_input_text():
    assert await EchoTool().execute(text="hi") == "hi"


# ---- stage2: ToolRegistry schema 导出 ----


class _SchemaInput(BaseModel):
    file_path: str
    limit: int = 2000


class _SchemaTool(Tool):
    name = "read"
    description = "读文件。参数:file_path(必填)、limit(默认 2000)。输出:带行号文本。"
    parameters = _SchemaInput
    kind = "read"
    parallel_safe = True

    async def execute(self, **args: object) -> str:
        return ""


def test_to_anthropic_tools_shape():
    r = ToolRegistry()
    r.register(_SchemaTool())
    tools = r.to_anthropic_tools()
    assert len(tools) == 1
    t = tools[0]
    assert t["name"] == "read"
    assert "file_path" in t["description"]
    schema = t["input_schema"]
    assert schema["type"] == "object"
    assert "file_path" in schema["properties"]
    assert schema["required"] == ["file_path"]
    # 默认值经 model_json_schema 落入 properties
    assert schema["properties"]["limit"]["default"] == 2000


def test_to_openai_tools_shape():
    r = ToolRegistry()
    r.register(_SchemaTool())
    tools = r.to_openai_tools()
    assert len(tools) == 1
    t = tools[0]
    assert t["type"] == "function"
    fn = t["function"]
    assert fn["name"] == "read"
    assert "file_path" in fn["description"]
    schema = fn["parameters"]
    assert schema["type"] == "object"
    assert schema["required"] == ["file_path"]
    assert schema["properties"]["limit"]["default"] == 2000


def test_schema_export_empty_registry():
    assert ToolRegistry().to_anthropic_tools() == []
    assert ToolRegistry().to_openai_tools() == []


# ---- stage2 polish(§7): tools/__init__ 导出 ----


def test_tools_init_exports_executor_and_default_registry():
    import birdcode.tools as pkg

    assert hasattr(pkg, "ToolExecutor")
    assert hasattr(pkg, "default_registry")
    assert "ToolExecutor" in pkg.__all__
    assert "default_registry" in pkg.__all__
    # default_registry() 可调
    assert pkg.default_registry().get("echo") is not None


# ---- stage3: default_registry 注册只读工具 + 并行执行 ----


def test_default_registry_has_readonly_tools():
    from birdcode.tools.executor import default_registry

    r = default_registry()
    # 三个新工具(Glob/Grep/Read)+ stage1 stub(echo)
    for name in ("glob", "grep", "read_file", "echo"):
        assert r.get(name) is not None, f"{name} 未注册"


@pytest.mark.asyncio
async def test_readonly_tools_run_parallel_via_executor(tmp_path):
    """三个只读工具均 parallel_safe=True → executor 走 asyncio.gather 并行。"""
    (tmp_path / "x.py").write_text("hello\n", encoding="utf-8")
    ex = ToolExecutor(default_registry())
    tus = [
        ToolUseBlock(id="g", name="glob", input={"pattern": "*.py", "path": str(tmp_path)}),
        ToolUseBlock(id="r", name="read_file", input={"file_path": str(tmp_path / "x.py")}),
    ]
    res = await ex.execute_batch(tus)
    by_id = {r.tool_use_id: r for r in res}
    assert by_id["g"].ok and "x.py" in by_id["g"].output
    assert by_id["r"].ok and "hello" in by_id["r"].output


@pytest.mark.asyncio
async def test_execute_batch_preserves_input_order_for_mixed_batch():
    """回归(executor 高危):parallel(parallel_safe=True)与 serial(False)混合时,
    返回必须保持输入原序——agent_loop 用位置 zip 消费,错位会把结果装进错误的 tool_use_id。
    旧实现返回 [所有 parallel, 所有 serial],与输入原序不一致。"""

    class _SlowTool(Tool):
        name = "slow"
        description = "d"
        parameters = _EchoInput
        kind = "write"
        parallel_safe = False

        async def execute(self, **args: object) -> str:
            return f"slow:{args.get('text', '')}"

    r = _reg()
    r.register(_SlowTool())
    ex = ToolExecutor(r)
    tus = [
        ToolUseBlock(id="a", name="echo", input={"text": "A"}),  # parallel
        ToolUseBlock(id="b", name="slow", input={"text": "B"}),  # serial
        ToolUseBlock(id="c", name="echo", input={"text": "C"}),  # parallel
    ]
    res = await ex.execute_batch(tus)
    # 位置断言(非 by_id,后者会掩盖顺序错位):结果序必须等于输入序
    assert [x.tool_use_id for x in res] == ["a", "b", "c"]
    assert "echo:A" in res[0].output
    assert "slow:B" in res[1].output
    assert "echo:C" in res[2].output


# ---- 中断正确性:serial 分支不得吞 CancelledError ----


@pytest.mark.asyncio
async def test_serial_branch_propagates_cancelled_error():
    """回归(中断正确性):serial 分支曾用 except BaseException 误吞 CancelledError
    (写类工具 ESC 失效、turn 不停)。收窄为 except Exception 后,CancelledError 必须透传。"""
    import asyncio

    class _SlowInput(BaseModel):
        pass

    class _SlowWrite(Tool):
        name = "slowwrite"
        description = "d"
        parameters = _SlowInput
        kind = "write"
        parallel_safe = False

        async def execute(self, **args):  # type: ignore[override]
            await asyncio.sleep(10)
            return "done"

    r = ToolRegistry()
    r.register(_SlowWrite())
    ex = ToolExecutor(r)

    task = asyncio.create_task(ex.execute_batch([ToolUseBlock(id="s", name="slowwrite", input={})]))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---- 中断语义:cancel 时保留已完成的真实结果(避免已落盘写入被当"未执行")----


@pytest.mark.asyncio
async def test_execute_batch_exposes_partial_results_on_cancel():
    """回归(中断语义):串行批次中途被 cancel,已完成的工具结果不应随异常丢失。

    修复前 execute_batch 整体抛 CancelledError,by_id 丢失 → agent_loop 只能对全部
    tool_use 回填 error → 已落盘的 write/edit 被模型当"未执行",下轮重试造成重复写入。
    修复后 executor 在 re-raise 前把已完成的 by_id 存进 last_partial() 供上层取用。
    """
    import asyncio

    class _In(BaseModel):
        pass

    class _Fast(Tool):
        name = "fast"
        parameters = _In
        kind = "write"
        parallel_safe = False

        async def execute(self, **args):  # type: ignore[override]
            return "fast-done"

    class _Slow(Tool):
        name = "slow"
        parameters = _In
        kind = "write"
        parallel_safe = False

        async def execute(self, **args):  # type: ignore[override]
            await asyncio.sleep(10)
            return "slow-done"

    r = ToolRegistry()
    r.register(_Fast())
    r.register(_Slow())
    ex = ToolExecutor(r)

    task = asyncio.create_task(
        ex.execute_batch(
            [
                ToolUseBlock(id="f1", name="fast", input={}),
                ToolUseBlock(id="s1", name="slow", input={}),
            ]
        )
    )
    await asyncio.sleep(0.3)  # 等 fast 完成、进入 slow 的 sleep(10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    partial = ex.last_partial()
    # 已完成的 fast 真实结果保留;未完成的 slow 不在 partial
    assert "f1" in partial and partial["f1"].ok and "fast-done" in partial["f1"].output
    assert "s1" not in partial


# ---- 不变式:并行执行的必须是只读工具(防并行写工具静默丢部分写入)----


@pytest.mark.asyncio
async def test_parallel_safe_write_tool_rejected():
    """回归(中断语义边界):parallel_safe=True 的写工具,取消时 gather 不返回已完成结果,
    会静默丢失部分写入(见 last_partial 的串行限定)。executor 在并行分组处断言
    parallel_safe 工具必须是 read 类,让未来的错误配置尽早暴露而非静默丢数据。"""

    class _BadParallelWrite(Tool):
        name = "badwrite"
        description = "d"
        parameters = _EchoInput
        kind = "write"
        parallel_safe = True  # 矛盾配置:写工具不应并行

        async def execute(self, **args):  # type: ignore[override]
            return "x"

    r = ToolRegistry()
    r.register(_BadParallelWrite())
    ex = ToolExecutor(r)
    with pytest.raises(AssertionError):
        await ex.execute_batch([ToolUseBlock(id="b", name="badwrite", input={"text": "x"})])


# ---- Task 9: MCP 原始 schema 跳过 Pydantic + ToolError 转 ok=False ----


class _RawInputTool(Tool):
    name = "raw_in"
    description = "d"
    input_schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    received: dict = {}

    async def execute(self, **args):  # type: ignore[override]
        type(self).received = args
        return f"got={args}"


@pytest.mark.asyncio
async def test_raw_schema_tool_skips_pydantic_validation():
    """MCP 工具带 input_schema → 跳过 Pydantic 校验,原样转发(server 端自校验)。
    未声明字段 extra 也应原样透传到 execute,不被 Pydantic 默认剥离。"""
    reg = ToolRegistry()
    reg.register(_RawInputTool())
    ex = ToolExecutor(reg)
    tu = ToolUseBlock(id="1", name="raw_in", input={"extra": 1})
    res = await ex.execute_batch([tu])
    assert res[0].ok is True
    assert _RawInputTool.received == {"extra": 1}


class _BoomTool(Tool):
    name = "boom"
    description = "d"
    # 给真实 BaseModel;默认 BaseModel.model_validate 会抛 PydanticUserError(非 ValidationError)
    parameters = _EchoInput

    async def execute(self, **args):  # type: ignore[override]
        raise ToolError("<error>boom</error>")


@pytest.mark.asyncio
async def test_tool_error_becomes_ok_false_not_exception():
    """工具执行期抛 ToolError(McpServerDownError 等子类)→ executor 转干净 ok=False,
    绝不让 Traceback 冒泡到 agent_loop。"""
    reg = ToolRegistry()
    reg.register(_BoomTool())
    ex = ToolExecutor(reg)
    res = await ex.execute_batch([ToolUseBlock(id="2", name="boom", input={"text": "x"})])
    assert res[0].ok is False
    assert "boom" in res[0].error_message


@pytest.mark.asyncio
async def test_raw_schema_tool_non_dict_input_does_not_crash():
    """回归(#5):模型给出非对象 args(json "42"/list/null)时,MCP 分支 dict(input) 抛
    TypeError 绕过 ValidationError 通路。收口为非 dict→{} 干净转发,绝不抛异常冒泡。"""
    reg = ToolRegistry()
    reg.register(_RawInputTool())
    ex = ToolExecutor(reg)
    tu = ToolUseBlock(id="1", name="raw_in", input=["not", "dict"])  # type: ignore[arg-type]
    res = await ex.execute_batch([tu])
    assert res[0].ok is True  # 非崩;非 dict 收口为 {} 转发,execute 正常返回
    assert _RawInputTool.received == {}


# ---- Task 5: executor 三轨外存(UI 2000 / LLM 占位 / 本地落盘)----


class _RecordingSink:
    """记录 persist 调用,模拟 SessionStore.as_output_sink(dumb sink:阈值在 executor/tool 层)。

    鸭子类型匹配 OutputSink Protocol。被调即返回 PersistInfo(executor 据 tool.max_result_chars
    决定是否调);不再自带阈值——与 store.persist_tool_output 的 dumb-sink 契约一致。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def persist(self, tool_use_id: str, full_output: str):  # type: ignore[no-untyped-def]
        self.calls.append((tool_use_id, full_output))
        return _make_info(tool_use_id, full_output)


def _make_info(tool_use_id: str, full_output: str):
    from birdcode.session.models import PersistInfo

    return PersistInfo(
        llm_content=f"<persisted-output>\nsaved {tool_use_id}\n",
        path=f"/p/{tool_use_id}.txt",
        size=len(full_output),
    )


@pytest.mark.asyncio
async def test_read_file_never_persists(tmp_path):
    """read_file max_result_chars=inf → 永不落盘(避免「读落盘文件→又超阈→循环」)。

    即便输出远超其他工具阈值(bash 30K/grep 20K),read_file 也直接给 LLM 全量
    (由 read_file 自身 limit=500 行 + 500KB 护栏界定输出)。仿 CC FileReadTool=Infinity。
    """
    from birdcode.tools.read_tool import ReadTool
    from birdcode.tools.registry import ToolRegistry

    f = tmp_path / "big.txt"
    # 40K:超 bash(30K)/grep(20K) 阈值,但低于 read 的 _MAX_INLINE_CHARS(50K)→ 不截断、不落盘
    f.write_text("y" * 40000, encoding="utf-8")
    reg = ToolRegistry()
    reg.register(ReadTool())
    sink = _RecordingSink()
    ex = ToolExecutor(reg, output_sink=sink)
    res = await ex.execute_batch(
        [ToolUseBlock(id="r1", name="read_file", input={"file_path": str(f)})]
    )
    assert res[0].persisted_path is None  # 永不落盘(inf)
    assert sink.calls == []  # sink 根本没被调
    assert "<persisted-output>" not in res[0].llm_content  # inline 全文(未截断),非占位


@pytest.mark.asyncio
async def test_per_tool_threshold_low_tool_persists_early(tmp_path):
    """低阈值工具(max_result_chars=100)在 1000 字符就落盘;默认阈值工具(100K)不落盘。

    证明阈值是 per-tool、不是全局。
    """
    reg = _reg()  # 含 echo(默认 100K) + lowecho(100)
    sink = _RecordingSink()
    ex = ToolExecutor(reg, output_sink=sink)
    big = "x" * 1000
    await ex.execute_batch([ToolUseBlock(id="lo", name="lowecho", input={"text": big})])
    await ex.execute_batch([ToolUseBlock(id="hi", name="echo", input={"text": big})])
    assert any(c[0] == "lo" for c in sink.calls)  # lowecho(100) 落盘
    assert not any(c[0] == "hi" for c in sink.calls)  # echo(100K) 不落盘


@pytest.mark.asyncio
async def test_truncate_uses_2000_ui_limit_by_default():
    """UI 截断阈值默认 2000(原 20000)。"""
    ex = ToolExecutor(_reg())  # 默认 char_limit
    res = await ex.execute_batch([ToolUseBlock(id="a", name="echo", input={"text": "x" * 3000})])
    assert res[0].truncated is True
    assert len(res[0].output) < 3000


@pytest.mark.asyncio
async def test_no_sink_small_output_stays_inline():
    """有 sink 但输出低于 tool.max_result_chars(echo=100K)→ 不落盘,llm_content=原文。"""
    sink = _RecordingSink()
    ex = ToolExecutor(_reg(), output_sink=sink)
    res = await ex.execute_batch([ToolUseBlock(id="a", name="echo", input={"text": "small"})])
    assert res[0].llm_content == "echo:small"
    assert res[0].full_output == ""
    assert res[0].persisted_path is None
    assert sink.calls == []


@pytest.mark.asyncio
async def test_over_threshold_persists_and_llm_content_is_placeholder():
    """超 tool.max_result_chars(lowecho=100) → 落盘 + sink.persist 被调、llm_content=占位。"""
    sink = _RecordingSink()
    ex = ToolExecutor(_reg(), output_sink=sink, char_limit=50)
    big = "x" * 1000
    res = await ex.execute_batch([ToolUseBlock(id="a", name="lowecho", input={"text": big})])
    assert res[0].full_output == f"echo:{big}"  # 完整原文
    assert sink.calls and sink.calls[0][0] == "a"
    assert "saved" in res[0].llm_content  # 占位文本
    assert res[0].persisted_path == "/p/a.txt"
    assert res[0].persisted_size == len(f"echo:{big}")
    assert res[0].truncated is True  # UI 也截断(char_limit=50)


@pytest.mark.asyncio
async def test_output_still_truncated_for_ui_when_persisted():
    """超阈落盘后:UI 的 output 仍走 head/tail 截断(UI 阈值 2000);三轨分离。"""
    sink = _RecordingSink()
    ex = ToolExecutor(_reg(), output_sink=sink, char_limit=50)
    res = await ex.execute_batch([ToolUseBlock(id="a", name="lowecho", input={"text": "x" * 1000})])
    assert res[0].truncated is True
    assert "已截断" in res[0].output  # UI 截断
    llm = res[0].llm_content
    assert "<persisted-output>" in llm or "saved" in llm  # 给 LLM 占位


@pytest.mark.asyncio
async def test_no_sink_degrades_to_truncate_only():
    """无 sink(旧路径/测试)→ UI output 走 char_limit 截断(默认 2000)、
    llm_content 走 _DEFAULT_LLM_CHAR_LIMIT=20000 截断(F11 恢复旧行为),full_output 空。

    F11 修复前 llm_content=output(UI 截断,2000 量级);修复后 LLM 拿 20000 量级,
    与 UI 解耦。用 char_limit=100 + 1000 字符 input 验证:UI 走 100、LLM 仍走 20000(完整保留)。
    """
    ex = ToolExecutor(_reg(), char_limit=100)
    res = await ex.execute_batch([ToolUseBlock(id="a", name="echo", input={"text": "x" * 1000})])
    # UI output 走 char_limit=100 → 截断(head/tail 各 20)
    assert res[0].truncated is True
    assert len(res[0].output) < 200
    # F11:llm_content 走 _DEFAULT_LLM_CHAR_LIMIT=20000 → 不截断(1005 < 20000),完整保留
    assert res[0].llm_content == f"echo:{'x' * 1000}"
    assert res[0].llm_content != res[0].output  # 三轨分离:LLM 不再跟 UI 跌到 100
    assert res[0].full_output == ""
    assert res[0].persisted_path is None


@pytest.mark.asyncio
async def test_persist_failure_degrades_to_truncated_inline():
    """sink.persist 返回 None(落盘失败)→ llm_content 退回 LLM 截断版(20000),不阻塞。"""

    class _FailSink:
        async def persist(self, tool_use_id: str, full_output: str):  # type: ignore[no-untyped-def]
            return None

    ex = ToolExecutor(_reg(), output_sink=_FailSink(), char_limit=50)
    res = await ex.execute_batch([ToolUseBlock(id="a", name="lowecho", input={"text": "x" * 1000})])
    # UI output 仍走 char_limit=50
    assert len(res[0].output) < 100
    # F11:llm_content 走 20000(完整保留,不跟 UI 跌到 50)
    assert res[0].llm_content == f"echo:{'x' * 1000}"
    assert res[0].persisted_path is None


# ---- F11: 无 sink/落盘失败时 LLM 截断退化为 2000 → 应恢复 20000 ----


@pytest.mark.asyncio
async def test_no_sink_llm_content_uses_20000_limit_not_2000():
    """F11 回归守护:无 sink + 5000 字符输出 → llm_content 截到 20000 量级(完整保留),
    不是退化到 UI 的 2000。UI 的 output 仍截到 2000(三轨分离)。"""
    ex = ToolExecutor(_reg())  # 默认 char_limit=2000(给 UI 用),无 sink
    # 5000 字符:超过 UI 2000、低于 LLM 20000 → 验证 LLM 看到完整、UI 仍截断
    res = await ex.execute_batch([ToolUseBlock(id="a", name="echo", input={"text": "x" * 5000})])
    assert res[0].ok is True
    # UI output 仍截断到 2000 量级(三轨分离不变)
    assert res[0].truncated is True
    assert len(res[0].output) < 2500  # 2000 阈值 + head/tail 拼接少量 overhead
    # llm_content 必须保留完整(5000 字符原文),不是被截到 2000
    assert len(res[0].llm_content) >= 5000
    assert res[0].llm_content == f"echo:{'x' * 5000}"


@pytest.mark.asyncio
async def test_no_sink_huge_output_llm_truncates_at_20000():
    """F11:无 sink + 巨量输出(>20000)→ llm_content 截到 20000 阈值(head/tail 各
    cap/5=4000,~8000+label),不是 2000(UI cap 的 head/tail 才 ~800)。"""
    ex = ToolExecutor(_reg())  # 默认 char_limit=2000,无 sink
    res = await ex.execute_batch([ToolUseBlock(id="a", name="echo", input={"text": "x" * 50000})])
    assert res[0].ok is True
    # llm_content:cap=20000,head/tail 各 4000 → ~8000+label(远大于 UI 的 ~800)
    assert 7000 < len(res[0].llm_content) < 12000
    # UI output 仍截到 2000 量级(cap=2000,head/tail 各 400 → ~800+label)
    assert len(res[0].output) < 1500


@pytest.mark.asyncio
async def test_sink_below_threshold_llm_gets_full_output():
    """F11:有 sink 但不超阈 → llm_content = 完整原文(不截断、不落盘)。
    设计决策:不超阈没落盘,LLM 必须看全量(read_file 取不到落盘文件)。"""
    sink = _RecordingSink()
    ex = ToolExecutor(_reg(), output_sink=sink)
    # 5000 字符:超过 UI 2000、低于 LLM 20000、低于 echo 的 max_result_chars(100K)
    res = await ex.execute_batch([ToolUseBlock(id="a", name="echo", input={"text": "x" * 5000})])
    assert res[0].ok is True
    assert res[0].llm_content == f"echo:{'x' * 5000}"  # 完整原文
    assert res[0].full_output == ""  # 没落盘
    assert res[0].persisted_path is None
    assert sink.calls == []  # sink.persist 没被调


# ---- F5: sink.persist 抛非 OSError 异常不该让成功工具翻 ok=False ----


@pytest.mark.asyncio
async def test_sink_persist_raising_runtime_error_does_not_fail_tool():
    """F5 回归守护:sink.persist 抛 RuntimeError(逃过 ToolError except)→ ToolResult
    仍 ok=True、llm_content 降级为 LLM 截断版(20000,不是占位/不是 UI 2000)。"""

    class _BoomSink:
        async def persist(self, tool_use_id: str, full_output: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("sink boom")

    ex = ToolExecutor(_reg(), output_sink=_BoomSink(), char_limit=50)
    res = await ex.execute_batch([ToolUseBlock(id="a", name="lowecho", input={"text": "x" * 1000})])
    assert res[0].ok is True  # 不被 sink 异常连累
    assert res[0].persisted_path is None
    assert res[0].full_output == ""  # 落盘失败 → 没存
    # llm_content 降级为 LLM 截断版(_DEFAULT_LLM_CHAR_LIMIT=20000),不是 UI 50、不是占位
    assert "persisted-output" not in res[0].llm_content
    assert len(res[0].llm_content) > 50  # 远大于 UI char_limit=50


# ---- F1: ToolExecutor.update_output_sink ----


def test_executor_update_output_sink_replaces_sink():
    """F1: update_output_sink 在 /clear 后替换 _output_sink(指向新 session 的 sink)。"""

    class _Stub:
        async def persist(self, tool_use_id: str, full_output: str):  # type: ignore[no-untyped-def]
            return None

    ex = ToolExecutor(_reg())
    assert ex._output_sink is None  # noqa: SLF001
    new_sink = _Stub()
    ex.update_output_sink(new_sink)
    assert ex._output_sink is new_sink  # noqa: SLF001
    # 接受 None(无 store 测试路径)
    ex.update_output_sink(None)
    assert ex._output_sink is None  # noqa: SLF001


# ---- #4: UiPermissionGate.replace_extra_root ----


def test_gate_replace_extra_root_swaps_and_handles_absent_old(tmp_path):
    """#4:replace_extra_root 移旧加新(根数恒定);old 不在 roots 时优雅追加(降级路径)。

    覆盖 happy path(old 在)与降级 path(old 不在——如 prior /clear 抛异常后 gate 仍持
    原 dir)。后者不抛、追加 new,避免异常路径再次累积。
    """
    from birdcode.permission.gate import UiPermissionGate

    async def _get_mode():
        return "default"

    async def _req(name, summary, path):  # noqa: ANN001
        return "approve"

    old = tmp_path / "old-tr"
    new = tmp_path / "new-tr"
    gate = UiPermissionGate(get_mode=_get_mode, request_permission=_req, extra_roots=[old])
    assert old.resolve() in gate.extra_roots

    # old 在 roots → 移旧加新,根数不变
    gate.replace_extra_root(old, new)
    roots = gate.extra_roots
    assert new.resolve() in roots
    assert old.resolve() not in roots

    # old 不在 roots(降级路径)→ 不抛,追加 new
    other = tmp_path / "other-tr"
    gate.replace_extra_root(tmp_path / "absent-tr", other)
    assert other.resolve() in gate.extra_roots
