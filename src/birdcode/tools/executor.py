# src/birdcode/tools/executor.py
"""工具执行器:Pydantic 校验 + 输出截断 + 多工具混合执行(只读并行/写入串行)。

校验失败不抛异常——转为 ToolResult(ok=False) 回传,让模型自纠(subagent.md 要求)。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from birdcode.blocks import ToolUseBlock
from birdcode.mcp.errors import ToolError
from birdcode.tools.base import ToolOutput, ToolResult
from birdcode.tools.registry import ToolRegistry
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    from birdcode.permission.gate import PermissionGate
    from birdcode.permission.rules import Rule

log = get_logger("birdcode.tools.executor")  # 进 debug.log

_DEFAULT_CHAR_LIMIT = 2000  # UI 折叠区截断阈值(原 20000)
# 无 sink / 落盘失败时给 LLM 的截断阈值,恢复旧行为(20000)。三轨分离后,
# 不应让 LLM 跟 UI 跌到 2000(读小输出本可全看,不该被 UI 阈值连累)。
_DEFAULT_LLM_CHAR_LIMIT = 20000


class PersistInfo(Protocol):
    """executor 落盘返回的元信息协议(session.models.PersistInfo 鸭子类型匹配)。"""

    llm_content: str
    path: str
    size: int


class OutputSink(Protocol):
    """executor 落盘外存的协议(SessionStore.as_output_sink 实现)。

    用 Protocol 而非直接 import birdcode.session,避免 tools → session 反向依赖
    (session 已依赖 conversation/blocks)。结构匹配即可。
    """

    async def persist(self, tool_use_id: str, full_output: str) -> PersistInfo | None: ...


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        char_limit: int = _DEFAULT_CHAR_LIMIT,
        permission_gate: PermissionGate | None = None,
        output_sink: OutputSink | None = None,
    ) -> None:
        self._registry = registry
        self._char_limit = char_limit
        self._permission = permission_gate
        self._output_sink = output_sink
        self._last_partial: dict[str, ToolResult] = {}  # cancel 时已完成的真实结果(见 last_partial)

    async def execute_batch(self, tool_uses: list[ToolUseBlock]) -> list[ToolResult]:
        if not tool_uses:
            return []
        self._last_partial = {}  # 每次重置:反映本次部分结果(不被上次残留污染)
        parallel: list[ToolUseBlock] = []
        serial: list[ToolUseBlock] = []
        for tu in tool_uses:
            tool = self._registry.get(tu.name)
            if tool is not None and tool.parallel_safe:
                # 不变式:并行执行的必须是只读工具。否则取消时 asyncio.gather 不返回已完成
                # 结果(见 last_partial 的串行限定),会静默丢失部分写入。写工具一律串行。
                assert tool.kind != "write", (
                    f"工具 {tu.name} 同时是 write 类与 parallel_safe=True —— "
                    "并行写工具在取消时会静默丢失部分写入,请改为 parallel_safe=False"
                )
                parallel.append(tu)
            else:
                serial.append(tu)

        # 按 tu.id 收集,再按 tool_uses 原序返回——保证 results[i] 对应 tool_uses[i]。
        # (parallel/serial 分组执行,但回传必须保输入序,否则 agent_loop 的位置 zip 会错位:
        #  混合批次时 parallel 结果在前、serial 在后,会与 tool_uses 原序错位。)
        by_id: dict[str, ToolResult] = {}
        try:
            if parallel:
                raw = await asyncio.gather(
                    *(self._exec_one(tu) for tu in parallel), return_exceptions=True
                )
                for tu, r in zip(parallel, raw, strict=True):
                    by_id[tu.id] = r if isinstance(r, ToolResult) else self._exception_result(tu, r)
            for tu in serial:
                try:
                    by_id[tu.id] = await self._exec_one(tu)
                except Exception as exc:  # noqa: BLE001
                    # 只接 Exception、不接 BaseException:CancelledError 是 BaseException,
                    # 必须透传(否则 ESC 中断被吞、写类工具 turn 不停)→ 由 agent_loop 兜底回填。
                    by_id[tu.id] = self._exception_result(tu, exc)
        except asyncio.CancelledError:
            # ESC 中断:把已完成的真实结果存出再透传,供 agent_loop 只对未完成的回填 error。
            # 注意覆盖范围仅限【串行】工具:取消若落在并行 gather 期间,gather 直接抛
            # CancelledError、其下 zip 收集循环不执行,故并行分支在取消前已完成的结果不入
            # by_id。这无害——并行工具被上面的不变式限定为只读(无副作用,丢失可重算)。
            self._last_partial = dict(by_id)
            raise
        return [by_id[tu.id] for tu in tool_uses]

    def last_partial(self) -> dict[str, ToolResult]:
        """上次 execute_batch 被 cancel 时已完成的真实结果(按 tool_use_id);正常完成则空。

        供 agent_loop 在中断分支保留已执行结果:已完成用真实 output,未完成才标 error。
        **覆盖范围仅限串行工具**——取消落在并行 gather 期间时,已完成并行结果不被捕获
        (并行工具限定只读,丢失无害;见 execute_batch 的并行不变式断言)。
        """
        return self._last_partial

    async def _exec_one(self, tu: ToolUseBlock) -> ToolResult:
        tool = self._registry.get(tu.name)
        if tool is None:
            return ToolResult(tu.id, ok=False, error_message=f"未知工具: {tu.name}")
        try:
            if tool.input_schema is not None:
                # 原始 JSON schema 工具(MCP):跳过 Pydantic 校验,原样转发(server 端自校验)。
                # 防御:模型偶发给出非对象 args(json "42"/list/null)→ dict() 抛 TypeError 绕过
                # ValidationError 通路。非 dict 收口为 {},干净转发(server 通常 isError 回错)。
                args_dict = dict(tu.input) if isinstance(tu.input, dict) else {}
            else:
                args = tool.parameters.model_validate(tu.input)
                args_dict = dict(args.model_dump())
        except ValidationError as exc:
            return ToolResult(tu.id, ok=False, error_message=f"参数校验失败: {exc.errors()}")

        # 所有工具执行前查权限(校验通过后才查,避免无谓弹窗)。
        # 读类在 UiPermissionGate 内走 read-default 无条件 allow,不触 HITL;故不会弹窗。
        # gate_exempt 工具(team 协调工具,纯内存无 FS/命令副作用)跳过 gate。
        if self._permission is not None and not getattr(tool, "gate_exempt", False):
            verdict = await self._permission.check(tool, args_dict)
            if verdict.action != "allow":
                # 把 deny 的 reason 回传模型(如 L1 黑名单 hint / L2 越界 / plan 模式说明)。
                # reason 为空 → 退回默认拒绝文案,保持向后兼容。
                return ToolResult(
                    tu.id,
                    ok=False,
                    error_message=verdict.reason or "用户/规则拒绝写入(plan 模式或用户 Reject)。",
                )

        try:
            result = await tool.execute(**args_dict)
        except ToolError as exc:
            # 工具执行期失败(McpServerDownError 等):干净 ok=False,绝不抛 Traceback。
            return ToolResult(tu.id, ok=False, error_message=str(exc))

        tool_use_result: dict[str, Any] | None = None
        if isinstance(result, ToolOutput):
            output = result.text
            tool_use_result = result.tool_use_result
        else:
            output = result

        # 三轨分离(executor 双轨外存):
        #   output       → 给 UI(head/tail 截到 2000)
        #   llm_content  → 给 LLM(超阈=占位 / 不超阈=原文 / 无 sink 或落盘失败=20000 截断)
        #   full_output  → 完整原文(超阈落盘时填,供 sink 写文件;不进 jsonl)
        # UI 与 LLM 截断阈值分离——UI 沿用 2000,_DEFAULT_LLM_CHAR_LIMIT=20000,
        # 无 sink / 落盘失败时 LLM 拿 20000 截断版(恢复旧行为),不再跟 UI 跌到 2000。
        truncated, ui_output = self._truncate(output)  # UI 用默认 char_limit=2000(不变)
        full_output = ""
        persisted_path: str | None = None
        persisted_size: int | None = None
        # 默认:LLM 拿 20000 截断版(无 sink / 落盘失败均走此,恢复旧行为)
        llm_content = self._truncate(output, _DEFAULT_LLM_CHAR_LIMIT)[1]
        if self._output_sink is not None:
            # per-tool 阈值(tool.max_result_chars);math.inf → 永不落盘(read_file,
            # 避免读落盘文件→又超阈→循环)。仿 CC maxResultSizeChars。
            if len(output) > tool.max_result_chars:
                # sink.persist 抛非 OSError 异常时,旧代码会让异常逃过 ToolError except、
                # 被 execute_batch 的 except Exception 捕获 → 成功工具翻 ok=False。包 try/except:
                # 失败 → info=None + log.warning,llm_content 保持 LLM 截断版(20000)。不抛。
                try:
                    info = await self._output_sink.persist(tu.id, output)
                except Exception:  # noqa: BLE001 - 任何 persist 异常都降级,不杀工具
                    log.warning("output_sink.persist 失败,降级 inline", exc_info=True)
                    info = None
                if info is not None:
                    full_output = output
                    persisted_path = info.path
                    persisted_size = info.size
                    llm_content = info.llm_content  # 超阈:给 LLM 占位
                # 落盘失败(info=None):llm_content 保持 20000 截断版(默认值)
            else:
                # 有 sink 不超阈:给 LLM 完整原文(不截断、不落盘)。
                # 设计决策:不超阈没落盘,LLM 必须看全量(read_file 取不到落盘文件)。
                llm_content = output
        # 无 sink:llm_content 保持 20000 截断版(默认值,恢复旧行为)
        diff = self._take_diff(tu.name)  # 取走该工具的侧信道(若有)
        return ToolResult(
            tu.id,
            ok=True,
            output=ui_output,
            truncated=truncated,
            diff=diff,
            llm_content=llm_content,
            full_output=full_output,
            persisted_path=persisted_path,
            persisted_size=persisted_size,
            tool_use_result=tool_use_result,
        )

    @staticmethod
    def _take_diff(tool_name: str) -> tuple[str, str] | None:
        """从工具模块的侧信道取走最近一次 diff(Edit/Write 填;其他 None)。

        并发安全:write 类工具 parallel_safe=False → executor serial 分支串行 await,
        模块级 _last_diff 无竞态。
        """
        if tool_name == "write_file":
            from birdcode.tools.write_tool import take_last_diff

            return take_last_diff()
        if tool_name == "edit_file":
            from birdcode.tools.edit_tool import take_last_diff

            return take_last_diff()
        return None

    def list_permissions(self) -> list[Rule]:
        """暴露权限规则(经 gate 转发;无 gate → 空)。供 controller/app 的 /permissions。"""
        gate = self._permission
        if gate is not None:
            list_fn = getattr(gate, "list_rules", None)
            if callable(list_fn):
                return list(list_fn())
        return []

    def revoke_permission(self, rule_id: int) -> bool:
        gate = self._permission
        if gate is not None:
            revoke_fn = getattr(gate, "revoke", None)
            if callable(revoke_fn):
                return bool(revoke_fn(rule_id))
        return False

    def _truncate(self, text: str, limit: int | None = None) -> tuple[bool, str]:
        """截断到 limit(默认 self._char_limit=UI 2000);支持 LLM 用更大阈值。

        limit=None → 用 self._char_limit(UI 阈值,默认 2000,向后兼容);
        limit=int → 用该阈值(如 _DEFAULT_LLM_CHAR_LIMIT=20000 给 LLM)。
        逻辑同现状:head/tail 各 20%。
        """
        cap = limit if limit is not None else self._char_limit
        if len(text) <= cap:
            return False, text
        keep = max(1, cap // 5)  # head/tail 各 20%
        omitted = len(text) - 2 * keep
        head, tail = text[:keep], text[-keep:]
        return True, f"{head}\n... [已截断 {omitted} 字符 — 用 read_file 查看特定段] ...\n{tail}"

    def update_output_sink(self, sink: OutputSink | None) -> None:
        """替换 _output_sink。/clear 后从 ui/app.py 重接,指向新 session 的 sink。"""
        self._output_sink = sink

    @staticmethod
    def _exception_result(tu: ToolUseBlock, exc: BaseException) -> ToolResult:
        log.exception("tool %s failed", tu.name, exc_info=exc)
        msg = str(exc) or exc.__class__.__name__
        return ToolResult(tu.id, ok=False, error_message=f"执行异常: {msg}")


def default_registry() -> ToolRegistry:
    """默认 registry。

    只读: Glob/Grep/Read/ReadHistory;写(经权限门): Write/Edit/Delete/Bash;stub: Echo。
    """
    from birdcode.tools.bash_tool import BashTool
    from birdcode.tools.delete_tool import DeleteTool
    from birdcode.tools.echo import EchoTool
    from birdcode.tools.edit_tool import EditTool
    from birdcode.tools.glob_tool import GlobTool
    from birdcode.tools.grep_tool import GrepTool
    from birdcode.tools.read_history import ReadHistoryTool
    from birdcode.tools.read_tool import ReadTool
    from birdcode.tools.write_tool import WriteTool

    r = ToolRegistry()
    r.register(GlobTool())
    r.register(GrepTool())
    r.register(ReadTool())
    r.register(ReadHistoryTool())  # 读当前会话被压缩掉的历史对话原文
    r.register(WriteTool())
    r.register(EditTool())
    r.register(DeleteTool())  # 文件 CRUD 之 D;kind=write,经 stage4 权限门
    r.register(BashTool())  # stage5:写类,经 stage4 权限拦截
    r.register(EchoTool())  # stage1 stub,mock_provider 端到端测试仍依赖
    return r
