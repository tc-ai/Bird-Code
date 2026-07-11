# src/birdcode/tools/agent_tools.py
"""子 agent 工具化:每个 agent defn(builtin + 自定义 .md)→ 一个 _AgentTool 实例。

取代旧的单个 AgentTool(带 subagent_type 参数)。模型选 tool 即选 agent,无需再传类型。
每工具只暴露 prompt + run_in_background;defn 绑死(含 kind/parallel_safe,按来源)。
execute 逻辑搬自旧 AgentTool(sync/async 两分支),runner/manager/侧链/唤醒零改动。

is_agent_tool=True 标记:build_child_registry 据此排除全部 agent tool(递归防护),
且规避 runner↔agent_tools 循环 import(不用 isinstance import)。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from birdcode.agents.definition import AgentDefinition, render_body
from birdcode.agents.manager import SubagentManager
from birdcode.agents.registry import AgentRegistry, CapabilityConflictError
from birdcode.agents.report import (
    build_agent_tool_use_result,
    build_launch_tool_use_result,
    build_task_notification,
)
from birdcode.agents.runner import MAX_SPAWN_DEPTH, ParentProvider, SubagentRunner
from birdcode.config.schema import AppConfig
from birdcode.permission.gate import PermissionGate
from birdcode.session.models import SessionContext
from birdcode.tools.base import Tool, ToolOutput
from birdcode.tools.registry import ToolRegistry
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.tools.agent_tools")


class AgentToolInput(BaseModel):
    prompt: str = Field(
        ...,
        description=(
            "交给子 agent 执行的完整任务描述，"
            "若任务需要一些上下文来辅助更好的理解，则提供一段精简的上下文"
        ),
    )
    isolation: Literal[None, "worktree"] = Field(
        default=None,
        description="隔离模式:'worktree'=给子 agent 一个独立 git worktree(改文件不踩主仓/兄弟),"
                    "强制异步运行。None=复用当前工作区(默认)。",
    )
    run_in_background: bool | None = Field(
        default=None,
        description="True=异步后台跑(立即回执,完成后回流);False=同步阻塞等结果。"
                    "None(默认)=用 agent 定义的 run_in_background。"
                    "isolation='worktree' 时强制 True。",
    )


class InlineSkillInput(BaseModel):
    args: str = Field("", description="传给 skill 的参数,替换正文里的 $ARGUMENTS")


class ForkSkillInput(BaseModel):
    args: str = Field("", description="传给 skill 的参数,替换正文里的 $ARGUMENTS")


class _AgentTool(Tool):
    """每个 agent defn 对应一个实例。name/description/kind/parallel_safe 取自 defn(按来源)。

    同步(defn.run_in_background=False):execute 内 await SubagentRunner.run()——阻塞调用方,
    跑完返回 ToolOutput(text=报告, tool_use_result=AgentToolUseResult)。子 agent 已销毁。

    异步(defn.run_in_background=True):构造 runner(is_async=True)交 SubagentManager 后台托管,
    立即回执 ToolOutput(text=ack, tool_use_result=launched),不阻塞;子 agent 完成后由
    manager 向主线注入 <task-notification> 并唤醒主 agent。无 SubagentManager → 回退错误。
    """

    parameters = AgentToolInput
    is_agent_tool = True  # 标记:build_child_registry 排除(递归防护 + 规避循环 import)

    def __init__(
        self,
        *,
        defn: AgentDefinition,
        cfg: AppConfig,
        app: object | None,
        ctx: SessionContext,
        project_root: Path,
        parent_provider: ParentProvider,
        parent_registry: ToolRegistry | None,
        parent_gate: PermissionGate | None,
        spawn_depth: int,
        progress_cb: Callable[..., Awaitable[None]] | None = None,
        subagent_mgr: SubagentManager | None = None,
    ) -> None:
        self.name = defn.name
        self.description = defn.description
        self.kind = defn.kind
        self.parallel_safe = defn.parallel_safe
        self._run_in_background = defn.run_in_background
        self._defn = defn
        self._cfg = cfg
        self._app = app
        self._ctx = ctx
        self._project_root = project_root
        self._parent_provider = parent_provider
        self._parent_registry = parent_registry
        self._parent_gate = parent_gate
        self._spawn_depth = spawn_depth
        self._progress_cb = progress_cb
        self._subagent_mgr = subagent_mgr

    def rebind(
        self, *, ctx: SessionContext | None = None, provider: ParentProvider | None = None
    ) -> None:
        """/clear(新 session)→ 重绑 ctx;/profile(切 provider)→ 重绑 parent_provider。

        on_mount 捕获 ctx/parent_provider;但 /clear 重开 store、/profile 重建 provider,
        二者不会自动传导。App 层状态变更时显式调用(遍历所有 agent tool)。
        """
        if ctx is not None:
            self._ctx = ctx
        if provider is not None:
            self._parent_provider = provider

    def _make_runner(
        self, *, prompt: str, is_async: bool, tool_use_id: str,
        isolation: Literal[None, "worktree"] = None,
    ) -> SubagentRunner:
        """构造子 agent runner(sync/async 共用)。execute 与 skill 的 invoke_async 共享(DRY)。"""
        return SubagentRunner(
            defn=self._defn, prompt=prompt, description=self._defn.description,
            tool_use_id=tool_use_id, model_override="",
            spawn_depth=self._spawn_depth + 1, is_async=is_async, isolation=isolation,
            parent_provider=self._parent_provider, parent_registry=self._parent_registry,
            parent_gate=self._parent_gate, cfg=self._cfg, app=self._app, ctx=self._ctx,
            project_root=self._project_root, progress_cb=self._progress_cb,
        )

    async def execute(  # type: ignore[override]
        self, *, prompt: str,
        isolation: Literal[None, "worktree"] = None,
        run_in_background: bool | None = None,
    ) -> str | ToolOutput:
        defn = self._defn
        if self._spawn_depth >= MAX_SPAWN_DEPTH:  # 防御性守卫
            return f"错误:已达最大派生深度 {MAX_SPAWN_DEPTH}(递归禁止),拒绝派生。"
        # run_in_background call 参数覆盖 defn;isolation=worktree 强制异步。
        if run_in_background is not None:
            eff_async = run_in_background
        else:
            eff_async = self._run_in_background
        if isolation == "worktree":
            eff_async = True  # worktree 必异步(sync-worktree 不支持)
        if eff_async:
            # 异步(Phase2):无 SubagentManager(旧路径/测试)→ 回退错误。
            # isolation=worktree 已强制异步,「请用 run_in_background=false」是死路 → 单独提示。
            if self._subagent_mgr is None:
                if isolation == "worktree":
                    return (
                        "错误:worktree 隔离的子 agent 未启用(SubagentManager 未注入),"
                        "无法在独立 worktree 中运行子 agent。"
                    )
                return (
                    "错误:异步子 agent(Phase2)未启用(SubagentManager 未注入);"
                    "请用 run_in_background=false。"
                )
            # tool_use_id 真实 plumb(contextvars)留后续;本轮用占位标识异步分支。
            runner = self._make_runner(
                prompt=prompt, is_async=True, isolation=isolation, tool_use_id="(async-agent)",
            )
            launch = await self._subagent_mgr.launch_async(runner)
            return ToolOutput(
                text=launch.ack_text,
                tool_use_result=build_launch_tool_use_result(
                    defn, agent_id=runner.agent_id, prompt=prompt,
                    output_file=str(runner.sidechain_path),
                ),
            )

        runner = self._make_runner(
            prompt=prompt, is_async=False, isolation=isolation, tool_use_id="(sync-agent)",
        )
        report = await runner.run()  # 同步阻塞,跑完即销毁
        return ToolOutput(
            text=build_task_notification(
                report, defn, agent_id=runner.agent_id, prompt=prompt,
                is_worktree=(runner.isolation == "worktree"),
            ),
            tool_use_result=build_agent_tool_use_result(
                report, defn, agent_id=runner.agent_id, prompt=prompt, is_async=False
            ),
        )


class _InlineSkillTool(Tool):
    """inline skill:调即返回渲染后正文(注入点 C)。不派生子 agent → is_agent_tool=False
    (子注册表可见,返回文本无递归风险)。只持 defn(不可变),无需 rebind。"""

    parameters = InlineSkillInput
    is_agent_tool = False
    kind = "read"           # 仅返回文本,无副作用;read→免派生确认、可并行
    parallel_safe = True

    def __init__(self, *, defn: AgentDefinition) -> None:
        self.name = defn.name
        self.description = defn.description
        self._defn = defn

    async def execute(self, *, args: str) -> str:  # type: ignore[override]
        return render_body(self._defn, args)


class _ForkSkillTool(_AgentTool):
    """fork skill:调即起子 agent 回摘要。复用 _AgentTool.execute(sync/async 全沿用),
    唯一差异:task = render_body(defn, args) 而非模型直传 prompt。
    is_agent_tool=True 继承 → 子注册表剔除(递归防护)。"""

    parameters = ForkSkillInput  # type: ignore[assignment]  # 窄化基类 AgentToolInput→异构 schema

    async def execute(self, *, args: str) -> str | ToolOutput:  # type: ignore[override]
        prompt = render_body(self._defn, args)
        return await super().execute(prompt=prompt)

    async def invoke_async(self, args: str) -> str:
        """slash 命令路径:始终异步启动(不阻塞 UI),返回 ack 文本。摘要经 wake 回流。
        与 execute 的 async 分支同构(复用 _make_runner);走独立方法因 slash 无 tool_use_id
        配对约束、且恒异步(不论 defn.run_in_background)。"""
        if self._subagent_mgr is None:
            return "错误:异步子 agent 未启用(SubagentManager 未注入)。"
        prompt = render_body(self._defn, args)
        runner = self._make_runner(prompt=prompt, is_async=True, tool_use_id="(skill-cmd)")
        launch = await self._subagent_mgr.launch_async(runner)
        return launch.ack_text


def register_agent_tools(
    registry: ToolRegistry, agents: AgentRegistry, **deps: object
) -> None:
    """启动时:对每个**非透明** defn(agent)造一个 _AgentTool 并注册。

    透明 defn(skill)交由 register_skill_tools 处理,此处跳过。kind/parallel_safe 直接取自
    defn(按来源)——builtin 走硬编码值,project .md 覆盖同名时走 md 值(未设则
    write/False)。名冲突(覆盖非 agent 工具)只 warning 不阻断。
    """
    for defn in agents.all():
        if defn.is_transparent:
            continue  # skill 不走 agent tool(交 register_skill_tools)
        existing = registry.get(defn.name)
        if existing is not None and not getattr(existing, "is_agent_tool", False):
            log.warning("agent '%s' 覆盖已注册的非 agent 工具,后者被取代", defn.name)
        registry.register(_AgentTool(defn=defn, **deps))  # type: ignore[arg-type]


def register_skill_tools(
    registry: ToolRegistry, caps: AgentRegistry, **deps: object
) -> dict[str, Tool]:
    """注册透明 skill 的 tool,返回 {name: tool}(供命令注册关闭 fork tool 用)。

    inline → _InlineSkillTool(仅需 defn);fork → _ForkSkillTool(需全套 deps,subclass _AgentTool)。
    非透明 defn(agent)跳过(由 register_agent_tools 处理)。
    """
    skill_tools: dict[str, Tool] = {}
    for defn in caps.all():
        if not defn.is_transparent:
            continue
        if defn.mode == "inline":
            tool: Tool = _InlineSkillTool(defn=defn)
        else:  # fork
            tool = _ForkSkillTool(defn=defn, **deps)  # type: ignore[arg-type]
        # 守卫:skill 不得静默覆盖核心工具(registry.register 只 warning+overwrite,
        # 否则名为 read_file/bash 的 skill 会顶替核心工具 → 模型调 read_file 返回 skill 正文)。
        # 能力工具(agent tool/fork-skill 都 is_agent_tool=True,inline-skill 是 _InlineSkillTool)
        # 允许互相覆盖(层覆盖语义);其余一律 fail-fast。skill-vs-agent 早由
        # build_capability_registry 拦截,此处 belt-and-suspenders + 兜底核心工具场景。
        existing = registry.get(defn.name)
        if existing is not None:
            is_capability = getattr(existing, "is_agent_tool", False) or isinstance(
                existing, _InlineSkillTool
            )
            if not is_capability:
                raise CapabilityConflictError(
                    f"skill '{defn.name}' 与已注册的核心工具同名,覆盖会破坏该工具,请改名"
                )
        registry.register(tool)
        skill_tools[defn.name] = tool
    return skill_tools
