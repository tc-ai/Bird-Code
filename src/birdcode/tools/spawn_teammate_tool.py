# src/birdcode/tools/spawn_teammate_tool.py
"""P5:SpawnTeammate 工具——lead 调,起一个长驻 teammate(mailbox 模式)。

teammate ≠ 一次性 subagent:跑完首轮 → park(await mailbox.get)→ 收 SendMessage 唤醒续跑,
长驻到 cancel/shutdown。复用 SubagentRunner 全套入参(同 _AgentTool);区别:mailbox 非空 +
TeamManager 托管。is_agent_tool=True → build_child_registry 排除(teammate 不再 spawn teammate,
递归防护);gate_exempt=True(协调操作,无 FS/命令副作用)。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, Field

from birdcode.agents.definition import AgentDefinition
from birdcode.agents.runner import ParentProvider, SubagentRunner
from birdcode.agents.teammate import TeamManager
from birdcode.config.schema import AppConfig
from birdcode.permission.gate import PermissionGate
from birdcode.session.models import SessionContext
from birdcode.tools.base import Tool
from birdcode.tools.registry import ToolRegistry


class SpawnTeammateInput(BaseModel):
    name: str = Field(..., description="teammate 的人话名(team 内唯一,经它 SendMessage 寻址)")
    prompt: str = Field(..., description="交给 teammate 的首轮任务描述(跑完后 park 等唤醒)")


class SpawnTeammateTool(Tool):
    """起一个长驻 teammate。lead 协调用:派 teammate 干活、跑完 park、可被 SendMessage 唤醒。"""

    name = "SpawnTeammate"
    description = (
        "起一个长驻 teammate(跑完首轮后 park,经 SendMessage 唤醒续跑)。"
        "lead 协调多 teammate 并行干活时用。"
    )
    parameters = SpawnTeammateInput
    kind = "write"
    parallel_safe = False
    gate_exempt = True
    is_agent_tool = True  # 子注册表排除:teammate 不再 spawn teammate(递归防护)

    def __init__(
        self, *,
        team_mgr: TeamManager,
        defn: AgentDefinition,
        cfg: AppConfig,
        app: object | None,
        ctx: SessionContext,
        project_root: Path,
        parent_provider: ParentProvider,
        parent_registry: ToolRegistry | None,
        parent_gate: PermissionGate | None,
        progress_cb: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self.team_mgr = team_mgr
        self.defn = defn
        self.cfg = cfg
        self.app = app
        self.ctx = ctx
        self.project_root = project_root
        self.parent_provider = parent_provider
        self.parent_registry = parent_registry
        self.parent_gate = parent_gate
        self.progress_cb = progress_cb

    async def execute(self, *, name: str, prompt: str) -> str:
        if name in self.team_mgr.names():
            return f"错误:teammate '{name}' 已存在,请换名。"
        mailbox: asyncio.Queue = asyncio.Queue()
        runner = SubagentRunner(
            defn=self.defn, prompt=prompt, description=self.defn.description,
            tool_use_id="(teammate)", model_override="",
            spawn_depth=1, is_async=True,  # lead(depth 0)→ teammate depth 1
            parent_provider=self.parent_provider, parent_registry=self.parent_registry,
            parent_gate=self.parent_gate, cfg=self.cfg, app=self.app, ctx=self.ctx,
            project_root=self.project_root, progress_cb=self.progress_cb,
            mailbox=mailbox,
        )
        await self.team_mgr.spawn(name, runner)
        return (
            f"已启动 teammate {name}(agent_id={runner.agent_id}),"
            "跑完首轮后 park 等 SendMessage。"
        )
