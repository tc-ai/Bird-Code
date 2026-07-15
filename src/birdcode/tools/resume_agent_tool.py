# src/birdcode/tools/resume_agent_tool.py
"""主 agent 中介续跑工具:调 resume_subagent,带用户新方向。

主 agent 看到「未完成任务」提示 + 用户「继续[+方向]」后调本工具(agent_id + direction)。
sync 子 agent → 阻塞 inline 返回报告;async → 回 ack,完成走 task-notification。
授权闸门在触发方(UI /「继续」路由,T11/T13);本工具假定已授权。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from birdcode.agents.resume import ResumeDeps, resume_subagent
from birdcode.tools.base import Tool

if TYPE_CHECKING:
    from birdcode.agents.runner import ParentProvider
    from birdcode.session.models import SessionContext


class ResumeAgentInput(BaseModel):
    agent_id: str = Field(
        ...,
        description="要续跑的子 agent id(sub-<hex>,来自未完成任务提示)",
    )
    direction: str = Field(
        ...,
        description="续跑指令/新方向,会作为新 user turn 进子 agent 上下文",
    )


class ResumeAgentTool(Tool):
    """单个工具(非 per-defn)。按 agent_id 查 meta、复用 id、加载侧链续跑。

    与 _AgentTool 的 per-defn 注册不同:本工具是注册一次的单一工具,name 固定
    'resume_agent'。execute 直接转发给 resume_subagent(sync/async 分支由其按
    meta.is_async 派发),返回 result.text(sync=报告 / async=ack / in_progress=已在跑)。
    """

    parameters = ResumeAgentInput
    kind = "write"            # 续跑会执行工具,有副作用
    parallel_safe = False
    # 派生子 agent(复用 id)→ build_child_registry 排除,堵递归(同类 _AgentTool/SpawnTeammate)
    is_agent_tool = True

    def __init__(self, *, deps: ResumeDeps) -> None:
        self.name = "resume_agent"
        self.description = (
            "续跑一个被中断的子 agent(断电/关终端/ESC)。传 agent_id 与续跑方向;"
            "sync 子 agent 阻塞等结果,async 后台跑完成后通知。"
        )
        self._deps = deps

    def rebind(
        self, *, ctx: SessionContext | None = None, provider: ParentProvider | None = None
    ) -> None:
        """/clear(新 session)→ 重绑 ctx + session_id;/profile(切 provider)→ 重绑 parent_provider。

        与 _AgentTool.rebind 同语义。deps.manager 由 SubagentManager.rebind 独立刷新(同一实例,
        引用随之更新);root/worktree_name 随 cwd 不变。漏重绑则 /clear 后本工具仍指向旧 session
        目录 → 续跑找不到新 session 的子 agent meta。也使 app._rebind_agent_tool 遍历 is_agent_tool
        工具时不再因缺 rebind 抛 AttributeError(此前 /clear 会跳过其后的 manager/team 重绑)。
        """
        if ctx is not None:
            self._deps.ctx = ctx
            self._deps.session_id = ctx.session_id
        if provider is not None:
            self._deps.parent_provider = provider

    async def execute(self, *, agent_id: str, direction: str) -> str:  # type: ignore[override]
        result = await resume_subagent(
            agent_id=agent_id, direction=direction, deps=self._deps
        )
        return result.text
