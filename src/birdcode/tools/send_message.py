# src/birdcode/tools/send_message.py
"""agent teams:SendMessage 工具——给 lead/teammate 发私信(fire-and-forget)。

sender 取自 _AGENT_NAME contextvar(每 agent task 经 _run_named 设置),故单一工具实例
可被 lead + 所有 teammate 共享(build_child_registry fork 返回 self)。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from birdcode.agents.mailbox import _AGENT_NAME, MailboxMessage
from birdcode.tools.base import Tool

if TYPE_CHECKING:
    from birdcode.agents.teammate import TeamManager


class SendMessageInput(BaseModel):
    to: str = Field(..., description="收件人 name(lead 或 teammate 名)")
    content: str = Field(..., description="消息正文")
    summary: str = Field("", description="一行摘要(强烈建议;recipient 一眼可见)")


class SendMessageTool(Tool):
    """给另一个 agent 发私信。sender 取自 _AGENT_NAME;recipient 经 TeamManager.resolve。"""

    name = "SendMessage"
    description = "给另一个 agent(lead 或 teammate)发私信。fire-and-forget:不阻塞、不等回复。"
    parameters = SendMessageInput
    kind = "write"
    parallel_safe = False
    gate_exempt = True  # 纯内存(投 mailbox queue),无 FS/命令副作用;async teammate 需免 gate
    team_scoped = True  # 仅 teammate(mailbox)继承;一次性 subagent 排除

    def __init__(self, team: TeamManager) -> None:
        super().__init__()
        self.team = team

    async def execute(self, *, to: str, content: str, summary: str = "") -> str:
        deliver = self.team.resolve(to)
        if deliver is None:
            known = ", ".join(self.team.names()) or "(空)"
            return f"发送失败:未知 recipient '{to}'。已知: {known}"
        deliver(MailboxMessage(sender=_AGENT_NAME.get(), to=to, content=content, summary=summary))
        return f"已发送给 {to}"
