"""命令数据模型 + CommandContext Protocol（命令实现不绑 Textual）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from birdcode.agent.context import CompactionResult
    from birdcode.agent.provider import TokenUsage
    from birdcode.agents.teammate import TeamManager
    from birdcode.config.schema import AppConfig
    from birdcode.conversation import Turn as ConversationTurn
    from birdcode.conversation import TurnController
    from birdcode.mcp.client import McpManager
    from birdcode.session.store import SessionStore
    from birdcode.tools.registry import ToolRegistry


class CommandType(StrEnum):
    LOCAL = "local"  # 纯展示/只读
    STATE = "state"  # 改 UI/会话/agent 状态
    PROMPT = "prompt"  # 送预设 prompt 给 AI


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    usage: str
    type: CommandType
    handler: Callable[[CommandContext, str], Awaitable[None]]
    aliases: tuple[str, ...] = ()
    arg_hint: str = ""
    hidden: bool = False

    def all_keys(self) -> tuple[str, ...]:
        """主名 + 别名（均含 "/"），供注册表查重。"""
        return (self.name, *self.aliases)


@runtime_checkable
class CommandContext(Protocol):
    """命令 handler 的能力面。BirdApp 实现；测试用 fake。"""

    # —— UI 协议（框架无关）——
    async def show_message(self, text: str, *, kind: str = "info") -> None: ...
    async def send_user_message(self, text: str) -> None: ...
    def get_token_usage(self) -> TokenUsage | None: ...
    def refresh_status(self) -> None: ...
    def exit_app(self) -> None: ...

    # —— 高层状态方法（复杂 mutation）——
    async def clear_conversation(self) -> str | None: ...
    async def compact_now(self) -> tuple[CompactionResult | None, str]: ...
    async def switch_profile(self, name: str) -> bool: ...

    # —— 只读句柄 ——
    @property
    def store(self) -> SessionStore | None: ...
    @property
    def tools(self) -> ToolRegistry | None: ...
    @property
    def mcp_manager(self) -> McpManager | None: ...
    @property
    def controller(self) -> TurnController | None: ...
    @property
    def cfg(self) -> AppConfig | None: ...
    @property
    def model(self) -> str: ...

    # —— agent teams(P5)——
    @property
    def team_mgr(self) -> TeamManager | None: ...

    def list_teammates(self) -> list[dict[str, str]]: ...

    def read_teammate_transcript(self, agent_id: str) -> list[ConversationTurn]: ...
