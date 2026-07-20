"""子 agent 定义注册表:层覆盖(project > user > builtin)。"""

from __future__ import annotations

from birdcode.agents.definition import AgentDefinition


class CapabilityConflictError(Exception):
    """skill(.birdcode/skill/) 与 agent(.birdcode/agents/) 同名冲突,启动期 raise。"""


class AgentRegistry:
    """name → AgentDefinition。register 后注册者覆盖先注册者——故按
    builtin→user→project 顺序注册即得正确优先级(模仿 CommandRegistry,但加层覆盖)。"""

    def __init__(self) -> None:
        self._by_name: dict[str, AgentDefinition] = {}

    def register(self, defn: AgentDefinition) -> None:
        self._by_name[defn.name] = defn

    def resolve(self, name: str) -> AgentDefinition | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def all(self) -> list[AgentDefinition]:
        return list(self._by_name.values())
