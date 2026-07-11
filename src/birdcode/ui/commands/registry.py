"""命令注册表:name+alias → Command,启动期撞名 raise。"""

from __future__ import annotations

from birdcode.ui.commands.command import Command


class CommandConflictError(Exception):
    """两个命令占用同一 name/alias(启动期注册时检测,panic 退出)。"""


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}  # canonical name → Command
        self._alias_to_name: dict[str, str] = {}  # alias → canonical name

    def register(self, cmd: Command) -> None:
        keys = cmd.all_keys()
        # 查重:name/alias 不得撞任何已注册的 name 或 alias
        for k in keys:
            if k in self._commands:
                raise CommandConflictError(f"命令冲突: {k!r} 已注册为主名")
            if k in self._alias_to_name:
                raise CommandConflictError(f"命令冲突: {k!r} 已注册为别名")
        self._commands[cmd.name] = cmd
        for a in cmd.aliases:
            self._alias_to_name[a] = cmd.name

    def resolve(self, name: str) -> Command | None:
        """name 已小写、含 "/"。先查主名,再查别名。"""
        cmd = self._commands.get(name)
        if cmd is not None:
            return cmd
        canonical = self._alias_to_name.get(name)
        return self._commands.get(canonical) if canonical else None

    def names(self, *, include_hidden: bool = False) -> list[str]:
        return sorted(c.name for c in self._commands.values() if include_hidden or not c.hidden)

    def complete(self, prefix: str) -> list[Command]:
        """前缀匹配 name+alias(均含 "/"),排除 hidden;返回去重后的 Command 列表。"""
        if not prefix:
            return []
        out: list[Command] = []
        seen: set[str] = set()
        for c in self._commands.values():
            if c.hidden:
                continue
            hit = any(key.startswith(prefix) for key in c.all_keys())
            if hit and c.name not in seen:
                seen.add(c.name)
                out.append(c)
        return sorted(out, key=lambda c: c.name)
