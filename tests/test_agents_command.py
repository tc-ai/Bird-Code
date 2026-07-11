# tests/test_agents_command.py
"""S8:/agents 命令(观察/发消息/打断 teammate)。用 fake CommandContext + fake team_mgr。"""
from birdcode.ui.commands.builtins import _agents


class _FakeHandle:
    def __init__(self, name: str, *, done: bool = False) -> None:
        self.name = name
        self.agent_id = f"sub-{name}"
        self.sent: list[tuple[str, str]] = []
        self.cancelled = False
        self._done = done

    async def send(self, msg: str, *, sender: str = "lead") -> None:
        if self._done:
            raise RuntimeError(f"teammate {self.name} 已结束,无法投递消息")
        self.sent.append((sender, msg))

    def cancel(self) -> None:
        self.cancelled = True

    def done(self) -> bool:
        return self._done


class _FakeTeam:
    def __init__(self) -> None:
        self._h: dict[str, _FakeHandle] = {}

    def register(self, name: str, h: _FakeHandle) -> None:
        self._h[name] = h

    def handle(self, name: str) -> _FakeHandle | None:
        return self._h.get(name)


class _FakeCtx:
    def __init__(self, team: _FakeTeam, info: list[dict] | None = None) -> None:
        self._team = team
        self._info = info or []
        self.messages: list[tuple[str, str]] = []

    @property
    def team_mgr(self) -> _FakeTeam:
        return self._team

    def list_teammates(self) -> list[dict]:
        return self._info

    def read_teammate_transcript(self, agent_id: str) -> list:  # noqa: ARG002
        return []

    async def show_message(self, text: str, *, kind: str = "info") -> None:
        self.messages.append((kind, text))


async def test_agents_list_empty():
    ctx = _FakeCtx(_FakeTeam(), info=[])
    await _agents(ctx, "")
    assert any("无 teammate" in m[1] for m in ctx.messages)


async def test_agents_list_with_teammates():
    ctx = _FakeCtx(
        _FakeTeam(),
        info=[{"name": "bob", "agent_id": "x", "status": "idle", "description": "做 X"}],
    )
    await _agents(ctx, "")
    assert any("bob" in m[1] and "idle" in m[1] for m in ctx.messages)


async def test_agents_send_to_teammate():
    team = _FakeTeam()
    h = _FakeHandle("bob")
    team.register("bob", h)
    ctx = _FakeCtx(team)
    await _agents(ctx, "bob 去做 Y")
    assert h.sent == [("lead", "去做 Y")]
    assert any("已发给 bob" in m[1] for m in ctx.messages)


async def test_agents_stop_teammate():
    team = _FakeTeam()
    h = _FakeHandle("bob")
    team.register("bob", h)
    ctx = _FakeCtx(team)
    await _agents(ctx, "stop bob")
    assert h.cancelled
    assert any("已打断" in m[1] for m in ctx.messages)


async def test_agents_unknown_teammate():
    ctx = _FakeCtx(_FakeTeam())
    await _agents(ctx, "nobody hi")
    assert any("未知 teammate" in m[1] for m in ctx.messages)


async def test_agents_send_to_done_warns():
    """#15:/agents 发给已结束 teammate → warn(不抛、不静默丢)。"""
    team = _FakeTeam()
    team.register("bob", _FakeHandle("bob", done=True))
    ctx = _FakeCtx(team)
    await _agents(ctx, "bob hi")
    assert any("已结束" in m[1] and m[0] == "warn" for m in ctx.messages)


async def test_agents_no_team_mgr():
    """team_mgr=None(无 cfg/store)→ 友好提示。"""
    ctx = _FakeCtx(_FakeTeam())  #假;手动覆盖 team_mgr
    ctx._team = None  # type: ignore[assignment]
    await _agents(ctx, "")
    assert any("未启用" in m[1] for m in ctx.messages)


def test_agents_command_registered():
    """S8:/agents 命令在 build_builtin_registry(LOCAL 类型)。"""
    from birdcode.ui.commands.builtins import build_builtin_registry
    from birdcode.ui.commands.command import CommandType

    reg = build_builtin_registry()
    cmd = reg.resolve("/agents")
    assert cmd is not None
    assert cmd.type == CommandType.LOCAL
