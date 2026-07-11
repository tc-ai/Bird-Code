import pytest

from birdcode.ui.commands.command import Command, CommandType
from birdcode.ui.commands.registry import CommandConflictError, CommandRegistry


async def _noop(_ctx, _args):
    return None


def _cmd(name, aliases=()):
    return Command(
        name=name,
        description="d",
        usage=name,
        type=CommandType.LOCAL,
        handler=_noop,
        aliases=aliases,
    )


def test_command_all_keys_includes_name_and_aliases():
    c = Command(
        name="/help",
        description="d",
        usage="/help",
        type=CommandType.LOCAL,
        handler=_noop,
        aliases=("/?",),
    )
    assert c.all_keys() == ("/help", "/?")


def test_command_default_aliases_empty():
    c = Command(name="/exit", description="d", usage="/exit", type=CommandType.LOCAL, handler=_noop)
    assert c.aliases == ()
    assert c.hidden is False


def test_register_and_resolve_by_name():
    r = CommandRegistry()
    r.register(_cmd("/help", aliases=("/?",)))
    assert r.resolve("/help").name == "/help"
    assert r.resolve("/?").name == "/help"  # 别名解析回主名


def test_resolve_unknown_returns_none():
    r = CommandRegistry()
    r.register(_cmd("/help"))
    assert r.resolve("/nope") is None


def test_names_excludes_hidden_by_default():
    r = CommandRegistry()
    r.register(_cmd("/help"))
    r.register(
        Command(
            name="/secret",
            description="d",
            usage="/secret",
            type=CommandType.LOCAL,
            handler=_noop,
            hidden=True,
        )
    )
    assert r.names() == ["/help"]
    assert "/secret" in r.names(include_hidden=True)


def test_complete_filters_prefix_and_excludes_hidden():
    r = CommandRegistry()
    r.register(_cmd("/help", aliases=("/?",)))
    r.register(_cmd("/sessions", aliases=("/session",)))
    r.register(
        Command(
            name="/secret",
            description="d",
            usage="/secret",
            type=CommandType.LOCAL,
            handler=_noop,
            hidden=True,
        )
    )
    got = {c.name for c in r.complete("/se")}
    assert got == {"/sessions"}  # /secret(hidden) 不参与
    assert {c.name for c in r.complete("/s")} == {"/sessions"}
    # 别名前缀也能匹配,返回主名
    assert {c.name for c in r.complete("/sessi")} == {"/sessions"}
    assert r.complete("/zzz") == []


@pytest.mark.parametrize(
    "first, second, label",
    [
        (_cmd("/help"), _cmd("/help"), "name-vs-name"),
        (_cmd("/help"), _cmd("/x", aliases=("/help",)), "name-vs-alias"),
        (_cmd("/x", aliases=("/help",)), _cmd("/help"), "alias-vs-name"),
        (_cmd("/x", aliases=("/q",)), _cmd("/y", aliases=("/q",)), "alias-vs-alias"),
    ],
)
def test_register_collision_raises(first, second, label):
    r = CommandRegistry()
    r.register(first)
    with pytest.raises(CommandConflictError):
        r.register(second)
