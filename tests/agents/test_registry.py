from birdcode.agents.definition import AgentDefinition, AgentSource
from birdcode.agents.registry import AgentRegistry


def _d(name, source, desc="x"):
    return AgentDefinition(name=name, description=desc, system_prompt="SP", source=source)


def test_register_and_resolve():
    r = AgentRegistry()
    r.register(_d("explore", AgentSource.BUILTIN))
    assert r.resolve("explore") is not None
    assert r.resolve("nope") is None


def test_layer_override_user_beats_builtin():
    r = AgentRegistry()
    r.register(_d("explore", AgentSource.BUILTIN, "builtin-desc"))
    r.register(_d("explore", AgentSource.USER, "user-desc"))
    assert r.resolve("explore").description == "user-desc"  # 后注册覆盖


def test_layer_order_project_beats_user_beats_builtin():
    r = AgentRegistry()
    r.register(_d("x", AgentSource.BUILTIN, "b"))
    r.register(_d("x", AgentSource.USER, "u"))
    r.register(_d("x", AgentSource.PROJECT, "p"))
    assert r.resolve("x").description == "p"


def test_names_and_all():
    r = AgentRegistry()
    r.register(_d("a", AgentSource.BUILTIN))
    r.register(_d("b", AgentSource.BUILTIN))
    assert set(r.names()) == {"a", "b"}
    assert len(r.all()) == 2
