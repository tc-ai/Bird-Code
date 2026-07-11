from pathlib import Path

import pytest

from birdcode.config.loader import ConfigError, deep_merge, expand_env_vars, load_config


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_missing_file_returns_none(tmp_path):
    assert load_config(tmp_path / "nope.yaml") is None


def test_parses_profiles_and_sets_name(tmp_path):
    p = _write(
        tmp_path,
        """
default: ds
providers:
  ds:
    protocol: openai
    model: deepseek-reasoner
    base_url: https://api.deepseek.com
    api_key: sk-1
""",
    )
    cfg = load_config(p)
    assert cfg is not None
    assert cfg.default == "ds"
    assert cfg.providers["ds"].name == "ds"


def test_env_overrides_api_key_by_protocol(tmp_path, monkeypatch):
    p = _write(
        tmp_path,
        """
providers:
  a:
    protocol: anthropic
    model: claude-sonnet-4-5
    base_url: https://api.anthropic.com
    api_key: FILE-KEY
  b:
    protocol: openai
    model: gpt-4o
    base_url: https://api.openai.com/v1
    api_key: FILE-KEY
""",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ENV-ANT")
    monkeypatch.setenv("OPENAI_API_KEY", "ENV-OAI")
    cfg = load_config(p)
    assert cfg.providers["a"].api_key == "ENV-ANT"
    assert cfg.providers["b"].api_key == "ENV-OAI"


def test_missing_api_key_is_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = _write(
        tmp_path,
        """
providers:
  a:
    protocol: anthropic
    model: m
    base_url: u
    api_key: ""
""",
    )
    with pytest.raises(ConfigError, match="api_key"):
        load_config(p)


def test_budget_must_be_less_than_max_tokens(tmp_path):
    p = _write(
        tmp_path,
        """
max_tokens: 4096
providers:
  a:
    protocol: anthropic
    model: m
    base_url: u
    api_key: k
    thinking: { budget_tokens: 4096 }
""",
    )
    with pytest.raises(ConfigError, match="budget_tokens"):
        load_config(p)


def test_default_points_to_unknown_profile(tmp_path):
    p = _write(
        tmp_path,
        """
default: nope
providers:
  a:
    protocol: openai
    model: m
    base_url: u
    api_key: k
""",
    )
    with pytest.raises(ConfigError, match="default"):
        load_config(p)


def test_empty_providers_is_error(tmp_path):
    p = _write(tmp_path, "providers: {}\n")
    with pytest.raises(ConfigError, match="provider"):
        load_config(p)


def test_expand_env_vars_substitutes_present(monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "abc123")
    raw = {
        "mcp_servers": {
            "g": {
                "type": "streamable_http",
                "url": "https://x",
                "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
            }
        }
    }
    out = expand_env_vars(raw)
    assert out["mcp_servers"]["g"]["headers"]["Authorization"] == "Bearer abc123"


def test_expand_env_vars_missing_becomes_empty(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    raw = {"mcp_servers": {"g": {"type": "stdio", "command": "${MISSING_VAR}run"}}}
    out = expand_env_vars(raw)
    assert out["mcp_servers"]["g"]["command"] == "run"


def test_expand_env_vars_recurses_nested(monkeypatch):
    monkeypatch.setenv("A", "1")
    raw = {"a": {"b": {"c": "${A}-${A}"}}, "lst": ["${A}", "plain"]}
    out = expand_env_vars(raw)
    assert out["a"]["b"]["c"] == "1-1"
    assert out["lst"] == ["1", "plain"]


def test_expand_env_vars_leaves_non_string(monkeypatch):
    raw = {"n": 5, "b": True, "x": None}
    assert expand_env_vars(raw) == raw


def test_deep_merge_dict_recurses():
    a = {"mcp_servers": {"x": {"type": "stdio", "command": "a"}}, "default": "u"}
    b = {"mcp_servers": {"y": {"type": "stdio", "command": "b"}}, "default": "p"}
    out = deep_merge(a, b)
    assert set(out["mcp_servers"]) == {"x", "y"}
    assert out["default"] == "p"


def test_deep_merge_scalar_overrides():
    assert deep_merge({"k": 1}, {"k": 2})["k"] == 2
    assert deep_merge({"k": [1]}, {"k": [2]})["k"] == [2]


def test_load_config_two_layer_project_overrides_user(tmp_path):
    user = tmp_path / "user.yaml"
    proj = tmp_path / "proj.yaml"
    user.write_text(
        "providers:\n"
        "  u:\n"
        "    protocol: anthropic\n"
        "    model: m\n"
        "    base_url: http://u\n"
        "    api_key: k\n"
        "mcp_servers:\n"
        "  shared:\n"
        "    type: stdio\n"
        "    command: from-user\n",
        encoding="utf-8",
    )
    proj.write_text(
        "providers:\n"
        "  p:\n"
        "    protocol: anthropic\n"
        "    model: m2\n"
        "    base_url: http://p\n"
        "    api_key: k2\n"
        "mcp_servers:\n"
        "  shared:\n"
        "    type: stdio\n"
        "    command: from-proj\n",
        encoding="utf-8",
    )
    cfg = load_config(paths=[user, proj])
    assert cfg.mcp_servers["shared"].command == "from-proj"
    assert "u" in cfg.providers and "p" in cfg.providers


def test_check_perms_runs_on_all_loaded_files(tmp_path, monkeypatch):
    """回归(#7):两层合并时 _check_perms 须对【每个】实际加载的文件执行,不只是
    last_path(=项目层)——用户级 ~/.birdcode(含 api_key)也要查权限。"""
    user = tmp_path / "user.yaml"
    proj = tmp_path / "proj.yaml"
    for p, name in [(user, "u"), (proj, "p")]:
        p.write_text(
            f"providers:\n  {name}:\n    protocol: anthropic\n    model: m\n"
            f"    base_url: http://{name}\n    api_key: k\n",
            encoding="utf-8",
        )
    checked: list[Path] = []
    import birdcode.config.loader as L

    monkeypatch.setattr(L, "_check_perms", lambda p: checked.append(p))
    load_config(paths=[user, proj])
    assert checked == [user, proj]


def test_expand_env_vars_warns_on_unparseable_ref(monkeypatch):
    """回归(#11):${MY-VAR}/${host.name} 等非 \\w+ 名不会被 _VAR_RE 消费,原样残留。
    旧实现静默放过→晦涩的认证失败;须显式告警。(logger.propagate=False,故 monkeypatch)"""
    import birdcode.config.loader as L

    warned: list[str] = []

    def fake_warning(msg, *args, **kwargs):
        warned.append(msg % args if args else msg)

    monkeypatch.setattr(L.log, "warning", fake_warning)
    monkeypatch.delenv("MY_VAR", raising=False)
    out = expand_env_vars({"h": "Bearer ${MY-VAR}"})
    assert out["h"] == "Bearer ${MY-VAR}"  # 无法展开,原样保留
    assert any("MY-VAR" in w for w in warned)
