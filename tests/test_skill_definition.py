from birdcode.agents.definition import AgentDefinition, render_body


def test_agent_defaults_preserved():
    d = AgentDefinition(name="x", description="y", system_prompt="sp")
    assert d.mode == "inline"
    assert d.is_transparent is False
    assert d.body == ""


def test_skill_fields_settable():
    d = AgentDefinition(
        name="commit", description="d", system_prompt="",
        mode="fork", is_transparent=True, body="steps $ARGUMENTS",
    )
    assert d.mode == "fork"
    assert d.is_transparent is True
    assert d.body == "steps $ARGUMENTS"


def test_render_body_replaces_placeholder():
    d = AgentDefinition(name="s", description="d", system_prompt="", body="do $ARGUMENTS now")
    assert render_body(d, "X") == "do X now"


def test_render_body_no_placeholder_empty_args_unchanged():
    # 无占位符且无参数:原样返回。
    d = AgentDefinition(name="s", description="d", system_prompt="", body="no placeholder")
    assert render_body(d, "") == "no placeholder"


def test_render_body_no_placeholder_appends_args():
    # 无占位符但有参数:绝不丢弃用户输入——追加到正文末尾(冒烟回归:
    # /brainstorming <问题> 的 <问题> 曾被静默吞掉)。
    d = AgentDefinition(name="s", description="d", system_prompt="", body="body text")
    assert render_body(d, "my question") == "body text\n\nmy question"


def test_render_body_no_placeholder_whitespace_args_unchanged():
    # 纯空白参数视为无参数:原样返回(不追加空段)。
    d = AgentDefinition(name="s", description="d", system_prompt="", body="body")
    assert render_body(d, "   ") == "body"
