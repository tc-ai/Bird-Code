import pytest

from birdcode.agent.system_prompt import build_system_reminder, build_system_text


def test_build_system_text_assembles_modules_and_env():
    t = build_system_text()
    assert "① 身份" in t  # 模块
    assert "环境" in t and "工作目录:" in t  # 稳定环境


def test_build_system_text_override_short_circuits():
    assert build_system_text(override="CUSTOM_PROMPT") == "CUSTOM_PROMPT"


def test_build_system_text_override_empty_falls_through():
    # 空字符串 / None 都走拼装
    assert "① 身份" in build_system_text(override="")
    assert "① 身份" in build_system_text(override=None)


@pytest.mark.asyncio
async def test_build_system_reminder_wraps_dynamic_env():
    r = await build_system_reminder()
    assert r.startswith("<system-reminder>")
    assert r.endswith("</system-reminder>")
    assert "日期:" in r


def test_build_system_text_includes_mcp_instructions():
    text = build_system_text(mcp_instructions={"grafana": "use promql here"})
    assert "MCP" in text
    assert "grafana" in text
    assert "use promql here" in text


def test_build_system_text_omits_mcp_section_when_empty():
    text = build_system_text(mcp_instructions=None)
    assert "MCP" not in text


# —— 子 agent 变体 ——


def test_build_system_text_subagent_uses_subagent_modules():
    text = build_system_text(subagent=True)
    assert "从主agent派生出来的子agent" in text  # 子 agent 身份(替代 _IDENTITY)
    assert "⑤ 工具使用" not in text  # _TOOLS 去掉
    assert "你是 BirdCode" not in text  # 主 agent 身份不在
    assert "工作目录:" in text  # 稳定环境仍在


@pytest.mark.asyncio
async def test_build_system_reminder_subagent_appends_workdir_constraint(monkeypatch):
    import birdcode.agent.system_prompt as sp

    monkeypatch.setattr(sp, "_resolve_cwd", lambda cwd=None: "/fake/wt")
    r = await build_system_reminder(subagent=True)
    assert r.startswith("<system-reminder>") and r.endswith("</system-reminder>")
    assert "你是个子agent" in r
    assert "严禁编辑/修改非工作目录下的任何文件" in r
    assert "可以读取" in r
    assert "/fake/wt" in r  # cwd 注入


@pytest.mark.asyncio
async def test_build_system_reminder_main_has_no_workdir_constraint():
    r = await build_system_reminder(subagent=False)
    assert "你是个子agent" not in r  # 主 agent 不带子 agent 工作目录约束
