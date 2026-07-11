from pydantic import BaseModel

from birdcode.tools.base import Tool


class _Plain(Tool):
    name = "plain"
    parameters = BaseModel

    async def execute(self, **args):  # type: ignore[override]
        return ""


class _Deferred(_Plain):
    name = "deferred"

    def should_defer(self) -> bool:
        return True


def test_default_input_schema_is_none():
    assert _Plain().input_schema is None


def test_default_should_defer_false():
    assert _Plain().should_defer() is False


def test_override_should_defer():
    assert _Deferred().should_defer() is True


def test_input_schema_settable():
    t = _Plain()
    t.input_schema = {"type": "object"}
    assert t.input_schema == {"type": "object"}


def test_tool_result_has_tool_use_result_field():
    from birdcode.tools.base import ToolResult

    r = ToolResult(tool_use_id="c1", ok=True)
    assert r.tool_use_result is None  # 默认 None(未填充)

    r2 = ToolResult(
        tool_use_id="c1", ok=True, tool_use_result={"agentId": "a1", "status": "completed"}
    )
    assert r2.tool_use_result == {"agentId": "a1", "status": "completed"}
