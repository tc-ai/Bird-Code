from birdcode.mcp.errors import McpServerDownError, ToolError


def test_mcp_server_down_is_tool_error():
    err = McpServerDownError("grafana", "query_prometheus", "连接已断开")
    assert isinstance(err, ToolError)


def test_mcp_server_down_message_is_structured():
    err = McpServerDownError("grafana", "query_prometheus", "连接已断开")
    msg = str(err)
    assert "mcp__grafana__query_prometheus" in msg
    assert "<error>" in msg and "<hint>" in msg
    assert "连接已断开" in msg


def test_mcp_down_hint_does_not_promise_retry():
    """#4:断连后 call_tool pop session 且不重连,重试会再撞 session=None。
    hint 不得承诺「稍后重试」(误导),应说「重启会话后重试」。"""
    msg = str(McpServerDownError("g", "q", "断开"))
    assert "稍后重试" not in msg
    assert "重启会话" in msg
