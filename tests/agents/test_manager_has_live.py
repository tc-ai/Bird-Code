"""SubagentManager.has_live:幂等守卫(续跑前查,避免重复 launch)。

has_live 是 _live dict 的公共只读访问器,替代外部对 manager._live 私有字段的直访
(如 resume.py 的幂等检查)。仅查在跑异步 task,不 mutate。
"""

from birdcode.agents.manager import SubagentManager


def test_has_live_reports_tracked_agent():
    mgr = SubagentManager.__new__(SubagentManager)
    mgr._live = {"sub-1": object()}
    assert mgr.has_live("sub-1") is True
    assert mgr.has_live("sub-2") is False


def test_has_live_empty_when_no_live_dict():
    """无 _live(未 launch)→ 对任意 id 返回 False,不抛 AttributeError。"""
    mgr = SubagentManager.__new__(SubagentManager)
    mgr._live = {}
    assert mgr.has_live("anything") is False
