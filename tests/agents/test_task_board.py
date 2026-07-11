# tests/agents/test_task_board.py
"""TaskBoard CRUD(无 Lock,同步原子)+ TeamManager.task_board 接线。"""
from birdcode.agents.task_board import TaskBoard


def test_task_board_create_get_list_update():
    board = TaskBoard()
    t1 = board.create(title="review auth", assignee="bob", created_by="lead")
    assert t1.id == "1" and t1.status == "pending"
    assert t1.assignee == "bob" and t1.created_by == "lead"
    t2 = board.create(title="second")
    assert t2.id == "2" and t2.assignee is None
    assert board.get("1") is t1
    assert board.get("999") is None
    assert len(board.list()) == 2
    updated = board.update("1", status="completed")
    assert updated.status == "completed" and updated.assignee == "bob"  # assignee 不变
    assert board.update("999", status="x") is None  # 未知 id


def test_team_manager_has_task_board():
    from birdcode.agents.teammate import TeamManager

    team = TeamManager()
    assert team.task_board is not None
    team.task_board.create(title="x")
    assert len(team.task_board.list()) == 1


async def test_task_tools_crud_via_execute():
    """TaskCreate/List/Update 经 execute → 看板变化;created_by 取 _AGENT_NAME。"""
    from birdcode.agents.mailbox import _AGENT_NAME
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.task_tools import TaskCreateTool, TaskListTool, TaskUpdateTool

    team = TeamManager()
    create = TaskCreateTool(team)
    lst = TaskListTool(team)
    update = TaskUpdateTool(team)

    _AGENT_NAME.set("lead")
    out = await create.execute(title="review auth", assignee="bob")
    assert "#1" in out and "review auth" in out
    out2 = await lst.execute()
    assert "#1" in out2 and "bob" in out2
    out3 = await update.execute(id="1", status="completed")
    assert "completed" in out3
    t = team.task_board.get("1")
    assert t.status == "completed" and t.created_by == "lead"
    out4 = await update.execute(id="999", status="x")
    assert "未知" in out4


async def test_task_update_assignee_only_does_not_echo_status():
    """#12 回归:仅改 assignee(status 未传)→ 回执不应附带 status=(旧实现误报未变的 status)。

    旧实现无条件 `parts=[f"status={t.status}"]`,只改 assignee 时会把未变的 status(如 pending)
    一并回显,误导模型以为自己也设了状态。修复后只回显实际传入的字段。
    """
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.task_tools import TaskUpdateTool

    team = TeamManager()
    team.task_board.create(title="x", assignee="bob", created_by="lead")
    update = TaskUpdateTool(team)
    out = await update.execute(id="1", assignee="carol")  # 只改 assignee
    assert "assignee=carol" in out
    assert "status=" not in out, "仅改 assignee 时不应回显 status(旧实现误报 status=pending)"

    # 反向:传了 status → 应回显 status
    out2 = await update.execute(id="1", status="in_progress")
    assert "status=in_progress" in out2
