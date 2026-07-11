# tests/agents/test_task_board.py
"""TaskBoard CRUD(无 Lock,同步原子)+ TeamManager.task_board 接线。"""
import pytest

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


def test_task_board_claim_atomic():
    """T1:claim 同步 CAS 原子语义。

    pending + (assignee 是 None 或自己) → in_progress + 落自己名下;
    非 pending / 指派给别人 / 不存在 → None。pull 不抢 push 已派给别人的。
    """
    board = TaskBoard()
    # pending unassigned → 任意 teammate 可抢
    board.create(title="x")  # #1
    claimed = board.claim("1", "bob")
    assert claimed is not None
    assert claimed.status == "in_progress" and claimed.assignee == "bob"

    # 已 in_progress → 再 claim(自己/他人)均 None
    assert board.claim("1", "bob") is None
    assert board.claim("1", "carol") is None

    # pending 指派给别人 → 他人抢不到,本人可 claim
    board.create(title="y", assignee="alice")  # #2 pending assignee=alice
    assert board.claim("2", "bob") is None  # bob 抢不到 alice 的
    got = board.claim("2", "alice")
    assert got is not None and got.status == "in_progress"

    # 不存在 → None
    assert board.claim("999", "bob") is None

    # 已 completed → None
    board.create(title="z")  # #3
    board.update("3", status="completed")
    assert board.claim("3", "bob") is None


def test_task_board_claim_double_claim_only_one_wins():
    """T4/#6 回归:两 teammate 先后 claim 同一 pending → 只第一个成功。

    旧实现(裸 TaskUpdate(status=in_progress))非原子:last-writer-wins,两人都以为抢到。
    claim 同步段无 await → 整段原子,第二个必见 in_progress → 失败。
    """
    board = TaskBoard()
    board.create(title="race")  # #1 pending unassigned

    first = board.claim("1", "alice")
    second = board.claim("1", "bob")

    assert first is not None and first.assignee == "alice"
    assert second is None  # bob 抢不到(已 in_progress)
    t = board.get("1")
    assert t.status == "in_progress" and t.assignee == "alice"  # last-writer-wins 根除


def test_task_board_dependency_reverse_edges():
    """T1:Task 加 blocked_by/blocks;create 建反向边。

    B blocked_by=[A] → A.blocks 追加 B.id(反向),B.blocked_by=[A.id]。无依赖 → 均空。
    """
    board = TaskBoard()
    a = board.create(title="A")  # #1
    b = board.create(title="B", blocked_by=["1"])  # #2 依赖 A
    assert b.blocked_by == ["1"]
    assert a.blocks == ["2"]  # 反向边:A blocks B

    # 无依赖 → 两字段均空
    c = board.create(title="C")  # #3
    assert c.blocked_by == [] and c.blocks == []


def test_task_board_create_status_blocked_vs_pending():
    """T2:create 判初始 status:有未完成依赖→blocked;依赖全已完成→pending;无依赖→pending。"""
    board = TaskBoard()
    board.create(title="A")  # #1 pending(未完成)
    b = board.create(title="B", blocked_by=["1"])  # 依赖未完成 A → blocked
    assert b.status == "blocked"

    # A 完成(注:自动解锁是 T3,此处仅让 A 变 completed 以测 create 判定)
    board.update("1", status="completed")
    d = board.create(title="D", blocked_by=["1"])  # 依赖已完成 A → pending
    assert d.status == "pending"

    e = board.create(title="E")  # 无依赖 → pending
    assert e.status == "pending"


def test_task_board_autounblock_on_completion():
    """T3:前置 completed → blocked 后继自动解锁为 pending。单依赖 + 多依赖。"""
    board = TaskBoard()
    board.create(title="A")  # #1
    b = board.create(title="B", blocked_by=["1"])  # #2 blocked
    assert b.status == "blocked"
    board.update("1", status="completed")  # A done → B 解锁
    assert board.get("2").status == "pending"

    # 多依赖:D 依赖 A(已完成)+ C(未完成)→ blocked;C 也完成后 → pending
    board.create(title="C")  # #3
    d = board.create(title="D", blocked_by=["1", "3"])  # #4 blocked(C 未完成)
    assert d.status == "blocked"
    board.update("3", status="completed")  # C done → D 两依赖皆完成 → 解锁
    assert board.get("4").status == "pending"


def test_claim_respects_dependency_blocked():
    """T4:claim 尊重依赖——blocked 任务 claim→None(零改动复用:claim 只认 pending)。

    claim 现有逻辑 status=='pending' 才放行,blocked≠pending 天然被挡,无需为依赖改 claim。
    解锁(blocked→pending)后 claim 即可成功。回归保护:防未来误让 claim 接受 blocked。
    """
    board = TaskBoard()
    board.create(title="A")  # #1
    board.create(title="B", blocked_by=["1"])  # #2 blocked
    assert board.claim("2", "bob") is None  # blocked → 抢不到
    assert board.get("2").status == "blocked"  # 未被改动
    board.update("1", status="completed")  # A done → B 解锁 pending
    claimed = board.claim("2", "bob")
    assert claimed is not None and claimed.assignee == "bob"  # 解锁后可抢


async def test_task_create_blocked_by_and_list_shows_deps():
    """T5:TaskCreateTool 加 blocked_by → 建 blocked 任务 + 反向边;TaskList 标注依赖。"""
    from birdcode.agents.mailbox import _AGENT_NAME
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.task_tools import TaskCreateTool, TaskListTool

    team = TeamManager()
    create = TaskCreateTool(team)
    lst = TaskListTool(team)
    _AGENT_NAME.set("lead")
    await create.execute(title="A")  # #1
    out = await create.execute(title="B", blocked_by=["1"])  # #2 blocked
    assert "#2" in out
    t2 = team.task_board.get("2")
    assert t2.status == "blocked" and t2.blocked_by == ["1"]
    assert team.task_board.get("1").blocks == ["2"]  # 反向边建好

    listing = await lst.execute()
    line2 = [ln for ln in listing.splitlines() if ln.startswith("#2")][0]
    assert "blocked" in line2 and "依赖" in line2 and "1" in line2  # 标注了依赖


def test_task_board_blocked_by_must_exist():
    """T6:blocked_by 指向不存在的 id → ValueError(无悬空依赖)。"""
    board = TaskBoard()
    board.create(title="A")  # #1
    with pytest.raises(ValueError):
        board.create(title="B", blocked_by=["1", "999"])  # 999 不存在
    # 全不存在也报
    with pytest.raises(ValueError):
        TaskBoard().create(title="X", blocked_by=["1"])


def test_task_board_dependency_is_dag_by_construction():
    """T6 补充:依赖 create 时声明 + 不可变 + 只能指向已存在 task → 结构上保证 DAG。

    无需运行时环检测:建 A 时若想依赖 B,B 必须已存在;但 B 要依赖 A 又需 A 先存在
    → 鸡生蛋,A↔B 环不可能形成(时间序保证)。这是比 DFS 环检测更深的护栏,面试可讲。
    """
    board = TaskBoard()
    a = board.create(title="A")  # #1
    board.create(title="B", blocked_by=["1"])  # #2 依赖 A(B 晚于 A)
    # A 的 blocked_by 在创建时定为空,事后无法改(不可变)→ A 不可能再依赖 B
    assert a.blocked_by == []
    # 想建一个依赖"未来 task"的 task:那 id 此时不存在 → 校验拒(上一测试已证)


async def test_task_claim_tool_execute():
    """T2:TaskClaimTool.execute → claim(id, _AGENT_NAME);成功/失败文案区分原因。"""
    from birdcode.agents.mailbox import _AGENT_NAME
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.task_tools import TaskClaimTool

    team = TeamManager()
    team.task_board.create(title="x")  # #1 pending unassigned
    claim = TaskClaimTool(team)

    _AGENT_NAME.set("bob")
    out = await claim.execute(id="1")
    assert "已领取" in out and "bob" in out
    t = team.task_board.get("1")
    assert t.status == "in_progress" and t.assignee == "bob"

    # 第二人抢同一(已 in_progress)→ 失败,文案点明"非 pending"
    _AGENT_NAME.set("carol")
    out2 = await claim.execute(id="1")
    assert "领取失败" in out2 and "非 pending" in out2

    # 指派给别人、自己抢 → 失败,文案点明指派对象
    team.task_board.create(title="y", assignee="alice")  # #2 pending assignee=alice
    _AGENT_NAME.set("bob")
    out3 = await claim.execute(id="2")
    assert "领取失败" in out3 and "alice" in out3

    # 不存在 → 失败,文案点明"不存在"
    out4 = await claim.execute(id="999")
    assert "领取失败" in out4 and "不存在" in out4


async def test_task_list_status_filter():
    """T3:TaskListTool 加可选 status 过滤;teammate TaskList(status='pending') 看可抢的。"""
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.task_tools import TaskListTool

    team = TeamManager()
    team.task_board.create(title="a")  # #1 pending
    team.task_board.create(title="b")  # #2 pending
    team.task_board.update("2", status="completed")
    lst = TaskListTool(team)

    out_all = await lst.execute()  # 无参 = 全部
    assert "#1" in out_all and "#2" in out_all
    out_pending = await lst.execute(status="pending")
    assert "#1" in out_pending and "#2" not in out_pending
    out_completed = await lst.execute(status="completed")
    assert "#2" in out_completed and "#1" not in out_completed


def test_team_tools_register_to_registry():
    """S5:team 工具注册到 registry(lead + teammate 共用;App on_mount 注册循环核心)。"""
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.registry import ToolRegistry
    from birdcode.tools.send_message import SendMessageTool
    from birdcode.tools.task_tools import (
        TaskClaimTool,
        TaskCreateTool,
        TaskListTool,
        TaskUpdateTool,
    )

    team = TeamManager()
    reg = ToolRegistry()
    for tool in (
        SendMessageTool(team),
        TaskCreateTool(team),
        TaskListTool(team),
        TaskUpdateTool(team),
        TaskClaimTool(team),
    ):
        reg.register(tool)
    names = reg.names()
    for n in ("SendMessage", "TaskCreate", "TaskList", "TaskUpdate", "TaskClaim"):
        assert n in names


def test_build_child_registry_team_scoped_exclusion():
    """#14:team_scoped 工具仅 teammate(include_team_tools=True)继承;一次性排除。

    一次性 subagent(mailbox None)非 team 成员,继承 team 工具会误归属 sender="lead";
    build_child_registry 据 include_team_tools 区分(runner 据 mailbox 是否非空传入)。
    """
    from pydantic import BaseModel

    from birdcode.agents.runner import build_child_registry
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.base import Tool
    from birdcode.tools.registry import ToolRegistry
    from birdcode.tools.send_message import SendMessageTool
    from birdcode.tools.task_tools import TaskCreateTool

    class _NoArgs(BaseModel):
        pass

    class _Plain(Tool):  # 非 team 工具(对照:两种子 agent 都继承)
        name = "plain"
        description = "d"
        parameters = _NoArgs
        kind = "read"
        parallel_safe = True

        async def execute(self):  # noqa: ARG002
            return "ok"

    team = TeamManager()
    parent = ToolRegistry()
    parent.register(SendMessageTool(team))
    parent.register(TaskCreateTool(team))
    parent.register(_Plain())

    # 一次性 subagent(include_team_tools=False):team 工具排除,普通工具保留
    child_oneshot = build_child_registry(parent, (), include_team_tools=False)
    oneshot_names = child_oneshot.names()
    assert "SendMessage" not in oneshot_names
    assert "TaskCreate" not in oneshot_names
    assert "plain" in oneshot_names

    # teammate(include_team_tools=True):team 工具保留
    child_teammate = build_child_registry(parent, (), include_team_tools=True)
    teammate_names = child_teammate.names()
    assert "SendMessage" in teammate_names
    assert "TaskCreate" in teammate_names
    assert "plain" in teammate_names
