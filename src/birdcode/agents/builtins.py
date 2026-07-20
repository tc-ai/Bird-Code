"""内置子 agent 定义 + registry 组装(builtin→user→project)。"""

from __future__ import annotations

from pathlib import Path

from birdcode.agents.definition import AgentDefinition, AgentSource
from birdcode.agents.loader import load_md_agents
from birdcode.agents.registry import AgentRegistry, CapabilityConflictError

_EXPLORE_SP = (
    "你是一个文件搜索专家。这是一个只读探索任务。严禁：创建文件、修改文件、删除文件、执行任何改变系统状态的命令。"
    "你的工具使用策略：用glob做文件模式匹配，用grep搜索文件内容用read_file读取已知路径的文件"
    "bash只用于只读操作（ls、gitlog、git diff、find、cat)、"
    "尽可能并行发起多个工具调用以提高效率高效完成搜索请求，清晰报告发现。"
)
_PLAN_SP = (
    "你是一个软件架构师和规划专家。这是一个只读规划任务。严禁：创建文件、修改文件、删除文件、执行任何改变系统状态的命令。"
    "你的工作流程：1.理解需求，明确设计视角"
    "2，用搜索工具充分探索代码库：找到现有模式和约定，理解当前架构，识别可参考的类似功能"
    "3，设计方案：制定实现路径，考虑取舍和架构决策"
    "4，输出计划：提供分步实现策略，标明依赖和顺序，预判潜在挑战"
    "5.回复末尾必须列出3-5个对实现最关键的文件路径。"
)
_GP_SP = (
    "你是BirdCode的Agent。根据用户的消息，使用可用工具完成任务。把任务做完，不要过度设计，但也不要做一半就停。完成后用简洁的报告回复：做了什么、关键发现。"
    "调用方会把结果转述给用户，所以只需要包含要点。"
    "搜索策略：不确定位置时广泛搜索，确定路径时直接读取。"
    "优先编辑现有文件，不要主动创建文档文件。"
)

# 只读 agent(explore/plan)禁用的「可变」工具:用 registry 真实工具名(write_file/
# edit_file/delete_file,不是类名 Write/Edit/Delete——后者精确匹配永不命中,黑名单静默
# 失效)。bash 不在禁用之列:explore/plan 的 system_prompt 约束其只用于只读操作
# (ls/git log/git diff/find/cat);代价是 bypass 模式下「只读」契约降级为提示词级约束。
_READONLY_DISALLOWED: tuple[str, ...] = ("write_file", "edit_file", "delete_file")

BUILTIN_AGENTS: tuple[AgentDefinition, ...] = (
    AgentDefinition(
        name="explore",
        description="创建一个只读代码的子agent。专注在复杂代码库中进行广度探索，快速精准定位相关文件、代码片段与上下文依赖。仅提供信息检索与定位服务，不会进行任何代码评审或修改操作。",
        system_prompt=_EXPLORE_SP,
        disallowed_tools=_READONLY_DISALLOWED,
        source=AgentSource.BUILTIN,
        kind="read",
        parallel_safe=True,
    ),
    AgentDefinition(
        name="plan",
        description="创建一个软件架构师子agent，专注于规划任务。深度剖析现有代码结构与业务逻辑，设计严谨的技术实现方案。将复杂需求拆解为清晰、可执行的步骤计划，为后续编码提供全局视角的架构指导与任务编排，不会进行任何代码修改操作",
        system_prompt=_PLAN_SP,
        disallowed_tools=_READONLY_DISALLOWED,
        source=AgentSource.BUILTIN,
        kind="read",
        parallel_safe=True,
    ),
    AgentDefinition(
        name="general-purpose",
        description="创建一个与主agent能力相当的子agent。具备完整的读写与执行权限，负责落实具体的开发、调试与重构任务。作为主力开发节点，能独立处理各类常规编码需求，将架构计划高效转化为高质量落地代码。若以worktree来创建该子agent，则子agent最终产物在.birdcode/worktrees/<sub-id>/下",
        system_prompt=_GP_SP,
        disallowed_tools=(),
        source=AgentSource.BUILTIN,
        kind="write",
        parallel_safe=False,
    ),
)


def build_agent_registry(
    project_root: Path,
    *,
    user_dir: Path | None = None,
    project_dir: Path | None = None,
) -> AgentRegistry:
    """组装三层 registry:builtin → user(~/.birdcode/agents)→ project(<root>/.birdcode/agents)。

    后注册覆盖先 → project > user > builtin。user_dir/project_dir 仅测试覆盖用。
    """
    r = AgentRegistry()
    for defn in BUILTIN_AGENTS:
        r.register(defn)
    user = user_dir if user_dir is not None else Path.home() / ".birdcode" / "agents"
    for defn in load_md_agents(user, AgentSource.USER):
        r.register(defn)
    proj = project_dir if project_dir is not None else project_root / ".birdcode" / "agents"
    for defn in load_md_agents(proj, AgentSource.PROJECT):
        r.register(defn)
    return r


def build_capability_registry(
    project_root: Path,
    *,
    user_skill_dir: Path | None = None,
    project_skill_dir: Path | None = None,
    _project_agent_dir: Path | None = None,  # 测试覆盖用(避免依赖 ~/.birdcode)
    _user_agent_dir: Path | None = None,  # 测试覆盖用(避免依赖 ~/.birdcode/agents)
) -> AgentRegistry:
    """合并 agents(.birdcode/agents/,非透明)+ skills(.birdcode/skill/,透明)。

    跨目录同名 → CapabilityConflictError(transparent 与 hidden 暴露策略不同,
    静默覆盖会让老 agent 莫名变斜杠命令)。单目录内三层覆盖仍 last-wins。
    """
    from birdcode.agents.skill_loader import build_skill_registry

    # 测试覆盖模式:只要传了任一 _*_agent_dir,就视为隔离测试——未显式传入的层指向
    # 不存在的空目录(→ load_md_agents 返回 []),绝不回落真实 ~/.birdcode。否则只传
    # _project_agent_dir 而 _user_agent_dir=None 时,build_agent_registry 会把 None
    # 当「用 home」,加载开发机本机 agent(污染测试 + 可能同名冲突)。
    if _project_agent_dir is not None or _user_agent_dir is not None:
        _empty = project_root / "__birdcode_test_empty__"
        agents = build_agent_registry(
            project_root,
            user_dir=_user_agent_dir if _user_agent_dir is not None else _empty,
            project_dir=_project_agent_dir if _project_agent_dir is not None else _empty,
        )
    else:
        agents = build_agent_registry(project_root)
    skills = build_skill_registry(
        project_root, user_dir=user_skill_dir, project_dir=project_skill_dir
    )
    # agents 是本函数新建的局部 AgentRegistry、无外部别名 → 直接作合并基座,免去一次性
    # 拷贝(旧实现 merged=AgentRegistry()+遍历再插,全是垃圾分配)。
    merged = agents
    for defn in skills.all():
        if merged.resolve(defn.name) is not None:
            raise CapabilityConflictError(f"skill '{defn.name}'(.birdcode/skill/)与 agent 同名冲突")
        merged.register(defn)
    return merged
