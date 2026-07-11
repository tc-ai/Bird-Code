# src/birdcode/utils/paths.py
"""项目根发现:从给定目录逐级向上找标准标记。

替代「裸 Path.cwd() 作项目根」——无论从 repo 哪个子目录启动,都能定位到真正的项目根,
使沙箱 project_root 与项目级 YAML 路径(.birdcode/)落在同一处。
"""

from __future__ import annotations

from pathlib import Path

# 标记优先级(同一目录多标记时,前者胜):
#   .git          —— git 仓库根(最常见、最可靠的项目边界)
#   pyproject.toml —— Python 项目根(非 git 场景兜底)
# 不用 .birdcode 作标记:它是工具写入的状态目录(migrate/append_local 创建),与用户配置
# 目录 ~/.birdcode 同名冲突——用户 home 下总有 .birdcode,会把根错误坍缩成 ~(L2 覆盖
# 整个 home)。.git/pyproject 才是真正的项目边界。
_PROJECT_MARKERS: tuple[str, ...] = (".git", "pyproject.toml")


def find_project_root(start: Path | None = None) -> Path:
    """从 start(默认 cwd)逐级向上找项目根。

    遍历 start 自身及其全部祖先,在**每个目录**按优先级检查标记;首个命中的目录即根
    (因此「最近的标记」优先,同级则 .git > pyproject.toml)。
    都没命中 → 返回 start 的 resolve(向后兼容:回退到启动目录)。
    """
    p = (start or Path.cwd()).resolve()
    for directory in (p, *p.parents):
        for marker in _PROJECT_MARKERS:
            if (directory / marker).exists():
                return directory
    return p
