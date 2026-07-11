"""斜杠命令：注册表 + 分发 + CommandContext 抽象。"""

from birdcode.ui.commands.command import (  # noqa: F401
    Command,
    CommandContext,
    CommandType,
)
from birdcode.ui.commands.registry import (  # noqa: F401
    CommandConflictError,
    CommandRegistry,
)
