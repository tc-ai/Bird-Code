# tests/test_tool_thresholds.py
"""真实工具的 max_result_chars 阈值守护(防误改)。

补 4ff5cca review F2:此前阈值测试只用合成 _EchoTool/_LowEchoTool/read_file,
真实工具(bash/grep/edit/glob/write)的具体数值零覆盖——改坏任一值都不会有测试失败。
本文件锁定各工具的阈值,与 commit message 的「per-tool 阈值」卖点对齐。
"""
import math

from birdcode.tools.bash_tool import BashTool
from birdcode.tools.edit_tool import EditTool
from birdcode.tools.glob_tool import GlobTool
from birdcode.tools.grep_tool import GrepTool
from birdcode.tools.read_tool import ReadTool
from birdcode.tools.write_tool import WriteTool


def test_real_tool_max_result_chars():
    """各真实工具的持久化阈值(字符数)与设计一致。

    - read_file=inf:永不落盘(防「读落盘文件→再超阈」循环);context 安全靠 read_tool
      自身的 _MAX_INLINE_CHARS inline 截断兜底。
    - bash=30K / grep=20K:输出常大、可重新获取 → 紧阈值早落盘。
    - edit/glob/write=30K:BirdCode 选 30K(CC 为 100K)——更早落盘护 context。
    """
    assert ReadTool.max_result_chars == math.inf
    assert BashTool.max_result_chars == 30_000
    assert GrepTool.max_result_chars == 20_000
    assert EditTool.max_result_chars == 30_000
    assert GlobTool.max_result_chars == 30_000
    assert WriteTool.max_result_chars == 30_000
