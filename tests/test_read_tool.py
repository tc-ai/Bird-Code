import os
import stat
import sys
from pathlib import Path

import pytest

from birdcode.tools.read_tool import ReadTool


async def test_read_with_line_numbers(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")
    out = await ReadTool().execute(file_path=str(f))
    assert "1\tline1" in out
    assert "2\tline2" in out
    assert "3\tline3" in out


async def test_read_offset_limit_pagination(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"l{i}" for i in range(10)) + "\n", encoding="utf-8")
    out = await ReadTool().execute(file_path=str(f), offset=2, limit=3)
    # offset=2 从第 2 行起,limit=3 → 第 2,3,4 行(行号 2/3/4)
    assert "2\tl1" in out
    assert "4\tl3" in out
    assert "5\t" not in out  # 不超过 limit


async def test_read_default_limit_500(tmp_path: Path) -> None:
    """limit 默认 500:超过的文件默认只读前 500 行并提示分页。"""
    f = tmp_path / "huge.txt"
    f.write_text("\n".join(f"l{i}" for i in range(600)) + "\n", encoding="utf-8")
    out = await ReadTool().execute(file_path=str(f))
    assert "500\t" in out
    assert "501\t" not in out  # 不超过 limit
    assert "[已截断" in out or "共 600 行" in out  # 提示分页
    assert "offset" in out  # 引导下一步用 offset


async def test_read_caps_huge_output_inline(tmp_path: Path) -> None:
    """超 _MAX_INLINE_CHARS → inline 截断 + 提示分页(防爆 context)。

    长行文件(行数 < limit 但字符量巨大)触发。read 永不落盘(inf),故用 inline 截断
    而非 persist——模型无法「读落盘文件再超阈」→ 无循环。补 4ff5cca review F1。
    """
    from birdcode.tools.read_tool import _MAX_INLINE_CHARS

    f = tmp_path / "long.txt"
    f.write_text("x" * (_MAX_INLINE_CHARS * 2) + "\n", encoding="utf-8")  # 1 行,远超 cap
    out = await ReadTool().execute(file_path=str(f))
    assert len(out) < _MAX_INLINE_CHARS + 200  # 被截断(cap + 提示文案)
    assert "已截断" in out and "offset" in out  # 提示分页


async def test_read_missing_file_error_hint(tmp_path: Path) -> None:
    out = await ReadTool().execute(file_path=str(tmp_path / "nope.py"))
    assert "<error>" in out and "文件不存在" in out
    assert "<hint>" in out and "glob" in out


async def test_read_binary_file_error_hint(tmp_path: Path) -> None:
    f = tmp_path / "bin.dat"
    # 高 NUL 占比(3/5=0.60 > 0.30 阈值)→ 判二进制,避免假绿
    f.write_bytes(b"\x00\x00\x00\xff\xfe")
    out = await ReadTool().execute(file_path=str(f))
    assert "<error>" in out and "二进制" in out
    assert "<hint>" in out


async def test_oversized_file_rejected_with_grep_hint(tmp_path: Path) -> None:
    """文件 > _MAX_READ_BYTES → 拒读;提示 grep/bash,并明示 offset/limit 绕不过(防重试循环)。

    尺寸护栏按整文件 st_size 拦在 offset/limit 之前,且 read_file 整文件读入内存 ——
    故分页无法绕过。提示须指向真正可行的 grep / bash sed/head。
    """
    from birdcode.tools.read_tool import _MAX_READ_BYTES

    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (_MAX_READ_BYTES + 1024))  # 超 500KB 护栏
    out = await ReadTool().execute(file_path=str(f))
    assert "<error>" in out and "文件过大" in out
    assert "grep" in out  # 可行逃生
    assert "绕不过" in out  # 明确告知 offset/limit 无效,阻断重试循环


@pytest.mark.skipif(sys.platform == "win32", reason="unix permission model")
async def test_read_permission_denied_error_hint(tmp_path: Path) -> None:
    f = tmp_path / "secret.txt"
    f.write_text("shh", encoding="utf-8")
    os.chmod(f, 0)  # 去掉所有权限
    try:
        out = await ReadTool().execute(file_path=str(f))
        assert "<error>" in out
        assert "权限" in out or "Permission" in out
        assert "<hint>" in out
    finally:
        os.chmod(f, stat.S_IRUSR | stat.S_IWUSR)  # 恢复,免 tmp 清理失败


def test_read_description_and_name() -> None:
    d = ReadTool.description
    assert d  # 非空一句话
    assert "读取" in d  # 核心作用
    assert "行号" in d  # cat -n 式带行号
    assert ReadTool.name == "read_file"
