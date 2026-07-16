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


def test_set_tool_results_dir_sets_attr(tmp_path: Path) -> None:
    """set_tool_results_dir 注入会话 tool-results 目录(缓存 resolve() 后的路径,供豁免判定)。"""
    tool = ReadTool()
    assert tool._tool_results_dir is None  # 默认 None
    tool.set_tool_results_dir(tmp_path)
    assert tool._tool_results_dir == tmp_path.resolve()  # 缓存 resolve(免每次 read_file 解析)


def test_fork_for_child_does_not_propagate_tool_results_dir(tmp_path: Path) -> None:
    """fork_for_child 不透传 _tool_results_dir:子 agent sidecar 目录与父不同,
    且子 agent 无 compaction 不产生 sidecar → 子实例默认 None。"""
    tool = ReadTool()
    tool.set_tool_results_dir(tmp_path)
    child = tool.fork_for_child(cwd=str(tmp_path))
    assert child._tool_results_dir is None


async def test_oversized_sidecar_under_tool_results_dir_is_paginated(tmp_path: Path) -> None:
    """sidecar(在 tool-results 目录下)>500KB → 不拒,分页读回 + inline 截断兜底。

    Tier1 落盘的大输出需能被 read_file 按需读回(demand-paging);豁免仅跳过尺寸闸,
    进 context 的量仍由 limit/_MAX_INLINE_CHARS 限定。
    """
    from birdcode.tools.read_tool import _MAX_READ_BYTES

    sidecar_dir = tmp_path / "tool-results"
    sidecar_dir.mkdir()
    f = sidecar_dir / "toolu_big.txt"
    f.write_bytes(b"x" * (_MAX_READ_BYTES + 1024))  # 超 500KB
    tool = ReadTool()
    tool.set_tool_results_dir(sidecar_dir)
    out = await tool.execute(file_path=str(f), limit=10)
    assert "<error>" not in out or "文件过大" not in out  # 不被尺寸闸拒
    assert "1\t" in out  # 读到内容(行号 1)


async def test_oversized_sidecar_multiline_paginates_via_stream(tmp_path: Path) -> None:
    """>500KB 多行 sidecar:流式按 offset/limit 分页读回(不整文件 read_bytes 入内存)。

    内存恒为 O(limit×行长 + 块),sidecar 无上限也不 OOM。补 F1(流式分页读)。
    """
    from birdcode.tools.read_tool import _MAX_READ_BYTES

    sidecar_dir = tmp_path / "tool-results"
    sidecar_dir.mkdir()
    f = sidecar_dir / "big.txt"
    # ~600KB 多行:每行 "rowNNNNN" 8 字节 + \n = 9 字节,68000 行 ≈ 612KB > 500KB
    f.write_text("\n".join(f"row{i:05d}" for i in range(68000)) + "\n", encoding="utf-8")
    assert f.stat().st_size > _MAX_READ_BYTES
    tool = ReadTool()
    tool.set_tool_results_dir(sidecar_dir)
    out1 = await tool.execute(file_path=str(f), offset=1, limit=5)
    assert "<error>" not in out1
    assert "1\trow00000" in out1
    assert "5\trow00004" in out1
    assert "6\t" not in out1  # 不超 limit
    # 第二页:offset 跨过第一页,行号/内容正确续接
    out2 = await tool.execute(file_path=str(f), offset=6, limit=5)
    assert "6\trow00005" in out2
    assert "10\trow00009" in out2


async def test_sidecar_offset_beyond_scan_budget_hint(tmp_path: Path, monkeypatch) -> None:
    """offset 落在扫描字节预算外 → 报错并提示 bash sed/head(不无限扫描巨大 sidecar)。"""
    from birdcode.tools import read_tool as rt

    monkeypatch.setattr(rt, "_MAX_SIDECAR_SCAN_BYTES", 4096)  # 极小预算,逼预算耗尽分支
    sidecar_dir = tmp_path / "tool-results"
    sidecar_dir.mkdir()
    f = sidecar_dir / "big.txt"
    f.write_text("\n".join(f"row{i:05d}" for i in range(68000)) + "\n", encoding="utf-8")
    tool = ReadTool()
    tool.set_tool_results_dir(sidecar_dir)
    out = await tool.execute(file_path=str(f), offset=100_000)  # 远超 4KB 预算能扫到的行
    assert "<error>" in out
    assert "sed" in out or "head" in out or "bash" in out  # 指向可行逃生


async def test_oversized_non_sidecar_still_rejected(tmp_path: Path) -> None:
    """非 sidecar 路径(不在 tool-results 目录下)>500KB → 仍拒(行为不变)。"""
    from birdcode.tools.read_tool import _MAX_READ_BYTES

    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (_MAX_READ_BYTES + 1024))
    tool = ReadTool()
    tool.set_tool_results_dir(tmp_path / "tool-results")  # 指向另一目录
    out = await tool.execute(file_path=str(f))
    assert "<error>" in out and "文件过大" in out  # 仍拒


async def test_oversized_no_dir_no_exemption(tmp_path: Path) -> None:
    """_tool_results_dir=None(未启用持久化)→ 无豁免,行为同今天。"""
    from birdcode.tools.read_tool import _MAX_READ_BYTES

    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (_MAX_READ_BYTES + 1024))
    tool = ReadTool()
    assert tool._tool_results_dir is None
    out = await tool.execute(file_path=str(f))
    assert "<error>" in out and "文件过大" in out
