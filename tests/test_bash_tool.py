"""BashTool 测试:状态快照 / 超时 / 后台 / 危险拦截 / 错误提示词 / 校验失败。"""

from __future__ import annotations

import pytest

from birdcode.tools.bash_tool import _BASH_DESCRIPTION, BashTool


def test_bash_description_and_name():
    """description 一句话描述作用。"""
    d = _BASH_DESCRIPTION
    assert d
    assert "shell" in d.lower() or "命令" in d  # 核心作用


def test_bash_input_required_command_and_defaults():
    from birdcode.tools.bash_tool import _BashInput

    args = _BashInput(command="echo hi")
    assert args.command == "echo hi"
    assert args.timeout == 120  # §2 默认
    assert args.run_in_background is False


def test_bash_input_rejects_empty_command():
    """空 command 校验失败(由 Executor 转 ToolResult,不抛异常)——这里只验模型层。"""
    from pydantic import ValidationError

    from birdcode.tools.bash_tool import _BashInput

    with pytest.raises(ValidationError):
        _BashInput(command="")


def test_bash_tool_class_attributes():
    """§2 入参表:kind=write, parallel_safe=False。"""
    t = BashTool()
    assert t.name == "bash"
    assert t.kind == "write"
    assert t.parallel_safe is False
    assert t.description is _BASH_DESCRIPTION


# ---- Task 2: 危险拦截 + 四类错误提示词 ----


from birdcode.permission.blacklist import dangerous_hint, is_dangerous  # noqa: E402
from birdcode.tools.bash_tool import (  # noqa: E402
    _err_command_not_found,
    _err_nonzero,
    _err_timeout,
)


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf $HOME",
        "rm -rf .",  # 当前目录
        "rm -rf ./",  # 当前目录(尾斜杠)
        "rm -rf ./*",  # 当前目录通配(删光 cwd,曾漏网)
        "rm -rf .*",  # 隐藏文件通配
        "sudo apt install evil",
        "sudo rm x",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sda1",
        ":(){ :|:& };:",
        "echo hi > /dev/sda",
        ": > /dev/sdb1",
    ],
)
def test_is_dangerous_catches_blacklist(cmd):
    hit, pattern = is_dangerous(cmd)
    assert hit is True, f"应拦截但未拦截: {cmd}"
    assert pattern  # 返回命中的模式描述


@pytest.mark.parametrize(
    "cmd",
    [
        "echo hi",
        "ls -la",
        "pwd",
        "cd sub && npm test",
        "rg 'foo' src",
        "git status",
        "rm temp_file.txt",  # 非递归删单文件,不在黑名单
        "rm -rf ./build",  # 限定子目录,非 / 或 ~
    ],
)
def test_is_dangerous_passes_benign(cmd):
    hit, _ = is_dangerous(cmd)
    assert hit is False, f"误拦截: {cmd}"


def test_err_command_not_found_has_path_hint():
    msg = _err_command_not_found("npm install")
    assert msg.startswith("<error>")
    assert "command not found" in msg
    assert "npm install" in msg
    assert "<hint>" in msg
    assert "PATH" in msg or "which" in msg  # §3.6:引导 which/PATH 诊断


def test_err_nonzero_has_exit_code_and_stderr_tail():
    msg = _err_nonzero(exit_code=127, stderr_tail="command not found: foo")
    assert "127" in msg
    assert "command not found: foo" in msg
    assert "<hint>" in msg


def test_err_timeout_has_seconds_and_background_hint():
    msg = _err_timeout(60)
    assert "超时" in msg
    assert "60" in msg
    assert "<hint>" in msg
    assert "run_in_background" in msg  # §3.6:引导改后台


def test_err_dangerous_has_pattern_hint():
    msg = dangerous_hint("fork bomb / rm -rf")
    assert "危险命令" in msg
    assert "fork bomb / rm -rf" in msg
    assert "<hint>" in msg
    assert "拆解" in msg or "安全" in msg  # 引导拆解为非破坏性步骤


# ---- Task 3: execute 核心逻辑(状态快照 / 超时 / 后台 / 危险拦截) ----

import sys  # noqa: E402


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _pwd_command() -> str:
    # _resolve_shell 在 Windows 优先 git bash、POSIX 用 /bin/sh,都有 pwd
    return "pwd"


@pytest.mark.asyncio
async def test_execute_echo_returns_stdout_merged():
    t = BashTool()
    out = await t.execute(command="echo hello-bash")
    assert "hello-bash" in out


@pytest.mark.asyncio
async def test_execute_stderr_merged_into_stdout():
    """stderr=STDOUT:把 stderr 合进同一输出流(spec §6.1)。"""
    t = BashTool()
    cmd = "echo errline 1>&2"
    out = await t.execute(command=cmd)
    assert "errline" in out


@pytest.mark.asyncio
async def test_cwd_persists_across_commands():
    """§6 验收:连续 Bash 中 cd 后,下一条命令在新 cwd 执行(状态快照)。"""
    import tempfile
    from pathlib import Path

    t = BashTool()
    with tempfile.TemporaryDirectory() as d:
        # 无空格子目录:POSIX 与 Windows(cmd 的 cd)都接受不带引号的形式
        sub = Path(d) / "subdir"
        sub.mkdir()
        # cd 进 sub(带引号:git bash 下无引号的 C:\ 会被 \ 转义,引号走 MSYS mixed 路径)
        await t.execute(command=f'cd "{sub}"')
        # 第二条:pwd 反映新 cwd(git bash pwd 输出 MSYS /c/... 路径,只比目录名避免格式差异)
        out = await t.execute(command=_pwd_command())
        assert sub.name.lower() in out.replace("\\", "/").lower(), (
            f"cwd 未持久: out={out!r} expect contains {sub.name}"
        )


@pytest.mark.asyncio
async def test_cd_in_compound_command_only_affects_that_line():
    """§6.4 限制:`cd a && pwd` 中 cd 仅影响本条(同子进程)。"""
    import os

    t = BashTool()
    initial_cwd = os.getcwd()
    # 复合命令 cd . && pwd:cd . 跨平台安全(不换目录),关键是 _maybe_update_cwd
    # 因含 && 复合命令不更新 _cwd 快照
    out = await t.execute(command=f"cd . && {_pwd_command()}")
    # 本条 pwd 反映当前 cwd(git bash 输出 MSYS 路径,比目录名)
    base = os.path.basename(initial_cwd).lower()
    assert base in out.replace("\\", "/").lower(), f"pwd 未反映 cwd: {out!r}"
    # 复合命令结束后,_cwd 快照不应被改(仍是 initial cwd)
    assert t._cwd == initial_cwd, f"复合命令的 cd 误更新了快照: {t._cwd!r}"
    # 第二条独立命令在新子进程(其 cwd 取自快照 initial),输出应为 initial cwd
    out2 = await t.execute(command=_pwd_command())
    assert base in out2.replace("\\", "/").lower(), (
        f"快照未被改,但第二条未在 initial cwd 执行: out={out2!r}"
    )


@pytest.mark.asyncio
async def test_timeout_kills_process_and_returns_hint():
    """§6.2:超时 proc.kill() + 返回 §3.6 超时提示词。"""
    t = BashTool()
    if _is_windows():
        sleep_cmd = "ping -n 10 127.0.0.1 >nul"
    else:
        sleep_cmd = "sleep 5"
    out = await t.execute(command=sleep_cmd, timeout=1)
    assert "超时" in out
    assert "run_in_background" in out
    assert "1" in out


@pytest.mark.asyncio
async def test_background_returns_pid_hint():
    """§6.2:run_in_background=True 立即返回 pid(不阻塞)。"""
    import os
    import signal

    t = BashTool()
    if _is_windows():
        cmd = "ping -n 20 127.0.0.1 >nul"
    else:
        cmd = "sleep 10"
    out = await t.execute(command=cmd, run_in_background=True)
    assert "pid" in out.lower()
    assert any(ch.isdigit() for ch in out)
    # 解析 pid 并在测试结束前清理后台进程(避免残留 ping/sleep)
    import re as _re

    m = _re.search(r"pid=(\d+)", out)
    if m:
        pid = int(m.group(1))
        try:
            if _is_windows():
                os.system(f"taskkill /PID {pid} /F >nul 2>&1")
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_dangerous_command_blocked_returns_hint():
    """§6.3:危险命令不执行,返回提示词(含拆解 hint)。"""
    t = BashTool()
    out = await t.execute(command="sudo rm -rf /")
    assert "危险命令" in out
    assert "<hint>" in out
    assert "拆解" in out or "安全" in out


@pytest.mark.asyncio
async def test_command_not_found_returns_path_hint():
    """§3.6:command not found 给 PATH/which 诊断提示。"""
    t = BashTool()
    out = await t.execute(command="this_cmd_does_not_exist_xyz_123")
    assert "command not found" in out or "<error>" in out
    assert "PATH" in out or "which" in out


# ---- Task 4: 注册到 default_registry + Executor 链 ----


def test_default_registry_registers_bash():
    from birdcode.tools.executor import default_registry

    r = default_registry()
    t = r.get("bash")
    assert t is not None
    assert t.kind == "write"
    assert t.parallel_safe is False


@pytest.mark.asyncio
async def test_executor_runs_bash_serially():
    """Bash 是 parallel_safe=False → Executor 串行执行(stage1 _exec_one)。"""
    from birdcode.blocks import ToolUseBlock
    from birdcode.tools.executor import ToolExecutor, default_registry

    ex = ToolExecutor(default_registry())
    res = await ex.execute_batch(
        [ToolUseBlock(id="b1", name="bash", input={"command": "echo via-executor"})]
    )
    assert len(res) == 1
    assert res[0].ok is True
    assert "via-executor" in res[0].output


@pytest.mark.asyncio
async def test_executor_validation_failure_yields_error_result_for_bash():
    """缺 command → Pydantic 校验失败 → Executor 转 ToolResult(ok=False),不抛异常。"""
    from birdcode.blocks import ToolUseBlock
    from birdcode.tools.executor import ToolExecutor, default_registry

    ex = ToolExecutor(default_registry())
    res = await ex.execute_batch([ToolUseBlock(id="b2", name="bash", input={})])
    assert res[0].ok is False
    assert "校验失败" in res[0].error_message or "command" in res[0].error_message.lower()


@pytest.mark.asyncio
async def test_executor_serializes_two_bash_invocations():
    """两条 Bash 命令均 parallel_safe=False → 必须串行(顺序执行,互不抢占)。"""
    from birdcode.blocks import ToolUseBlock
    from birdcode.tools.executor import ToolExecutor, default_registry

    ex = ToolExecutor(default_registry())
    res = await ex.execute_batch(
        [
            ToolUseBlock(id="s1", name="bash", input={"command": "echo first"}),
            ToolUseBlock(id="s2", name="bash", input={"command": "echo second"}),
        ]
    )
    by_id = {r.tool_use_id: r for r in res}
    assert by_id["s1"].ok and "first" in by_id["s1"].output
    assert by_id["s2"].ok and "second" in by_id["s2"].output


@pytest.mark.asyncio
async def test_bash_subprocess_uses_new_process_group(monkeypatch):
    """子进程新进程组(Windows CREATE_NEW_PROCESS_GROUP / POSIX start_new_session),

    隔离 ctrl+C——配合 cli 层 SIGINT 屏蔽,防「触发 Bash 后 ctrl+C 直接退出 TUI」。
    """
    import asyncio
    import subprocess
    import sys

    captured: dict = {}

    class _FakeProc:
        pid = 123
        returncode = 0

        async def communicate(self):
            return (b"out", b"")

    async def fake_exec(*args, **kwargs):
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await BashTool().execute(command="echo hi", timeout=5)
    if sys.platform.startswith("win"):
        flags = captured.get("creationflags", 0)
        # 含 CREATE_NEW_PROCESS_GROUP(+ CREATE_NO_WINDOW:子进程不破坏主 console 鼠标 mode)
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert captured.get("start_new_session") is True


def test_decode_prefers_utf8():
    """UTF-8 优先(git bash / POSIX 输出),保证中文不被 GBK 解成乱码。"""
    from birdcode.tools.bash_tool import _decode_subprocess_output

    assert _decode_subprocess_output(b"") == ""
    assert _decode_subprocess_output("你好".encode()) == "你好"


def test_resolve_shell_windows_prefers_git_bash():
    """Windows 上若有 git bash(自带 ls/find/wc)→ 优先用它;否则回退 cmd。"""
    import sys

    if not sys.platform.startswith("win"):
        import pytest

        pytest.skip("Windows only")
    from birdcode.tools.bash_tool import _find_git_bash, _resolve_shell

    gb = _find_git_bash()
    shell = _resolve_shell()
    if gb:
        assert shell == [gb, "-c"]  # 有 git bash → 用它(Unix 命令能跑)
    else:
        assert shell == ["cmd", "/c"]


def test_extract_last_command_name():
    """取最后运行的命令名(决定退出码):管道 / && / ; 链都取最后段;跳过 VAR=;取 basename。"""
    from birdcode.tools.bash_tool import _extract_last_command_name

    assert _extract_last_command_name("cat f | grep pat") == "grep"
    assert _extract_last_command_name("echo x && grep pat f") == "grep"  # && 链取最后命令
    assert _extract_last_command_name("a ; grep pat f") == "grep"  # ; 链同理
    assert _extract_last_command_name("/usr/bin/rg foo") == "rg"
    assert _extract_last_command_name("FOO=bar cmd arg") == "cmd"
    assert _extract_last_command_name("ls") == "ls"


def test_maybe_update_cwd_skips_msys_path_on_windows():
    """Windows 上 cd /tmp(MSYS/POSIX 路径)不该把 _cwd 设成非法的 /tmp。

    即便 C:\\tmp 巧合存在致 isdir(\"/tmp\") 误判,也跳过(create_subprocess 的 cwd 需 Windows 形式)。
    """
    import sys

    if not sys.platform.startswith("win"):
        import pytest

        pytest.skip("Windows only")
    from birdcode.tools.bash_tool import BashTool

    t = BashTool()
    initial = t._cwd
    t._maybe_update_cwd("cd /tmp")
    assert t._cwd == initial  # MSYS 路径不更新快照


@pytest.mark.asyncio
async def test_grep_no_match_exit1_is_semantic_not_error():
    """grep exit 1(无匹配)是语义性退出 → 友好提示,不带 <error>(免模型误判重试)。"""
    from birdcode.tools.bash_tool import BashTool

    out = await BashTool().execute(command="grep __no_such_xyz__ /dev/null", timeout=5)
    assert "<error>" not in out
    assert "exit=1" in out and "no matches found" in out


@pytest.mark.asyncio
async def test_which_not_installed_exit1_is_semantic_not_error():
    """which <未安装命令> exit 1 = 未安装,是正常探测结果 → 友好提示,不带 <error>。

    agent 常用 `which python` 探依赖;修复前未装的命令会让 which exit 1 → 被当真错误,
    模型可能误判为失败而重试。纳入阈值表后按语义性退出处理。
    """
    from birdcode.tools.bash_tool import BashTool

    out = await BashTool().execute(command="which __no_such_cmd_xyz__", timeout=5)
    assert "<error>" not in out
    assert "exit=1" in out and "not installed" in out


def test_bash_description_recommends_dedicated_tools():
    assert "read_file" in BashTool.description
    assert "write_file" in BashTool.description
    assert "edit_file" in BashTool.description
    assert "优先" in BashTool.description


# ---- 中断正确性:ESC 中断(communicate 期 cancel)必须杀进程树,避免孤儿 ----


@pytest.mark.asyncio
async def test_cancel_during_communicate_kills_process_tree(monkeypatch):
    """回归(中断正确性):ESC 落在 communicate 的 await → CancelledError 必须杀进程树。

    修复前 execute 的 try 只接 TimeoutError,CancelledError 直接上抛、不杀进程 →
    长跑前台命令(npm run dev / pytest 长测)被 ESC 后子进程脱离 Python 继续跑,
    成孤儿占用端口/资源。修复后 except CancelledError 也调 _kill_process_tree。

    跨平台:_kill_process_tree 与进程组隔离(Windows CREATE_NEW_PROCESS_GROUP /
    POSIX start_new_session)本就双平台,同一 except 分支两端都生效。
    """
    import asyncio

    from birdcode.tools import bash_tool

    proc_killed: list = []

    class _HangProc:
        pid = 4242
        returncode = None

        async def communicate(self):
            await asyncio.sleep(1000)  # 模拟长跑命令,永不自然完成

    fake_proc = _HangProc()

    async def fake_exec(*args, **kwargs):
        return fake_proc

    def fake_kill(p):
        proc_killed.append(p)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(bash_tool, "_kill_process_tree", fake_kill)

    task = asyncio.create_task(BashTool().execute(command="echo x", timeout=120))
    await asyncio.sleep(0.1)  # 让它进入 communicate 的 sleep(1000)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert proc_killed == [fake_proc], "cancel 时未杀进程树 → 会留孤儿"
