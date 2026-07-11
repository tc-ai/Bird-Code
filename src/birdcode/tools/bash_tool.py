# src/birdcode/tools/bash.py
"""BashTool:跨命令 cwd/env 持久 + 超时杀进程 + 后台 + 危险命令拦截(状态快照会话)。

设计见 spec。状态快照(非常驻 shell):每条命令独立子进程,共用同一 cwd/env 快照,
单条超时不影响后续命令(隔离)。进阶的常驻 shell(export/source 跨命令持久)留未来。
"""

from __future__ import annotations

import asyncio
import locale
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from typing import Any, Literal

from pydantic import BaseModel, Field

from birdcode.permission.blacklist import dangerous_hint, is_dangerous
from birdcode.tools.base import Tool


def _decode_subprocess_output(data: bytes) -> str:
    """解码子进程输出。

    优先 UTF-8(git bash / POSIX 输出都是 UTF-8),失败再退系统 ANSI 代码页
    (Windows cmd 的 GBK/cp936),都不行 errors='replace' 兜底——保证非 UTF-8 输出
    不崩溃、文本可被提示词检测匹配。
    """
    if not data:
        return ""
    for enc in ("utf-8", locale.getpreferredencoding(False)):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode(errors="replace")


_BASH_DESCRIPTION = (
    "在 shell 会话中执行任意 bash 命令(运行、构建、测试、git、安装依赖等)。\n"
    "注意:\n"
    "1. 命令在同一会话中连续执行,工作目录(cwd)与环境变量状态会跨命令保留。\n"
    "2. 严禁使用交互式文本编辑器(vim/nano 等),请用 edit_file/write_file 代替。\n"
    "3. 长时间运行的命令请用 run_in_background=true(立即返回 pid 不阻塞),而非前台等待。\n"
    "4. 读/写/搜索文件优先用 read_file/write_file/edit_file/grep/glob,"
    "本工具用于它们覆盖不到的场景。"
)


# 危险命令黑名单已迁移至 birdcode.permission.blacklist(L1 单一来源)。
# BashTool 与 gate L1 共用同一份正则,避免双源漂移。


def _err_command_not_found(command: str) -> str:
    head = command.strip().split()[0] if command.strip() else command
    return (
        f"<error>command not found: {command}</error>"
        f"<hint>检查拼写 / 是否在 PATH;用 `which {head}` 定位可执行文件;"
        f"确认依赖已安装(如 {head} 需先 install)。</hint>"
    )


def _err_nonzero(*, exit_code: int, stderr_tail: str) -> str:
    tail = stderr_tail.strip()[-400:] if stderr_tail else ""
    return (
        f"<error>exit code {exit_code}: {tail}</error>"
        f"<hint>查 stderr 定位;常见:权限不足(加合适 flag)/参数错误/依赖缺失。"
        f"用 read_file 查日志、或重跑带 --verbose 看详情。</hint>"
    )


def _err_timeout(seconds: int) -> str:
    return (
        f"<error>超时(>{seconds}s 被杀)</error>"
        f"<hint>用 run_in_background=true 跑长任务(立即返回 pid 不阻塞),"
        f"或加 timeout 参数放宽。</hint>"
    )


# 语义性退出码:这些命令 exit 1 不是「执行出错」,而是有业务含义(grep 没匹配 / diff 有差异)。
# 只有 exit >= 阈值才算真错误——避免模型把「grep no match」当命令失败反复重试。
# 在工具层过滤退出码语义(而非靠 system_prompt 约束模型)。参考 mewcode/tools/bash.py。
_COMMAND_ERROR_THRESHOLDS: dict[str, int] = {
    "grep": 2,
    "egrep": 2,
    "fgrep": 2,
    "rg": 2,
    "diff": 2,
    "find": 2,
    "test": 2,
    "[": 2,
    # 环境探测:exit 1 = 命令未安装/未找到,是正常探测结果,非 Tool Error
    # (agent 常发 `which python` 探依赖;未装时 exit 1 不该被当失败重试)。
    "which": 2,
    "type": 2,
    "command": 2,
}

_EXIT_CODE_HINTS: dict[str, str] = {
    "grep": "no matches found",
    "egrep": "no matches found",
    "fgrep": "no matches found",
    "rg": "no matches found",
    "diff": "files differ",
    "find": "some directories were inaccessible",
    "test": "condition is false",
    "[": "condition is false",
    "which": "command not installed",
    "type": "command not installed",
    "command": "command not installed",
}


def _extract_last_command_name(command: str) -> str | None:
    """取最后运行的命令名(它决定整体退出码)。

    按 && / ; / | 拆,取最后一段:"cat f | grep pat"→"grep",
    "echo x && grep pat f"→"grep"。跳过 VAR=val 前缀;取 basename(/usr/bin/grep → grep)。
    """
    last_segment = re.split(r"\||&&|;", command)[-1].strip()
    if not last_segment:
        return None
    try:
        tokens = shlex.split(last_segment)
    except ValueError:
        tokens = last_segment.split()
    for token in tokens:
        if re.match(r"^[A-Za-z_]\w*=", token):
            continue
        return token.rsplit("/", maxsplit=1)[-1]
    return None


class _BashInput(BaseModel):
    command: str = Field(..., min_length=1, description="要执行的 shell 命令")
    timeout: int = Field(default=120, ge=1, description="墙钟超时秒数,超时杀进程")
    run_in_background: bool = Field(default=False, description="True: 不等待,立即返回 pid(长任务)")


def _find_git_bash() -> str | None:
    """找 git bash(MSYS,Windows 路径兼容,自带 ls/find/wc 等 coreutils)。

    优先 Program Files\\Git 下的 bash.exe;shutil.which 兜底,但排除 System32\\bash.exe
    (那是 WSL,路径用 /mnt/c/... 与 Windows 路径不兼容,模型给的 C:\\ 路径会错)。
    """
    for cand in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe"),
    ):
        if os.path.isfile(cand):
            return cand
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


def _resolve_shell() -> list[str]:
    """跨平台 shell。POSIX:/bin/sh -c。Windows:优先 git bash(MSYS——ls/find/wc/$() 可用、
    Windows 路径兼容);无 git bash 才回退 cmd /c。

    用 cmd /c 时,模型常发的 Unix 命令/语法(ls/find/wc/$(...))会全部失败(cmd 不认),
    导致「统计文件数」这种简单任务要绕 9 轮。git 安装时自带 bash.exe + coreutils,
    优先用它即可。避开 WSL 的 System32\\bash.exe(路径不兼容)。
    """
    if sys.platform.startswith("win"):
        bash = _find_git_bash()
        if bash:
            return [bash, "-c"]
        return ["cmd", "/c"]
    return ["/bin/sh", "-c"]


def _new_process_group_kwargs() -> dict[str, Any]:
    """子进程新进程组:Windows CREATE_NEW_PROCESS_GROUP / POSIX start_new_session。

    Windows 额外加 CREATE_NO_WINDOW:让子进程(git bash)拿不到主 console handle。
    否则 MSYS bash 启动时会 SetConsoleMode 改主 console 的 input mode(为 readline),
    退出后残留 → 破坏 Textual 的 ENABLE_MOUSE_INPUT,导致「触发 Bash 后鼠标点击/上滑
    全失效」。CREATE_NO_WINDOW 让子进程无 console → 不动主 mode;stdout 仍走 PIPE。
    """
    if sys.platform.startswith("win"):
        return {
            "creationflags": (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
        }
    return {"start_new_session": True}


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """杀整个进程组(含 shell 派生的子孙),避免超时留孤儿。

    仅 proc.kill() 只杀直接子 shell,不连带命令进程 → 孤儿;独立进程组(见
    _new_process_group_kwargs)让我们能 taskkill /T(Windows)或 killpg(POSIX)杀整树。
    参考 mewcode/tools/run_command.py。best-effort,失败兜底 proc.kill()。
    """
    pid = proc.pid
    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                f"taskkill /PID {pid} /T /F",
                shell=True,
                capture_output=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,  # 与 Bash 子进程一致,免闪 console 窗口
            )
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 - 杀进程 best-effort,不掩盖原超时
        pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


_CD_HEAD_RE = re.compile(r"^\s*cd\s+(.+?)\s*$")


class BashTool(Tool):
    name = "bash"
    description = _BASH_DESCRIPTION
    parameters = _BashInput
    kind: Literal["read", "write"] = "write"
    parallel_safe: bool = False
    max_result_chars: float = 30_000  # 仿 CC BashTool:30K 字符落盘

    def __init__(self) -> None:
        # 状态快照:cwd 跨命令持久,env 显式注入。
        self._cwd: str = os.getcwd()
        self._env: dict[str, str] = {}

    def fork_for_child(self, *, cwd: str | None = None) -> Tool:
        """子 agent 派生时给一个全新 BashTool。cwd 优先:worktree 子 agent 传 worktree dir
        作种子;非 worktree(cwd=None)快照父当前 _cwd(旧行为)。

        BashTool._cwd 是跨命令持久的会话级可变状态;若子 agent 复用父实例,其 `cd` 会改写
        共享实例的 _cwd、污染主 agent 的 bash 工作目录。返回全新实例:_cwd = cwd or 父当前
        快照(非进程 os.getcwd()——主 agent 用 `cd` 切换的目录不会改进程 cwd,子 agent 应
        继承主 agent 的工作目录)。子 agent 此后的 cd 只影响自己。
        """
        child = BashTool()
        child._cwd = cwd or self._cwd  # noqa: SLF001 - worktree dir 优先,否则父会话 cwd 快照
        return child

    async def execute(  # type: ignore[override]
        self, *, command: str, timeout: int = 120, run_in_background: bool = False
    ) -> str:
        # 1) 硬安全底线:危险命令拦截(bypass 模式也拦)。
        hit, label = is_dangerous(command)
        if hit:
            return dangerous_hint(label)

        shell = _resolve_shell()
        env = {**os.environ, **self._env}

        # 2) 后台:不 await,立即返回 pid。
        #    后台模式不读输出 → stdout 用 DEVNULL 避免悬空 PIPE transport
        #    (否则测试事件循环关闭时 transport GC 抛 PytestUnraisableExceptionWarning)。
        if run_in_background:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *shell,
                    command,
                    cwd=self._cwd,
                    env=env,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    **_new_process_group_kwargs(),
                )
            except OSError as exc:
                return _err_nonzero(exit_code=-1, stderr_tail=str(exc))
            return (
                f"[后台启动] pid={proc.pid}\n"
                "<hint>长任务在后台运行,不阻塞;本会话不追踪其输出/退出码。</hint>"
            )

        # 3) 前台:wait_for 包 communicate,超时 kill。
        try:
            proc = await asyncio.create_subprocess_exec(
                *shell,
                command,
                cwd=self._cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **_new_process_group_kwargs(),
            )
        except OSError as exc:
            # 启动失败(如 cwd 已不存在)→ 非零类提示词。
            return _err_nonzero(exit_code=-1, stderr_tail=str(exc))

        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            _kill_process_tree(proc)  # 杀整树(避免 shell 死、命令进程成孤儿)
            return _err_timeout(timeout)
        except asyncio.CancelledError:
            # ESC 中断:CancelledError 注入到 communicate 的 await(经 _stream_task.cancel
            # → run_agent_loop → execute_batch)。不杀则子进程脱离 Python 继续跑
            # (npm run dev / 长测占用端口)→ 孤儿。与超时同路径:_kill_process_tree
            # 杀整树,进程组已隔离(_new_process_group_kwargs),Windows/POSIX 双平台生效。
            _kill_process_tree(proc)
            raise

        text = _decode_subprocess_output(stdout_bytes or b"")
        rc = proc.returncode if proc.returncode is not None else -1

        # 4) 成功路径:更新 cwd 快照(命令首段恰为 cd <dir>);复合命令不更新。
        if rc == 0:
            self._maybe_update_cwd(command)
            return text

        # 5) 非零退出:先区分「语义性退出」(grep/find/diff 的 exit 1,非真错)与真错误。
        #    语义性退出给友好提示(不带 <error>),免得模型把「grep no match」当失败重试。
        cmd_name = _extract_last_command_name(command)
        if cmd_name is not None:
            threshold = _COMMAND_ERROR_THRESHOLDS.get(cmd_name)
            if threshold is not None and rc < threshold:
                hint = _EXIT_CODE_HINTS.get(cmd_name, "")
                tag = f"[exit={rc}" + (f", {hint}" if hint else "") + "]"
                return (text.rstrip() + "\n" + tag) if text.strip() else tag

        # 6) 真错误:127(POSIX)/9009(Win)= command not found → PATH 诊断;其余 nonzero。
        not_found_codes = {127, 9009}
        not_found_text = (
            "not recognized" in text
            or "not found" in text.lower()
            or "不是内部或外部命令" in text  # Windows cmd 中文版 command not found
        )
        if rc in not_found_codes or not_found_text:
            return text + "\n" + _err_command_not_found(command)
        return text + "\n" + _err_nonzero(exit_code=rc, stderr_tail=text)

    def _maybe_update_cwd(self, command: str) -> None:
        """命令首段恰为 `cd <dir>`(单条)→ 更新 _cwd。复合命令不在此处理。

        支持带空格路径(整体去引号,不再 split 取首 token)与 Windows `cd /d <drive:path>`。
        """
        stripped = command.strip()
        if "&&" in stripped or ";" in stripped or "|" in stripped:
            return  # 多命令:cd 仅影响本条子进程,不更新快照
        m = _CD_HEAD_RE.match(stripped)
        if not m:
            return
        target = m.group(1).strip()
        # Windows `cd /d D:\\foo`:去掉 /d 前缀(仅 cmd 用此语法切盘符)。
        if sys.platform.startswith("win") and target.startswith("/d "):
            target = target[3:].strip()
        elif sys.platform.startswith("win") and target in {"/d", "/D"}:
            return
        # 整体去外层引号(支持 `cd "my dir"` / `cd 'my dir'`),不再 split 取首 token
        # ——否则带空格路径被截成首单词,快照与现实分叉。
        target = target.strip('"').strip("'").strip()
        if not target:
            return
        # Windows + git bash:cd /tmp 这类 MSYS/POSIX 路径不能作 Windows cwd(create_subprocess
        # 的 cwd 需 C:\\... 形式);即便 C:\\tmp 巧合存在致 isdir("/tmp") 误判,也不把 _cwd
        # 设成非法的 "/tmp"。MSYS cd 仅单条有效(子进程内),不持久到快照。
        if sys.platform.startswith("win") and target.startswith("/"):
            return
        try:
            new_cwd = (
                target
                if os.path.isabs(target)
                else os.path.abspath(os.path.join(self._cwd, target))
            )
        except (OSError, ValueError):
            return
        if os.path.isdir(new_cwd):
            self._cwd = new_cwd
