# src/birdcode/agent/system_prompt/instructions.py
"""项目指令(BirdCode.md)加载:多层优先级 + @include 递归展开 + 安全护栏。

启动时一次性读取并拼成一段指令文本,prepend 进 system prompt(项目级在前=高优先级),
让 Agent 在处理用户请求前就「读完入职文档」。来源(高→低):
  1. <project_root>/BirdCode.md   —— 项目级
  2. ~/.birdcode/BirdCode.md      —— 用户级

@include 语法(仅在 fenced code block 之外的行首生效,见 _expand_includes):
    @include relative/path.md       # 引用文件
    @include src                     # 引用目录 → 自动找 src/BirdCode.md(找到则引入,否则放弃)
    @include "name with spaces.md"
相对路径按「所在文件目录」解析;绝对路径原样;~ 不展开(防 @include ~/.ssh/id_rsa)。

安全护栏(任一触发都记日志 + 留 <!-- --> 标记,绝不抛异常):
  - 嵌套深度上限 MAX_INCLUDE_DEPTH。
  - visited 集合:按 resolve() 后的绝对路径去重 + 防环(同一文件只展开一次)。
  - 路径越界拦截:每个来源的 include 链有独立 root(项目级=project_root,用户级=~/.birdcode),
    目标 resolve() 后必须落在 root 内(is_relative_to),拦 ../../、绝对路径越界、symlink 逃逸。
  - 单文件体积上限 MAX_FILE_BYTES、总量上限 MAX_TOTAL_BYTES。
  - 二进制嗅探(前 8KB 含 NUL → 跳过)。

同步 I/O 偏离 「asyncio for I/O」约定:此处在 run_tui 同步启动期、事件循环/TUI 尚未
运行,一次性读取,同步最简单;挪到异步上下文时再改。BirdCode.md 会话中途修改不热更新
(需重启)——换取 system block 字节稳定、利于前缀缓存命中。
"""

from __future__ import annotations

import re
from pathlib import Path

from birdcode.utils.logging import get_logger

log = get_logger("birdcode.instructions")

PROJECT_FILE = "BirdCode.md"
MAX_INCLUDE_DEPTH = 5
MAX_FILE_BYTES = 256 * 1024  # 单文件 256KB
MAX_TOTAL_BYTES = 1024 * 1024  # 总量 1MB

# 行首(允许缩进) @include + 路径(裸路径或双引号路径),行尾只允许空白。
# 行首锚定 + 围栏感知(见 _expand_includes)共同避免展开正文/代码样例里的 @include。
_INCLUDE_RE = re.compile(r'(?m)^[ \t]*@include\s+(?:"(?P<q>[^"]+)"|(?P<bare>\S+))[ \t]*$')
# fenced code block 起止行:``` 或 ~~~(允许缩进)。
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")


def load_project_instructions(project_root: Path) -> str:
    """加载并拼装项目指令。项目级在前(高优先级)、用户级在后;都无则返回 ""。

    每个来源独立 try/except:一个来源坏掉不波及另一个(优雅降级)。
    """
    proj_root = project_root.resolve()
    user_root = (Path.home() / ".birdcode").resolve()
    # (文件, 该链的 containment root, 来源标签) —— 高优先级在前。
    candidates: list[tuple[Path, Path, str]] = [
        (proj_root / PROJECT_FILE, proj_root, "项目级"),
        (user_root / PROJECT_FILE, user_root, "用户级"),
    ]
    visited: set[Path] = set()  # 全局共享:跨来源去重 + 防环
    acc: list[int] = [0]  # 已累积字节数(总量上限用;list 作可变引用)
    blocks: list[str] = []
    for path, root, origin in candidates:
        if not path.is_file():
            continue
        try:
            content = _resolve_file(path, root=root, visited=visited, depth=0, acc=acc)
        except Exception:  # noqa: BLE001 — 单来源失败不波及其他
            log.exception("加载项目指令失败 %s", path)
            continue
        if not content.strip():
            continue
        blocks.append(f"项目指令（BirdCode.md · {origin}）\n\n{content}")
    return "\n\n---\n\n".join(blocks) if blocks else ""


def _resolve_file(
    path: Path,
    *,
    root: Path,
    visited: set[Path],
    depth: int,
    acc: list[int],
) -> str:
    """读取 path 并递归展开其中的 @include。返回展开后文本(失败处留 <!-- 标记 -->)。"""
    resolved = path.resolve()

    # 1) 环路 / 去重(按 resolve 后绝对路径):同一文件只展开一次。
    if resolved in visited:
        log.warning("@include 循环/重复引用已跳过: %s", resolved)
        return f"<!-- @include 循环/重复引用已跳过: {resolved} -->"
    # 2) 嵌套深度。
    if depth > MAX_INCLUDE_DEPTH:
        log.warning("@include 嵌套过深(>%d)已跳过: %s", MAX_INCLUDE_DEPTH, resolved)
        return f"<!-- @include 嵌套过深(>{MAX_INCLUDE_DEPTH})已跳过: {resolved} -->"
    # 3) 总量上限:超出则停止继续展开。
    if acc[0] >= MAX_TOTAL_BYTES:
        log.warning("@include 总量超 %d 字节,停止展开: %s", MAX_TOTAL_BYTES, resolved)
        return f"<!-- @include 总量超限,停止展开: {resolved} -->"
    # 4) 存在性。
    if not resolved.exists():
        log.warning("@include 未找到: %s", resolved)
        return f"<!-- @include 未找到: {resolved} -->"
    # 4b) 兜底:目标是目录。正常路径下 _resolve_target 已把目录转为 <dir>/BirdCode.md,
    #     此处仅当 BirdCode.md 本身是个目录(病态)等极端情况触发,给出清晰提示而非笼统「读取失败」。
    if resolved.is_dir():
        log.warning("@include 目标是目录,需指向文件: %s", resolved)
        return f"<!-- @include 目标是目录,需指向文件: {resolved} -->"
    # 5) 单文件体积(stat 取大小,避免把超大文件整份读进内存)。
    try:
        size = resolved.stat().st_size
    except OSError:
        log.warning("@include 无法读取(stat 失败): %s", resolved)
        return f"<!-- @include 无法读取: {resolved} -->"
    if size > MAX_FILE_BYTES:
        log.warning("@include 单文件超 %d 字节已跳过: %s(%d 字节)", MAX_FILE_BYTES, resolved, size)
        return f"<!-- @include 单文件超限({size} 字节)已跳过: {resolved} -->"
    # 6) 读取(UTF-8;非 UTF-8 字节用 replace 兜底,防 decode 崩)。
    try:
        raw = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("@include 读取失败: %s", resolved)
        return f"<!-- @include 读取失败: {resolved} -->"
    # 7) 二进制嗅探:前 8KB 含 NUL → 大概率非文本,跳过(errors=replace 否则会灌入乱码)。
    if "\x00" in raw[:8192]:
        log.warning("@include 疑似二进制文件已跳过: %s", resolved)
        return f"<!-- @include 疑似二进制文件已跳过: {resolved} -->"

    visited.add(resolved)
    acc[0] += len(raw)
    return _expand_includes(
        raw, base=resolved.parent, root=root, visited=visited, depth=depth, acc=acc
    )


def _expand_includes(
    text: str,
    *,
    base: Path,
    root: Path,
    visited: set[Path],
    depth: int,
    acc: list[int],
) -> str:
    """逐行扫描:跟踪 fenced code block 状态,仅在围栏外把行首 @include 展开为文件内容。"""
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in text.split("\n"):
        m_fence = _FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            out.append(line)
            continue
        if in_fence:
            out.append(line)  # 围栏内:原样保留(含 @include 样例)
            continue
        m = _INCLUDE_RE.match(line)
        if not m:
            out.append(line)
            continue
        target = m.group("q") or m.group("bare") or ""
        inc_path = _resolve_target(target, base=base, root=root)
        if inc_path is None:
            log.warning("@include 路径越界已拒绝: %s (base=%s, root=%s)", target, base, root)
            out.append(f"<!-- @include 路径越界已拒绝: {target} -->")
            continue
        out.append(
            _resolve_file(inc_path, root=root, visited=visited, depth=depth + 1, acc=acc)
        )
    return "\n".join(out)


def _resolve_target(target: str, *, base: Path, root: Path) -> Path | None:
    """解析 @include 目标,返回 resolve() 后的绝对**文件**路径;越界(逃出 root)返回 None。

    相对路径按 base(=所在文件目录)解析;绝对路径原样;~ 不展开(原样当相对名,落在 base 下、
    通常不存在 → 上层报「未找到」,绝不会读到 home 下的私密文件)。

    **目录目标自动解析为「该目录下的 BirdCode.md」**——`@include src` ≡ `@include src/BirdCode.md`
    (找到则引入、未找到则由上层报「未找到」放弃)。边界用 resolve() 后的 is_relative_to(root):
    拦 ../../、绝对路径越界、symlink 指向 root 外的逃逸;目录→BirdCode.md 后再 resolve()+复检,
    防 src/BirdCode.md 是指向 root 外的 symlink。
    """
    p = Path(target)
    full = p.resolve() if p.is_absolute() else (base / p).resolve()
    root_resolved = root.resolve()
    if not full.is_relative_to(root_resolved):
        return None
    # 目录 → 自动找目录下的 BirdCode.md(@include src ≡ @include src/BirdCode.md)。
    if full.is_dir():
        full = (full / PROJECT_FILE).resolve()
        if not full.is_relative_to(root_resolved):
            return None
    return full
