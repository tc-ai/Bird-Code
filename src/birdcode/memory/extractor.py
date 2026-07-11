# src/birdcode/memory/extractor.py
"""记忆提取:每轮结束后异步回顾对话,经一次 LLM 调用决定 create/update/delete。

流程:组 prompt(现有记忆清单 + 本轮对话)→ provider.complete(model=extract_model)
→ 防御性解析 JSON 操作列表 → 逐条落盘 → 重建两个目录的 MEMORY.md。

串行(asyncio.Lock,防并发写冲突)+ 全程错误吞(绝不杀主循环)。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ValidationError

from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.memory.paths import project_memory_dir, type_to_dir, user_memory_dir
from birdcode.memory.store import delete_memory, list_memories, rebuild_index, write_memory
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    from birdcode.agent.provider import StreamingProvider
    from birdcode.conversation import Turn

log = get_logger("birdcode.memory.extractor")

_MAX_TURN_CHARS = 8000  # 渲染本轮对话的字符上限(防撑爆提取 prompt)

_EXTRACTION_SYSTEM = """你是 BirdCode 的记忆提取助手。分析最近一轮对话,提取值得长期记忆的信息。

记忆分四类(决定存哪个目录):
- user:用户个人编码偏好/风格(如「不要用 tab」「回复用中文」)→ 用户级,跨项目复用
- feedback:用户对 Agent 输出的纠正(如「函数名改成 ProcessOrder」)→ 用户级
- project:当前项目的技术信息(如「部署脚本在 scripts/」「DB 是 PostgreSQL 15」)→ 项目级
- reference:外部链接/资料(如「API 文档在 https://...」)→ 项目级

操作:
- create:新建记忆(name 用 kebab-case 英文 slug;description 一句话中文;body 详细内容)
- update:更新已有记忆(name 定位、description/body 给新内容,同名覆写);
  **改名**时加 new_name 字段(name=旧名定位、new_name=新名,删旧文件写新文件)
- delete:记忆已过时 → 删除(同时会从 MEMORY.md 移除)
- 没有值得记忆的内容 → 返回 {"operations": []}

规则:
- 已有相同含义的记忆不要重复 create,改用 update。
- 跳过密钥/密码/token/.env 内容/PII,绝不记忆这类敏感信息。
- name 必须是英文 kebab-case(如 prefers-any-over-interface)。

只返回 JSON,不要任何解释:
{"operations": [
  {"action": "create", "type": "feedback", "name": "...", "description": "...", "body": "..."},
  {"action": "update", "type": "project", "name": "...", "new_name": "(可选,仅改名时)",
   "description": "...", "body": "..."},
  {"action": "delete", "name": "..."}
]}
"""


class _MemoryOp(BaseModel):
    action: Literal["create", "update", "delete", "none"]
    type: Literal["user", "feedback", "project", "reference"] | None = None
    name: str = ""
    new_name: str = ""  # update 改名时填:name=旧名定位、new_name=新名
    description: str = ""
    body: str = ""


class MemoryManager:
    """每轮结束后异步提取记忆。串行(Lock)+ 错误吞,绝不杀主循环。

    extract_model 由 run_tui 解析为 profile.hak_model or profile.model(复用同一 provider
    端点/密钥,只换模型名)。/profile 运行时切换不会即时反映到提取(可接受,重启生效)。
    """

    def __init__(
        self, provider: StreamingProvider, *, project_root: Path, extract_model: str
    ) -> None:
        self._provider = provider
        self._project_root = project_root
        self._extract_model = extract_model
        self._lock = asyncio.Lock()

    def set_provider(self, provider: StreamingProvider, *, extract_model: str) -> None:
        """/profile 切换后重连:更新提取用的 provider + 模型。

        与 ContextManager.set_provider 同款。MemoryManager 持的是构造时的 provider,
        不重连则 /profile 后提取仍走旧端点/密钥/模型(旧 key 失效即静默失败)。
        """
        self._provider = provider
        self._extract_model = extract_model

    async def extract(self, turn: Turn, history: list[Turn]) -> None:
        """回顾本轮对话 → LLM 提取 → 落盘 → 重建索引。失败只 log,绝不抛。

        history 暂未使用(保留为 API 占位,将来传最近几轮做上下文)。
        """
        async with self._lock:
            try:
                await self._extract_inner(turn)
            except Exception:  # noqa: BLE001 - 提取失败绝不杀主循环
                log.exception("记忆提取失败")

    async def _extract_inner(self, turn: Turn) -> None:
        user_prompt = self._build_user_prompt(turn)
        raw = await self._provider.complete(
            _EXTRACTION_SYSTEM,
            user_prompt,
            max_tokens=1024,
            model=self._extract_model,
        )
        ops = self._parse(raw)
        for op in ops:
            self._apply(op)
        # 无条件重建两个目录的索引(扫描廉价,免跟踪哪个目录被动)
        rebuild_index(project_memory_dir(self._project_root))
        rebuild_index(user_memory_dir())

    def _build_user_prompt(self, turn: Turn) -> str:
        existing: list[str] = []
        for d in (project_memory_dir(self._project_root), user_memory_dir()):
            for m in list_memories(d):
                existing.append(f"- [{m.type}] {m.name}: {m.description}")
        existing_block = "\n".join(existing) if existing else "(暂无已有记忆)"
        return f"【已有记忆清单】\n{existing_block}\n\n【最近一轮对话】\n{self._render_turn(turn)}"

    def _render_turn(self, turn: Turn) -> str:
        parts: list[str] = []
        total = 0
        # tool_use_id → name:让后面的 ToolResultBlock 带上产生它的工具名
        # (ToolResultBlock 只存 tool_use_id,否则看不出 "PostgreSQL 15.3" 出自 bash 还是 rg)
        tool_names: dict[str, str] = {}
        for msg in turn.messages:
            for b in msg.content:
                if isinstance(b, TextBlock):
                    seg = f"[{msg.role}] {b.text}"
                elif isinstance(b, ToolUseBlock):
                    tool_names[b.id] = b.name
                    preview = json.dumps(b.input, ensure_ascii=False)[:200]
                    seg = f"[{msg.role} tool:{b.name}] {preview}"
                elif isinstance(b, ToolResultBlock):
                    # 工具输出常含最有价值的 project/reference 信号(rg 命中、psql 版本…),
                    # 回查工具名 + 单段截 500 字,再由总 8000 上限兜底。
                    name = tool_names.get(b.tool_use_id, "?")
                    tag = ":error" if getattr(b, "is_error", False) else ""
                    seg = f"[tool_result {name}{tag}] {(b.content or '')[:500]}"
                else:
                    continue
                # 先按剩余预算截断本段再加——防单个大 TextBlock/ToolResult 击穿 8000 上限
                remaining = _MAX_TURN_CHARS - total
                if remaining <= 0:
                    parts.append("…(截断)")
                    return "\n".join(parts)
                if len(seg) > remaining:
                    parts.append(seg[:remaining] + "…(截断)")
                    return "\n".join(parts)
                parts.append(seg)
                total += len(seg)
        return "\n".join(parts) or "(空对话)"

    @staticmethod
    def _parse(raw: str) -> list[_MemoryOp]:
        """防御性解析:剥 ```json 围栏、取首个 { 到末个 }、json.loads、逐条校验。

        单条 op 校验失败只跳过该条(log + continue,不波及其余);整体非 JSON / 非 dict /
        operations 非列表 → 返回 []。绝不抛异常。
        """
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        lo, hi = text.find("{"), text.rfind("}")
        if lo == -1 or hi == -1 or hi <= lo:
            log.warning("记忆提取返回非 JSON,跳过: %s", text[:200])
            return []
        try:
            data = json.loads(text[lo : hi + 1])
        except json.JSONDecodeError:
            log.warning("记忆提取 JSON 解析失败,跳过")
            return []
        raw_ops = data.get("operations", []) if isinstance(data, dict) else []
        if not isinstance(raw_ops, list):
            log.warning("记忆提取 operations 非列表,跳过")
            return []
        ops: list[_MemoryOp] = []
        for raw_op in raw_ops:
            try:
                ops.append(_MemoryOp.model_validate(raw_op))
            except ValidationError:
                log.warning("跳过畸形记忆操作: %s", raw_op)
        return ops

    def _apply(self, op: _MemoryOp) -> None:
        if op.action in ("none", "") or not op.name:
            return
        try:
            if op.action == "delete":
                # delete 可能跨目录:两个目录都试(幂等)
                delete_memory(project_memory_dir(self._project_root), op.name)
                delete_memory(user_memory_dir(), op.name)
                return
            if op.action in ("create", "update"):
                if op.type is None:
                    log.warning("记忆操作缺 type,跳过: %s", op.name)
                    return
                # 只写 type 决定的目录;**不清另一目录**——user/project 是独立命名空间,
                # 同名可合法共存(如个人偏好 user/db-url 与项目事实 project/db-url),
                # 跨目录删会静默销毁合法的跨作用域记忆。同名重复(若 LLM 真的在重分类)
                # 顶多是系统提示词里列两次(有 [type] 前缀可辨),远好过数据丢失。
                dir_ = type_to_dir(op.type, self._project_root)
                if op.action == "update" and op.new_name:
                    # rename:name 定位删旧 slug 文件、new_name 写新 slug 文件(仅本目录)。
                    # name 既是定位符又是文件名,改名必须显式双字段——否则只写新文件、留旧孤儿。
                    delete_memory(dir_, op.name)
                    write_memory(
                        dir_,
                        name=op.new_name,
                        description=op.description,
                        type_=op.type,
                        body=op.body,
                    )
                else:
                    # create / 纯内容更新:同名同 slug → write_memory 覆写(目录内幂等去重)。
                    write_memory(
                        dir_,
                        name=op.name,
                        description=op.description,
                        type_=op.type,
                        body=op.body,
                    )
                return
            log.warning("未知记忆操作 %r,跳过", op.action)
        except Exception:  # noqa: BLE001 - 单条失败不波及其余
            log.exception("应用记忆操作失败: %s", op.action)
