"""配置 schema：provider profile + 全局选项（Pydantic v2）。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ThinkingConfig(BaseModel):
    """Anthropic extended thinking 配置（仅 protocol=anthropic 生效）。"""

    budget_tokens: int = Field(..., ge=1024, description="须 ≥1024 且 < max_tokens")


class ProviderProfile(BaseModel):
    """单个后端 profile。name 由 loader 从 dict key 注入。"""

    name: str = ""
    protocol: Literal["anthropic", "openai"]
    model: str
    base_url: str
    api_key: str
    thinking: ThinkingConfig | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    # 提取记忆用的模型(未指定则复用 model)。复用同一 provider 端点/密钥,只换模型名。
    hak_model: str | None = None


class McpStdioServer(BaseModel):
    """MCP server —— 本地子进程(stdio 管道)。env 合并进 os.environ 传给子进程。"""

    type: Literal["stdio"]
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class McpHttpServer(BaseModel):
    """MCP server —— 远程 Streamable HTTP。"""

    type: Literal["streamable_http"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


# 判别联合:type 字段决定具体类型。
McpServerConfig = Annotated[McpStdioServer | McpHttpServer, Field(discriminator="type")]


class AppConfig(BaseModel):
    """顶层配置。"""

    providers: dict[str, ProviderProfile]
    default: str | None = None
    system_prompt: str = ""  # 空=模块拼装+BirdCode.md;非空=整段 verbatim 覆盖(旁路 BirdCode.md)
    # BirdCode.md(<project_root>/BirdCode.md + ~/.birdcode/BirdCode.md)的拼装结果,启动时由
    # run_tui 填充(非 YAML)。空=无项目指令。provider 经 self._app.project_instructions 读取。
    project_instructions: str = ""
    # 项目根(运行期由 run_tui 填充,非 YAML)。记忆索引每轮从该目录现读注入 <system-reminder>。
    project_root: Path | None = None
    max_tokens: int = 8192
    # 上下文管理(Phase 1 安全网)。阈值派生见 autocompact_threshold。
    context_window: int = Field(200_000, ge=10_000, description="模型上下文窗口(token)")
    compact_summary_reserve: int = Field(20_000, ge=1024, description="预留给摘要输出的 token")
    compact_safety_margin: int = Field(13_000, ge=0, description="防 token 估算抖动的安全余量")
    compact_tail_budget: int = Field(30_000, ge=1024, description="压缩后原样保留尾段的 token 预算")
    extra_roots: list[Path] = Field(
        default_factory=list,
        description="沙箱额外允许目录(L2);相对路径按启动 CWD 解析。空=仅允许 project_root",
    )
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)

    @property
    def autocompact_threshold(self) -> int:
        """自动压缩触发阈值 = 窗口 − 摘要预留 − 安全余量(默认 167_000)。"""
        return self.context_window - self.compact_summary_reserve - self.compact_safety_margin
