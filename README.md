# Bird-Code

> 一个会写代码的终端 AI 编程助手 🐦

Bird-Code 是一个运行在终端里的 AI 编码 agent：你与 LLM 对话，LLM 通过 agentic loop 在本地自主读取、编写、执行代码。灵感来源于 Claude Code，用 Python + Textual 构建 TUI，采用成熟的 Python 生态实现，不引入臃肿的全家桶框架。

## 功能特性

- **Agentic loop** — 清晰的 `stream → tool_use → execute → tool_result → 续 stream` 循环，带护栏（最大轮次 / 连续失败熔断 / 上下文超限 reactive 重试）
- **多提供商** — Anthropic、OpenAI 均可接入，profile 热切换，Anthropic 下自动启用 prompt caching
- **工具系统** — 内置 Read / Glob / Grep / Write / Edit / Delete / Bash 等，Pydantic 定义参数 schema；只读工具并行、写工具串行；三轨输出分离（UI / LLM / 落盘各自阈值）
- **分层权限** — L1–L5（黑名单 → 沙箱路径 → 规则 → 模式 → 人在回路），四种模式 `default` / `accept-edits` / `plan` / `bypass`
- **终端图形界面** — 基于 Textual 的 TUI，流式 Markdown、工具行、权限提示、固定底部输入框（CJK IME 友好）
- **子 Agent** — 内置 Explore（只读探索）、Plan（架构规划）、General-purpose（通用开发），支持 `worktree` 隔离（子 agent 在独立 git worktree 工作，产物清单随报告返回）
- **技能系统** — `.birdcode/skill/` 下的 markdown 技能（透明注入），支持 inline 注入当前对话与 fork 子 agent 两种模式
- **会话管理** — jsonl 持久化，中断恢复、跨会话 replay
- **上下文压缩** — 逼近窗口时自动摘要压缩，413 硬截断兜底
- **记忆系统** — 自动提取用户 / 项目记忆，跨会话注入
- **MCP 协议** — 接入 stdio / streamable_http 的 MCP 工具服务器
- **Git Worktree** — 集成 worktree 管理，多终端并行开发

## 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- [ripgrep](https://github.com/BurntSushi/ripgrep)（Grep / Glob 工具用）
- 至少一个 LLM API Key（Anthropic 或 OpenAI）

## 快速开始

```bash
git clone https://github.com/<owner>/Bird-Code.git Bird-Code
cd Bird-Code
uv sync
uv run birdcode
```

## 配置

Bird-Code 通过 YAML 文件配置，加载优先级从低到高（后者 deep-merge 覆盖前者）：

1. `~/.birdcode/config.yaml` — 用户级全局配置
2. `.birdcode/config.yaml` — 项目级配置

> 配置文件含 API key，已默认在 `.gitignore` 中忽略，不会提交。敏感值建议用 `${VAR}` 引用环境变量，避免明文落盘。

### API Key

`api_key` 支持 `${ENV_VAR}` 语法引用环境变量。此外，环境变量会覆盖对应 protocol profile 的 api_key：

| 协议 | 环境变量 |
| --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |

> 同一 protocol 的所有 profile 共享同一个 env-var（如两个 openai profile 都读 `OPENAI_API_KEY`）；需要不同 key 时请在 YAML 里直接填写。

### 最简配置

```yaml
default: claude
providers:
  claude:
    protocol: anthropic
    base_url: https://api.anthropic.com
    model: claude-sonnet-5
    api_key: ${ANTHROPIC_API_KEY}
```

### 完整配置示例

```yaml
default: claude
max_tokens: 8192
context_window: 200000            # 模型上下文窗口(token)

providers:
  claude:
    protocol: anthropic
    base_url: https://api.anthropic.com
    model: claude-sonnet-5
    api_key: ${ANTHROPIC_API_KEY}
    thinking:                     # Anthropic extended thinking(可选)
      budget_tokens: 4096
    hak_model: claude-haiku-4-5   # 提取记忆用的模型(可选,复用同端点/密钥)

  gpt:
    protocol: openai
    base_url: https://api.openai.com/v1
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}
    reasoning_effort: medium      # OpenAI 推理强度:minimal / low / medium / high

# MCP 工具服务器(可选)
mcp_servers:
  filesystem:
    type: stdio
    command: npx
    args: ["-y", "@anthropic-ai/mcp-server-filesystem", "/path/to/allowed"]
  remote:
    type: streamable_http
    url: https://example.com/mcp
    headers:
      Authorization: Bearer ${MCP_TOKEN}

# 沙箱额外允许目录(L2,可选;相对路径按启动 CWD 解析)
extra_roots: []
```

## 自定义 agent 与 skill

Bird-Code 支持用 markdown 文件自定义**子 agent**和 **skill**,放在 `.birdcode/` 下。两者都有用户级(`~/.birdcode/`)和项目级(`<项目>/.birdcode/`),项目级覆盖用户级,同名还可覆盖内置。

### 子 agent

在 `~/.birdcode/agents/` 或 `<项目>/.birdcode/agents/` 下放 `*.md`,每个文件是一个子 agent(派生独立上下文执行):

~~~markdown
---
name: translator
description: 把给定文本翻译成目标语言,只翻译不解释。
disallowed_tools: [bash, write_file, edit_file]   # 可选,工具黑名单
model: gpt                                          # 可选,profile 名(省略=继承父)
kind: read                                          # 可选,read / write(默认 write)
parallel_safe: true                                 # 可选(默认 false)
run_in_background: false                            # 可选,异步后台(默认 false)
---

你是一个专业翻译。把用户提供的文本翻译成目标语言,只输出译文,不解释。
~~~

`name` + `description` 必填;正文是 system prompt(persona)。内置 explore / plan / general-purpose 同样可被同名的项目级 `.md` 覆盖。

### Skill

在 `~/.birdcode/skill/` 或 `<项目>/.birdcode/skill/` 下放 markdown,支持两种形式:

- **文件型**:`foo.md`
- **目录型**:`foo/SKILL.md`(目录内其他文件由模型按需读取)

~~~markdown
---
name: commit
description: 按规范生成 commit message。
mode: inline            # 可选,inline(默认,正文注入当前对话) / fork(起子 agent)
---

把以下改动写成符合规范的 commit message:

$ARGUMENTS
~~~

`name` + `description` 必填;正文是任务模板,`$ARGUMENTS` 会被调用参数替换(无占位符时参数追加到正文末尾)。skill 是"透明"的——inline 模式正文直接注入当前对话,fork 模式起独立子 agent。

> skill 是项目级 / 用户级运行时配置(随用随加),默认不随仓库分发;按上面格式在 `.birdcode/skill/` 下自建即可。

### 覆盖与冲突

- 同名时:**项目级 > 用户级 > 内置**。
- agent 与 skill **跨目录同名会冲突**(注册报错),需改名。

## 使用

```bash
# 交互模式(TUI)
uv run birdcode

# 在 git worktree 里启动(多终端并行开发)
uv run birdcode --worktree <name>

# 接续本项目最近一次会话
uv run birdcode --continue

# 指定 profile 与主题启动
uv run birdcode --profile gpt --theme light
```

### 命令行参数

| 参数 | 简写 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--profile <name>` | `-p` | 配置里的 `default` | 使用的 provider profile;留空走配置 `default`,`mock` 为离线回退(无需 API key) |
| `--theme <name>` | `-t` | `dark` | TUI 主题:`dark` / `light` |
| `--config <path>` | — | 见「配置」加载顺序 | 指定配置文件路径,覆盖默认查找 |
| `--continue` | `-c` | 关 | 接续本项目最近一次会话(按 mtime) |
| `--resume <id>` | `-r` | — | 接续指定 sessionId(用 `/sessions` 查可用 id;与 `-c` 互斥) |
| `--worktree <name>` | `-w` | — | 在隔离的 git worktree 里起会话(并行开发;退出时清理) |
| `--delay <sec>` | — | `0.012` | `mock` profile 的 token 流式间隔(秒) |

> `-c` 与 `-r` 互斥,同时给会报错;`--resume <id>` 仅允许字母数字与连字符,且须对应已存在会话。worktree 会话的存储与主仓隔离——`birdcode -c`(在主仓)只找主仓会话,`birdcode -w <name> -c` 只找该 worktree 的会话。

### 会话可视化

把会话 jsonl 渲染成交互式 HTML 执行流程树（消息卡片 + 时间线 + 子 agent 分支），便于回看与调试：

```bash
# 按 sessionId 或 jsonl 路径渲染（默认输出 <stem>.html）
uv run birdcode session viz <sessionId>
uv run birdcode session viz path/to/session.jsonl

# 指定输出路径 + 生成后用浏览器打开
uv run birdcode session viz <sessionId> -o report.html --open
```

启动后在输入框输入 `/` 查看 slash 命令（`/help`、`/compact`、`/clear`、`/permissions`、`/sessions` 等），`Tab` 补全命令与文件路径。

## 项目结构

```
src/birdcode/
├── cli/          # Typer 入口、参数解析
├── agent/        # 核心 agent loop、provider 抽象、上下文管理、system prompt
├── agents/       # 子 agent 运行时（runner / manager / registry）
├── tools/        # 工具定义与执行器（Pydantic schema + executor）
├── ui/           # Textual App、widgets、命令系统
├── session/      # jsonl 会话存储、codec、恢复
├── permission/   # 分层权限 gate、沙箱
├── memory/       # 自动记忆提取
├── mcp/          # MCP client
├── config/       # provider profile 配置
└── utils/        # git worktree、logging 等
```

核心循环：用户输入 → `run_agent_loop`（`agent/`）→ provider stream → 工具执行（`tools/executor`）→ 结果回填 → 续 stream，直到无 tool_use。

## 技术栈

- **TUI** — [Textual](https://textual.textualize.io/)（复用 Rich 渲染管线）
- **CLI** — [Typer](https://typer.tiangolo.com/)
- **数据校验** — [Pydantic](https://docs.pydantic.dev/) v2
- **LLM SDK** — [anthropic](https://github.com/anthropics/anthropic-sdk-python) / [openai](https://github.com/openai/openai-python)
- **搜索** — [ripgrep](https://github.com/BurntSushi/ripgrep)
- **质量** — ruff + mypy strict + pytest

## License

MIT — 见 [LICENSE](LICENSE)。
