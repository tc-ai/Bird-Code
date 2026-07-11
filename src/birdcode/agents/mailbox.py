# src/birdcode/agents/mailbox.py
"""agent teams:MailboxMessage(agent 间私信载体)。

mailbox(asyncio.Queue)的 item = MailboxMessage | _TEAMMATE_SHUTDOWN(见 runner.py)。
recipient(teammate/lead)读到后,turn 以"[来自 sender]: content"呈现,让收方知道谁发的。
仅 type='text';shutdown_request/response 等后续加。Mailbox/MailboxQueue 封装也留后续。
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class MailboxMessage:
    """agent 间私信。sender/to 为 name(lead 或 teammate 名)。"""

    sender: str
    to: str
    content: str
    summary: str = ""  # 一行摘要(SendMessage 可强制要;recipient 一眼可见)
    type: str = "text"  # text|shutdown_request|shutdown_response(目前仅 text)


# 当前执行 agent 的 name(SendMessage 据此取 sender)。default="lead"(主 session 不设);
# 每个 teammate task 经 _run_named 设自己的名(asyncio task 隔离 context,互不污染)。
_AGENT_NAME: ContextVar[str] = ContextVar("_AGENT_NAME", default="lead")
