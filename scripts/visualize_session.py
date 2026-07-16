#!/usr/bin/env python3
"""薄包装:命令行直接调用(等价 `birdcode session viz`)。逻辑在 birdcode.session.viz。"""

from __future__ import annotations

import argparse
from pathlib import Path

from birdcode.session.viz import render_session_html, resolve_session_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="把 BirdCode 会话 jsonl 渲染成交互式 HTML 执行流程树")
    ap.add_argument("session", help="会话 jsonl 路径 或 sessionId")
    ap.add_argument(
        "-o", "--output", type=Path, default=None, help="输出 HTML 路径(默认 <stem>.html)"
    )
    args = ap.parse_args()

    jsonl = resolve_session_jsonl(args.session)
    out = args.output or jsonl.with_suffix(".html")
    out.write_text(render_session_html(jsonl), encoding="utf-8")
    # ASCII 消息:Windows GBK 终端无法编码 emoji(HTML 本身 UTF-8 不受影响)。
    print(f"[OK] generated {out}")
    print(f"     session: {jsonl}")
    print(f"     open in browser (self-contained, no external deps). "
          f"or: `uv run birdcode session viz {args.session}`")


if __name__ == "__main__":
    main()
