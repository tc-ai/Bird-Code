# tests/agent/__init__.py
"""agent 包测试:让 tests/agent/ 成为 pytest package(镜像 src/birdcode/agent/)。

注意:与 tests/agents/(复数,agent *定义* 系统)不同——本包覆盖 LLM provider /
agent loop 机制层(base_llm / factory / 两 provider 子类)。"""
