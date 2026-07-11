# tests/test_message_queue.py


def test_message_queue_enqueue_dequeue():
    from birdcode.conversation import MessageQueue

    q = MessageQueue()
    assert q.size == 0
    assert q.dequeue_nowait() is None
    q.enqueue("a")
    q.enqueue("b")
    assert q.size == 2
    assert q.dequeue_nowait() == "a"
    assert q.dequeue_nowait() == "b"
    assert q.size == 0


def test_turn_defaults():
    from birdcode.blocks import TextBlock
    from birdcode.conversation import Message, Turn

    t = Turn()
    assert t.messages == [] and t.usage is None and t.interrupted is False
    assert t.failed is False  # #2:wake 轮 provider 异常时置 True(默认 False)
    t2 = Turn(messages=[Message(role="user", content=[TextBlock("hi")])])
    assert t2.messages[0].role == "user"


def test_message_queue_size_excludes_wake_markers():
    """#11:状态栏「排队消息」计数只应含用户文本(str),不含系统唤醒标记(_WakeInput)。

    否则忙时若干异步子 agent 完成会把 _WAKE 计入 ⌛N,误显为「用户在排队输入」。
    """
    from birdcode.conversation import _WAKE, MessageQueue

    q = MessageQueue()
    q.enqueue("a")
    q.enqueue(_WAKE)
    assert q.size == 1  # _WAKE 不计入
    q.enqueue(_WAKE)
    assert q.size == 1  # 多个 _WAKE 仍不计入
    assert q.dequeue_nowait() == "a"
    assert q.size == 0  # 只剩 _WAKE
    assert q.dequeue_nowait() is _WAKE
    assert q.size == 0
