"""主对话/子 agent 文本面板:贴底跟随输出,用户上滑读历史时不被拉回。

(替代旧实现里 watch_busy 起的 80ms 无条件 _pin_to_bottom 定时器——它在同步子 agent
运行期间持续把用户上滑拉回底部。)

user_following:是否跟随新内容钉底。默认 True。
- 我们自己的钉底(scroll_end)只会增大或不变 scroll_y → 不触发「用户上滑」。
- 用户上滑(鼠标轮 / 滚动条拖动 / 键盘)→ scroll_y 减小 → user_following=False。
- 用户滚回底部 → scroll_y 到 max_scroll_y → user_following=True(恢复跟随)。
内容增长只改 virtual_size、不改 scroll_y → watch 不触发 → user_following 稳定;
此时由低频定时器在 user_following 时 scroll_end 追平(walk-down)。

不使用 Textual 的 anchor():它在内容未溢出时会算出负的 max_scroll_y,把 scroll_y
钉成负值(实测 -11)。本类不动 anchor,保留 scroll_end「无溢出→scroll_y=0」的原行为。
"""

from __future__ import annotations

from textual.containers import VerticalScroll


class TranscriptScroll(VerticalScroll):
    """带「用户上滑即停跟随」语义的滚动面板。"""

    # 是否跟随新内容钉底(True=跟随输出,False=用户在读历史,别打扰)。
    user_following: bool = True

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        # 滚回底部 → 恢复跟随(优先判定,覆盖刚追平到底的 tick)。
        if new_value >= self.max_scroll_y - 1:
            self.user_following = True
        # scroll_y 减小 = 用户主动上滑(我们的 scroll_end 只增不减)。
        elif new_value < old_value - 0.5:
            self.user_following = False
