"""
EpisodeBuffer — Episode 数据缓存，用于 REINFORCE 批量更新。

REINFORCE 是 on-policy 算法，每次更新后清空 buffer。
Buffer 存储完整 episode（不是单步 transition），
因为 REINFORCE 需要整条轨迹的折扣回报。
"""

from __future__ import annotations

from collections import deque
from typing import Optional
import numpy as np

from shopping_agent.rl.pomdp import Episode


class EpisodeBuffer:
    """
    固定容量的 episode 缓存。

    用法：
        buffer = EpisodeBuffer(capacity=256)
        buffer.add(episode)
        batch = buffer.sample(batch_size=32)
        buffer.clear()  # on-policy: 每次更新后清空
    """

    def __init__(self, capacity: int = 512):
        self._buffer: deque[Episode] = deque(maxlen=capacity)

    def add(self, episode: Episode) -> None:
        self._buffer.append(episode)

    def sample(self, batch_size: Optional[int] = None) -> list[Episode]:
        if batch_size is None or batch_size >= len(self._buffer):
            return list(self._buffer)
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        return [list(self._buffer)[i] for i in indices]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_episodes: int = 16) -> bool:
        return len(self._buffer) >= min_episodes

    def stats(self) -> dict:
        if not self._buffer:
            return {}
        success_rates = [float(ep.final_task_success) for ep in self._buffer]
        plan_scores = [ep.final_plan_score for ep in self._buffer]
        turns = [ep.total_turns for ep in self._buffer]
        return {
            "num_episodes": len(self._buffer),
            "success_rate": float(np.mean(success_rates)),
            "avg_plan_score": float(np.mean(plan_scores)),
            "avg_turns": float(np.mean(turns)),
            "min_turns": int(np.min(turns)) if turns else 0,
            "max_turns": int(np.max(turns)) if turns else 0,
        }
