"""
BundlePlanner — 组合级规划器的一等入口。

保留 ConstraintAwarePlanner 作为兼容层，同时提供更明确的 bundle-native
命名边界，便于后续围绕 BundlePlan / long-horizon planning 继续收束。
"""

from __future__ import annotations

from shopping_agent.planning.planner import ConstraintAwarePlanner


class BundlePlanner(ConstraintAwarePlanner):
    """面向组合级购物与分阶段升级的规划器。"""

