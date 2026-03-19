"""
PreferenceMemory — 用户偏好记忆层。

融合三类记忆：
  1. 短期工作记忆（当前会话状态）
  2. 长期用户画像（历史行为积累）
  3. 隐式行为偏好（点击、停留、跳过信号）

生产环境应接入 Redis（短期）+ 数据库（长期画像）。
当前实现为内存 stub，接口与生产版本保持一致。
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Optional

from shopping_agent.common.types import ShoppingTask, UserProfile
from shopping_agent.storage.profile_store import ProfileStore


class PreferenceMemory:
    def __init__(self, profile_store: Optional[ProfileStore] = None):
        self._store = profile_store or ProfileStore()
        self._implicit_signals: dict[str, list[dict]] = {}

    def load(self, user_id: str,
             task: Optional[ShoppingTask] = None) -> UserProfile:
        """
        加载用户画像。
        若用户无历史记录，返回空白画像（冷启动）。
        """
        profile = self._store.load(user_id)

        # 根据当前任务场景做上下文感知的偏好加权
        if task:
            profile = self._contextualize(profile, task)

        return profile

    def save(self, profile: UserProfile) -> None:
        """持久化用户画像。"""
        self._store.save(profile)

    def update_explicit(
        self,
        user_id: str,
        signal_type: str,       # "purchase", "reject", "add_to_cart"
        product_attrs: dict,    # {"brand": "Sony", "price": 2999, "category": "laptop"}
        delta: float = 0.1,
    ) -> None:
        """
        根据显式行为更新用户画像。
        signal_type="purchase"/"add_to_cart" → 正向更新
        signal_type="reject" → 负向更新
        """
        profile = self.load(user_id)
        direction = 1.0 if signal_type in ("purchase", "add_to_cart") else -1.0
        self._increment_signal_count(profile, signal_type)

        # 更新品牌权重
        brand = product_attrs.get("brand")
        if brand:
            current = profile.brand_weights.get(brand, 0.5)
            profile.brand_weights[brand] = max(0.0, min(1.0,
                                               current + direction * delta))

        # 更新品类亲和度
        category = product_attrs.get("category")
        if category and direction > 0:
            profile.category_affinity[category] = (
                profile.category_affinity.get(category, 0.0) + delta
            )

        # 更新价格敏感度（拒绝高价商品 → 提高敏感度）
        price = product_attrs.get("price")
        if price and signal_type == "reject":
            # 简单启发：拒绝高于均值的商品说明价格敏感
            profile.price_sensitivity = min(1.0,
                                            profile.price_sensitivity + 0.05)
        if price and signal_type in ("purchase", "add_to_cart"):
            profile.budget_anchor_history.append(float(price))
            profile.budget_anchor_history = profile.budget_anchor_history[-10:]

        profile.last_updated = datetime.now()
        self._store.log_interaction_signal(
            user_id=user_id,
            signal_type=signal_type,
            product_attrs=product_attrs,
        )
        self.save(profile)

    def update_implicit(
        self,
        user_id: str,
        signal_type: str,       # "dwell", "skip"
        product_attrs: dict,
        dwell_seconds: float = 0.0,
    ) -> None:
        """
        记录隐式行为信号，用于软更新画像。
        """
        signal = {
            "signal_type": signal_type,
            "product_attrs": product_attrs,
            "dwell_seconds": dwell_seconds,
            "timestamp": datetime.now().isoformat(),
        }
        signals = self._implicit_signals.setdefault(user_id, [])
        signals.append(signal)
        self._store.log_interaction_signal(
            user_id=user_id,
            signal_type=signal_type,
            product_attrs=product_attrs,
            dwell_seconds=dwell_seconds,
        )

        # 简单规则：停留超过 30 秒视为正向隐式信号
        if signal_type == "dwell" and dwell_seconds > 30:
            self.update_explicit(user_id, "add_to_cart", product_attrs, delta=0.05)

    def get_recent_signals(self, user_id: str, limit: int = 20) -> list[dict]:
        """返回用户近期交互信号，优先读取持久化记录。"""
        persisted = self._store.list_interaction_signals(user_id, limit=limit)
        if persisted:
            return persisted
        return list(reversed(self._implicit_signals.get(user_id, [])))[-limit:]

    def _contextualize(self, profile: UserProfile, task: ShoppingTask) -> UserProfile:
        """
        根据当前任务的隐式需求，临时调整画像权重。
        不修改原始画像，返回调整副本。
        """
        ctx_profile = copy.deepcopy(profile)

        # 出差场景：提高履约时效偏好
        if any("出差" in need or "商务" in need for need in task.implicit_needs):
            ctx_profile.prefer_fast_delivery = True

        # 有硬性预算约束时，临时提高价格敏感度
        budget = task.get_hard_constraints().get("budget_total")
        if budget and budget < 1000:
            ctx_profile.price_sensitivity = min(1.0,
                                                ctx_profile.price_sensitivity + 0.2)

        # 近期正向信号会轻微提升相关品牌/品类偏好
        for signal in self.get_recent_signals(profile.user_id, limit=10):
            signal_type = signal.get("signal_type", "")
            attrs = signal.get("product_attrs", {})
            if signal_type not in {"purchase", "add_to_cart", "dwell"}:
                continue

            brand = attrs.get("brand")
            category = attrs.get("category")
            if brand and (not task.categories or category in task.categories):
                current = ctx_profile.brand_weights.get(brand, 0.5)
                ctx_profile.brand_weights[brand] = min(1.0, current + 0.03)
            if category and (not task.categories or category in task.categories):
                ctx_profile.category_affinity[category] = (
                    ctx_profile.category_affinity.get(category, 0.0) + 0.05
                )

        return ctx_profile

    @staticmethod
    def _increment_signal_count(profile: UserProfile, signal_type: str) -> None:
        profile.interaction_signal_counts[signal_type] = (
            profile.interaction_signal_counts.get(signal_type, 0) + 1
        )
