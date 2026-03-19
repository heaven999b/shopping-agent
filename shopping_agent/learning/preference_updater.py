"""
PreferenceUpdater — 偏好更新器。

根据反馈信号更新用户画像。
区分七类信号，映射到不同的更新逻辑。

信号 → 更新目标：
  EXPLICIT_POSITIVE  → 品牌权重+、品类亲和度+
  EXPLICIT_NEGATIVE  → 品牌权重-、价格敏感度（如果因价格拒绝）
  IMPLICIT_POSITIVE  → 软更新（delta 较小）
  IMPLICIT_NEGATIVE  → 软降分
  REVISION          → 更新预算分布估计
  TASK_COMPLETE     → 记录成功 pattern
  TASK_ABANDON      → 记录失败场景
"""

from __future__ import annotations

from shopping_agent.common.types import FeedbackRecord, FeedbackSignal
from shopping_agent.memory.preference_memory import PreferenceMemory


class PreferenceUpdater:
    def __init__(self):
        self._memory = PreferenceMemory()

    def update(self, user_id: str, record: FeedbackRecord) -> None:
        """根据反馈信号更新用户画像。"""
        signal = record.signal
        context = record.context

        product_attrs = context.get("product_attrs", {})

        if signal == FeedbackSignal.EXPLICIT_POSITIVE:
            self._memory.update_explicit(
                user_id, "purchase", product_attrs, delta=0.15
            )
        elif signal == FeedbackSignal.EXPLICIT_NEGATIVE:
            self._memory.update_explicit(
                user_id, "reject", product_attrs, delta=0.1
            )
        elif signal == FeedbackSignal.IMPLICIT_POSITIVE:
            dwell = context.get("dwell_seconds", 0)
            self._memory.update_implicit(
                user_id, "dwell", product_attrs, dwell_seconds=dwell
            )
        elif signal == FeedbackSignal.IMPLICIT_NEGATIVE:
            self._memory.update_implicit(
                user_id, "skip", product_attrs
            )
        elif signal == FeedbackSignal.REVISION:
            # 用户主动修改约束：记录新的预算/偏好范围
            self._handle_revision(user_id, context)
        elif signal == FeedbackSignal.TASK_COMPLETE:
            # 成功下单：强正向信号
            self._memory.update_explicit(
                user_id, "purchase", product_attrs, delta=0.2
            )
        elif signal == FeedbackSignal.TASK_ABANDON:
            # 任务放弃：记录失败场景，暂不更新画像（等归因分析后再更新）
            pass

    def _handle_revision(self, user_id: str, context: dict) -> None:
        """处理用户主动修改约束的信号。"""
        profile = self._memory.load(user_id)

        # 如果用户提高了预算，说明价格敏感度较低
        old_budget = context.get("old_budget")
        new_budget = context.get("new_budget")
        if old_budget and new_budget and new_budget > old_budget:
            profile.price_sensitivity = max(0.0, profile.price_sensitivity - 0.05)

        self._memory._profiles[user_id] = profile
