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
    def __init__(self, memory: PreferenceMemory | None = None):
        self._memory = memory or PreferenceMemory()

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
            before = {
                "price_sensitivity": profile.price_sensitivity,
                "budget_sensitivity_profile": dict(profile.budget_sensitivity_profile),
            }
            profile.price_sensitivity = max(0.0, profile.price_sensitivity - 0.05)
            current = profile.budget_sensitivity_profile.get("premium", 0.0)
            profile.budget_sensitivity_profile["premium"] = min(1.0, current + 0.1)
            strict_current = profile.budget_sensitivity_profile.get("strict", 0.0)
            profile.budget_sensitivity_profile["strict"] = max(0.0, strict_current - 0.08)
            self._memory.register_persona_drift(
                user_id=user_id,
                drift_type="budget_drift",
                evidence=f"预算从{old_budget}调整到{new_budget}",
                magnitude=min(1.0, (new_budget - old_budget) / max(old_budget, 1)),
                before=before,
                after={
                    "price_sensitivity": profile.price_sensitivity,
                    "budget_sensitivity_profile": dict(profile.budget_sensitivity_profile),
                    "target": "premium",
                },
                profile=profile,
            )

        old_style = context.get("old_style")
        new_style = context.get("new_style")
        if new_style and new_style != old_style:
            current = profile.aesthetic_preference.get(new_style, 0.0)
            profile.aesthetic_preference[new_style] = min(1.0, current + 0.12)
            self._memory.register_persona_drift(
                user_id=user_id,
                drift_type="style_drift",
                evidence=f"风格偏好从{old_style or '未指定'}转向{new_style}",
                magnitude=0.35,
                before={"aesthetic_preference": dict(profile.aesthetic_preference)},
                after={"aesthetic_preference": dict(profile.aesthetic_preference), "target": new_style},
                profile=profile,
            )

        old_identity = context.get("old_identity_goal")
        new_identity = context.get("new_identity_goal")
        if new_identity and new_identity != old_identity:
            current = profile.identity_goal.get(new_identity, 0.0)
            profile.identity_goal[new_identity] = min(1.0, current + 0.12)
            self._memory.register_persona_drift(
                user_id=user_id,
                drift_type="identity_drift",
                evidence=f"身份表达从{old_identity or '未指定'}转向{new_identity}",
                magnitude=0.4,
                before={"identity_goal": dict(profile.identity_goal)},
                after={"identity_goal": dict(profile.identity_goal), "target": new_identity},
                profile=profile,
            )

        old_brand = context.get("old_brand_orientation")
        new_brand = context.get("new_brand_orientation")
        if new_brand and new_brand != old_brand:
            current = profile.brand_orientation.get(new_brand, 0.0)
            profile.brand_orientation[new_brand] = min(1.0, current + 0.12)
            self._memory.register_persona_drift(
                user_id=user_id,
                drift_type="brand_drift",
                evidence=f"品牌取向从{old_brand or '未指定'}转向{new_brand}",
                magnitude=0.3,
                before={"brand_orientation": dict(profile.brand_orientation)},
                after={"brand_orientation": dict(profile.brand_orientation), "target": new_brand},
                profile=profile,
            )

        self._memory.save(profile)
