"""
ClarificationPolicy — 自适应澄清策略。

核心原则：只问最有价值的问题。
  clarification_value(slot) = info_gain(slot) × impact_on_plan(slot) / fatigue_cost(round)
"""

from __future__ import annotations

from typing import Optional

from shopping_agent.common.constants import (
    CLARIFICATION_MAX_OPTIONS,
    MIN_INFO_GAIN_TO_ASK,
)
from shopping_agent.common.types import (
    ClarificationQuestion,
    ClarificationStyle,
    ShoppingTask,
    UserProfile,
)


# 各槽位的预设信息增益与方案影响分（可通过学习更新）
_SLOT_CONFIG: dict[str, dict] = {
    "budget_total": {
        "info_gain": 0.9,
        "impact": 1.0,
        "question_template": "您的总预算大概是多少？",
        "style": ClarificationStyle.OPEN_ENDED,
        "options": [],
    },
    "delivery_days": {
        "info_gain": 0.6,
        "impact": 0.7,
        "question_template": "您需要几天内收到商品？",
        "style": ClarificationStyle.CHOICE,
        "options": ["次日达", "3天内", "一周内都可以"],
    },
    "usage_scenario": {
        "info_gain": 0.8,
        "impact": 0.9,
        "question_template": "主要是什么场景使用？",
        "style": ClarificationStyle.CHOICE,
        "options": ["居家", "出差商务", "户外"],
    },
    "brand_preference": {
        "info_gain": 0.5,
        "impact": 0.6,
        "question_template": "有品牌偏好吗？",
        "style": ClarificationStyle.CHOICE,
        "options": ["不限", "只要国产品牌", "偏向国际品牌"],
    },
    "color": {
        "info_gain": 0.2,
        "impact": 0.2,
        "question_template": "颜色有要求吗？",
        "style": ClarificationStyle.OPEN_ENDED,
        "options": [],
    },
    "size": {
        "info_gain": 0.7,
        "impact": 0.8,
        "question_template": "您的尺寸是多少？（如衣服尺码/鞋码）",
        "style": ClarificationStyle.OPEN_ENDED,
        "options": [],
    },
}


class ClarificationPolicy:
    def __init__(self):
        # 可通过学习层更新的权重
        self._slot_weights: dict[str, float] = {}

    def generate(
        self,
        task: ShoppingTask,
        user_profile: Optional[UserProfile] = None,
    ) -> list[ClarificationQuestion]:
        """
        根据任务不确定性槽位，生成待问问题列表（按价值降序排列）。
        返回空列表表示无需澄清。
        """
        questions = []

        for slot, value in task.uncertainty_slots.items():
            if value is not None and value != "uncertain":
                continue  # 已知槽位跳过

            # 从用户画像填充（如果已知，不用问）
            if user_profile and self._can_fill_from_profile(slot, user_profile):
                continue

            question = self._build_question(slot, task)
            if question and question.info_gain >= MIN_INFO_GAIN_TO_ASK:
                questions.append(question)

        # 按 clarification_value 降序
        questions.sort(key=lambda q: self._score(q, task), reverse=True)
        return questions

    def _score(self, q: ClarificationQuestion, task: ShoppingTask) -> float:
        fatigue_cost = 1.0 + task.uncertainty_score * 0.5
        return (q.info_gain * q.impact_score) / fatigue_cost

    def _build_question(self, slot: str, task: ShoppingTask) -> Optional[ClarificationQuestion]:
        config = _SLOT_CONFIG.get(slot)
        if not config:
            # 未知槽位：生成通用开放式问题
            return ClarificationQuestion(
                slot=slot,
                question=f"关于「{slot}」，您有什么具体要求吗？",
                style=ClarificationStyle.OPEN_ENDED,
                info_gain=0.3,
                impact_score=0.3,
            )

        options = config["options"][:CLARIFICATION_MAX_OPTIONS]
        return ClarificationQuestion(
            slot=slot,
            question=config["question_template"],
            style=config["style"],
            options=options,
            info_gain=config["info_gain"],
            impact_score=config["impact"],
        )

    def _can_fill_from_profile(self, slot: str, profile: UserProfile) -> bool:
        if slot == "size" and profile.size_profile:
            return True
        if slot == "brand_preference" and profile.brand_weights:
            return True
        return False

    def update_weights(self, slot: str, new_weight: float) -> None:
        """由学习层调用，更新槽位权重。"""
        self._slot_weights[slot] = new_weight
