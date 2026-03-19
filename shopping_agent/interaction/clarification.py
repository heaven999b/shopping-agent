"""
ClarificationPolicy — 自适应澄清策略。

支持两种模式：
  - 启发式模式（默认，可作为 baseline）：
      clarification_value(slot) = info_gain(slot) × impact_on_plan(slot) / fatigue_cost(round)
  - RL 模式（Option A）：
      π_θ(a|s) = softmax(W_θ · φ(s))，由 REINFORCE 训练得到的策略决策

通过 set_rl_mode(enabled) 切换，便于消融实验。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from shopping_agent.common.constants import (
    CLARIFICATION_MAX_OPTIONS,
    CLARIFICATION_MAX_ROUNDS,
    MIN_INFO_GAIN_TO_ASK,
)
from shopping_agent.common.types import (
    ClarificationQuestion,
    ClarificationStyle,
    ShoppingTask,
    UserProfile,
)
from shopping_agent.rl.pomdp import ClarificationAction, ClarificationActionType, ClarificationState
from shopping_agent.rl.policy import RLClarificationPolicy


# 各槽位的预设信息增益与方案影响分（启发式模式使用；RL 模式用于构建初始状态）
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
    "platform": {
        "info_gain": 0.4,
        "impact": 0.5,
        "question_template": "您偏向在哪个平台购买？",
        "style": ClarificationStyle.CHOICE,
        "options": ["京东", "淘宝/天猫", "都可以"],
    },
    "style": {
        "info_gain": 0.3,
        "impact": 0.4,
        "question_template": "您对风格有偏好吗？",
        "style": ClarificationStyle.OPEN_ENDED,
        "options": [],
    },
    "compatibility": {
        "info_gain": 0.6,
        "impact": 0.8,
        "question_template": "需要与哪些已有设备兼容？",
        "style": ClarificationStyle.OPEN_ENDED,
        "options": [],
    },
}


class ClarificationPolicy:
    """
    澄清策略，支持启发式和 RL 两种模式。

    RL 模式下，每轮调用 decide_rl() 返回单个动作（ASK_SLOT 或 PROCEED），
    由 orchestrator 驱动对话循环。

    启发式模式下，generate() 返回所有候选问题列表（原有接口，向后兼容）。
    """

    def __init__(self, use_rl: bool = False):
        # 启发式模式权重（可由学习层更新）
        self._slot_weights: dict[str, float] = {}
        # RL 模式
        self._use_rl = use_rl
        self._rl_policy: Optional[RLClarificationPolicy] = None
        if use_rl:
            self._rl_policy = RLClarificationPolicy()

    # ------------------------------------------------------------------
    # RL 模式接口
    # ------------------------------------------------------------------

    def set_rl_mode(self, enabled: bool) -> None:
        """切换 RL / 启发式模式（消融实验用）。"""
        self._use_rl = enabled
        if enabled and self._rl_policy is None:
            self._rl_policy = RLClarificationPolicy()

    def set_rl_policy(self, policy: RLClarificationPolicy) -> None:
        """注入已训练的 RL 策略（由 trainer 传入）。"""
        self._rl_policy = policy
        self._use_rl = True

    def decide_rl(
        self,
        task: ShoppingTask,
        conversation_round: int,
        user_profile: Optional[UserProfile] = None,
        explore: bool = True,
    ) -> tuple[Optional[ClarificationQuestion], ClarificationAction, float]:
        """
        RL 模式：根据当前状态决定单个动作。

        返回 (question_or_None, action, log_prob)
          - 若 action.action_type == PROCEED，question 为 None
          - 若 action.action_type == ASK_SLOT，question 为对应问题

        由 orchestrator 的澄清循环逐轮调用。
        """
        assert self._rl_policy is not None, "请先调用 set_rl_mode(True) 或 set_rl_policy()"

        state = self._build_clar_state(task, conversation_round, user_profile)
        action, log_prob = self._rl_policy.decide(state, explore=explore)

        if action.action_type == ClarificationActionType.PROCEED:
            return None, action, log_prob

        question = self._build_question(action.slot or "", task)
        return question, action, log_prob

    def build_state(
        self,
        task: ShoppingTask,
        conversation_round: int,
        user_profile: Optional[UserProfile] = None,
    ) -> ClarificationState:
        """构建澄清状态（供 trainer 调用）。"""
        return self._build_clar_state(task, conversation_round, user_profile)

    # ------------------------------------------------------------------
    # 启发式模式接口（保持向后兼容）
    # ------------------------------------------------------------------

    def generate(
        self,
        task: ShoppingTask,
        user_profile: Optional[UserProfile] = None,
    ) -> list[ClarificationQuestion]:
        """
        启发式模式：根据任务不确定性槽位，生成待问问题列表（按价值降序排列）。
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

    def update_weights(self, slot: str, new_weight: float) -> None:
        """由学习层调用，更新槽位权重。"""
        self._slot_weights[slot] = new_weight

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _build_clar_state(
        self,
        task: ShoppingTask,
        conversation_round: int,
        user_profile: Optional[UserProfile] = None,
    ) -> ClarificationState:
        """将 ShoppingTask 转换为 RL 的 ClarificationState。"""
        slot_names = list(task.uncertainty_slots.keys())
        K = len(slot_names)

        uncertainty_vector = np.zeros(K)
        impact_vector = np.zeros(K)

        for i, slot in enumerate(slot_names):
            val = task.uncertainty_slots.get(slot)
            # 若槽位值为 None/"uncertain"，则不确定度为 1.0
            uncertainty_vector[i] = 1.0 if (val is None or val == "uncertain") else 0.0
            # 从 _SLOT_CONFIG 获取预设 impact；未知槽位默认 0.3
            config = _SLOT_CONFIG.get(slot, {})
            impact_vector[i] = config.get("impact", 0.3)

        budget_known = task.uncertainty_slots.get("budget_total") not in (None, "uncertain")

        # 用户画像特征：[price_sensitivity, prefer_fast, brand_diversity, 归一化品类数]
        profile_features = np.zeros(4)
        if user_profile:
            profile_features[0] = getattr(user_profile, "price_sensitivity", 0.5)
            profile_features[1] = 0.5  # prefer_fast 暂用默认值
            profile_features[2] = min(len(user_profile.brand_weights), 5) / 5.0
            profile_features[3] = min(len(task.categories), 5) / 5.0

        return ClarificationState(
            slot_names=slot_names,
            uncertainty_vector=uncertainty_vector,
            impact_vector=impact_vector,
            conversation_round=conversation_round,
            budget_known=budget_known,
            task_type_id=hash(task.task_type) % 6,  # 简单哈希映射到 0~5
            profile_features=profile_features,
        )

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

    def _score(self, q: ClarificationQuestion, task: ShoppingTask) -> float:
        fatigue_cost = 1.0 + task.uncertainty_score * 0.5
        weight = self._slot_weights.get(q.slot, 1.0)
        return (q.info_gain * q.impact_score * weight) / fatigue_cost

    def _can_fill_from_profile(self, slot: str, profile: UserProfile) -> bool:
        if slot == "size" and profile.size_profile:
            return True
        if slot == "brand_preference" and profile.brand_weights:
            return True
        return False
