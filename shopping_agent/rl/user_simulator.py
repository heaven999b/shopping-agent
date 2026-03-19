"""
User Simulator — 用于 RL 训练的用户行为模拟器。

两个核心功能：
  1. 模拟用户回答澄清问题（给 Clarification Policy 提供训练环境）
  2. 模拟用户对最终方案的满意度评分（提供 episode 级 reward）

实现两种模式：
  - LLMSimulator:  用 Claude 模拟用户（逼真但慢，用于少量 episode 验证）
  - RuleSimulator: 基于规则的模拟（快速，用于大量 RL 训练）

论文中可以写：
  用户被建模为一个隐式偏好向量 u* ∈ U，
  每次回答澄清问题是对 u* 的噪声观测 o = u*_slot + ε, ε ~ N(0, σ²)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only in minimal environments
    anthropic = None

from shopping_agent.common.constants import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from shopping_agent.common.types import ClarificationQuestion, ShoppingTask
from shopping_agent.common.exceptions import ToolUnavailableError


# ---------------------------------------------------------------------------
# 隐式用户偏好（训练时的 ground truth）
# ---------------------------------------------------------------------------

@dataclass
class HiddenUserPreference:
    """
    用户的真实偏好（RL 环境的隐藏状态）。

    在真实系统中这些是未知的，模拟器用它生成观测和 reward。
    """
    budget_total: float
    delivery_days: int
    usage_scenario: str           # "home" | "travel" | "outdoor"
    brand_preference: list[str]   # 偏好品牌列表
    color: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None

    # 用户耐心度（影响多问时的流失概率）
    patience: float = 0.8         # 1.0=非常有耐心，0.0=完全没耐心

    # 需求清晰度（影响初始不确定性）
    clarity: float = 0.5          # 1.0=需求非常清晰

    @classmethod
    def sample_random(cls) -> "HiddenUserPreference":
        """随机采样一个用户偏好（用于训练数据生成）。"""
        return cls(
            budget_total=random.choice([1000, 2000, 3000, 5000, 8000, 15000]),
            delivery_days=random.choice([1, 2, 3, 5, 7]),
            usage_scenario=random.choice(["home", "travel", "outdoor", "office"]),
            brand_preference=random.sample(
                ["Sony", "LG", "Samsung", "Xiaomi", "Apple", "Logitech"], k=2
            ),
            patience=random.uniform(0.5, 1.0),
            clarity=random.uniform(0.3, 0.9),
        )


# ---------------------------------------------------------------------------
# 规则模拟器（训练主力）
# ---------------------------------------------------------------------------

class RuleBasedUserSimulator:
    """
    基于规则的用户模拟器。

    特点：
    - 快速（无 LLM 调用），适合大量 RL 训练
    - 可控（通过 HiddenUserPreference 精确控制用户行为）
    - 支持噪声（模拟真实用户的不确定性）
    """

    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level  # 回答时的噪声（模拟用户不精确表达）

    def answer_clarification(
        self,
        question: ClarificationQuestion,
        user_pref: HiddenUserPreference,
    ) -> tuple[str, float]:
        """
        模拟用户回答澄清问题。

        返回 (answer_text, dropout_prob)
        dropout_prob：用户在此步放弃任务的概率（受耐心度影响）
        """
        # 耐心衰减：每次被问都有一定概率流失
        dropout_prob = max(0.0, (1.0 - user_pref.patience) * 0.2)

        slot = question.slot
        answer = self._get_slot_answer(slot, user_pref)

        # 加入噪声：以 noise_level 概率给出模糊回答
        if random.random() < self.noise_level:
            answer = self._add_noise(answer, question)

        return answer, dropout_prob

    def evaluate_plan(
        self,
        plan_summary: dict[str, Any],
        user_pref: HiddenUserPreference,
    ) -> float:
        """
        模拟用户对推荐方案的满意度评分 [0, 1]。

        用于计算 episode 级 task_success reward。
        """
        score = 0.0
        total_weight = 0.0

        # 预算满足度（权重 0.4）
        total_price = plan_summary.get("total_price", 0)
        if total_price <= user_pref.budget_total:
            budget_score = 1.0
        elif total_price <= user_pref.budget_total * 1.1:
            budget_score = 0.5
        else:
            budget_score = 0.0
        score += 0.4 * budget_score
        total_weight += 0.4

        # 品牌偏好满足度（权重 0.3）
        plan_brands = plan_summary.get("brands", [])
        if plan_brands:
            brand_overlap = len(
                set(plan_brands) & set(user_pref.brand_preference)
            ) / len(plan_brands)
            score += 0.3 * brand_overlap
        total_weight += 0.3

        # 履约时效满足度（权重 0.2）
        max_delivery = plan_summary.get("max_delivery_days")
        if max_delivery is not None:
            if max_delivery <= user_pref.delivery_days:
                score += 0.2
            elif max_delivery <= user_pref.delivery_days + 1:
                score += 0.1
        total_weight += 0.2

        # 评分质量（权重 0.1）
        avg_rating = plan_summary.get("avg_rating", 4.0)
        score += 0.1 * (avg_rating / 5.0)
        total_weight += 0.1

        return score / total_weight if total_weight > 0 else 0.0

    def _get_slot_answer(self, slot: str,
                         user_pref: HiddenUserPreference) -> str:
        answers = {
            "budget_total": str(user_pref.budget_total),
            "delivery_days": f"{user_pref.delivery_days}天内",
            "usage_scenario": user_pref.usage_scenario,
            "brand_preference": "、".join(user_pref.brand_preference),
            "color": user_pref.color or "不限",
            "size": user_pref.size or "不限",
            "style": user_pref.style or "不限",
        }
        return answers.get(slot, "不限")

    def _add_noise(self, answer: str,
                   question: ClarificationQuestion) -> str:
        """模拟用户给出模糊或不精确的回答。"""
        if question.options:
            # 从选项中随机选一个（可能不是最优选项）
            return random.choice(question.options)
        vague_answers = ["随便", "都可以", "没太大要求", "你帮我定吧"]
        return random.choice(vague_answers)


# ---------------------------------------------------------------------------
# LLM 模拟器（验证用）
# ---------------------------------------------------------------------------

class LLMUserSimulator:
    """
    基于 Claude 的用户模拟器。

    适合：
    - 验证 RL 策略在逼真用户行为下的表现
    - 生成少量高质量训练数据
    - 论文中的 human-in-loop 对比实验

    不适合：大规模 RL 训练（太慢）
    """

    def __init__(self):
        if anthropic is None:
            raise ToolUnavailableError("anthropic package is not installed")
        self._client = anthropic.Anthropic()

    def answer_clarification(
        self,
        question: ClarificationQuestion,
        user_pref: HiddenUserPreference,
        task_context: str = "",
    ) -> tuple[str, float]:
        """用 Claude 模拟用户回答澄清问题。"""
        system = (
            "你是一个正在购物的用户。你有明确的购物需求和预算，但你不会主动说出所有细节。"
            "当被问到问题时，给出自然、简短的回答，像真实用户一样。"
        )
        user_context = (
            f"你的购物需求背景：{task_context}\n"
            f"你的预算：{user_pref.budget_total}元\n"
            f"你偏好的品牌：{', '.join(user_pref.brand_preference)}\n"
            f"使用场景：{user_pref.usage_scenario}\n\n"
            f"销售助手问你：{question.question}\n"
        )
        if question.options:
            user_context += f"选项：{' / '.join(question.options)}\n"

        user_context += "请用1-2句话自然地回答。"

        response = self._client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": user_context}],
        )
        answer = next(
            (b.text for b in response.content if b.type == "text"), "不限"
        )
        # LLM 模拟的用户默认耐心度较高
        dropout_prob = 1.0 - user_pref.patience * 0.1
        return answer.strip(), dropout_prob

    def evaluate_plan(
        self,
        plan_summary: dict[str, Any],
        user_pref: HiddenUserPreference,
        task_context: str = "",
    ) -> float:
        """用 Claude 模拟用户满意度评分。"""
        system = (
            "你是一个正在购物的用户。根据你的需求，评价购物方案的满意度，"
            "给出 0.0 到 1.0 的分数，只输出数字。"
        )
        plan_text = "\n".join([
            f"- {k}: {v}" for k, v in plan_summary.items()
        ])
        user_msg = (
            f"你的需求：{task_context}\n"
            f"你的预算：{user_pref.budget_total}元\n"
            f"推荐方案：\n{plan_text}\n\n"
            f"满意度（0.0-1.0）："
        )
        response = self._client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=10,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), "0.5"
        ).strip()
        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            return 0.5
