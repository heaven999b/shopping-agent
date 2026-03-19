"""
ConstraintAwarePlanner — 约束感知多目标规划器。

支持两种模式：
  - 启发式模式（默认）：贪心多目标打分，生成三档方案（主推/省钱/升级）
  - RL 模式（Option A）：π_φ(a|s) 学习如何在候选商品图上逐步选品

流程：
  1. 任务分解：将 bundle 任务按品类拆分为独立坑位
  2. 候选分组：从 retrieved_products 按品类分组
  3. 启发式打分：constraint_score + preference_score + value_score
  4. 多目标生成：主推方案 / 最省钱方案 / 最高评分方案
  5. 返回 Top-N 方案

评分公式：
  composite = 0.5 * constraint_score
            + 0.3 * preference_score
            + 0.2 * value_score

  constraint_score  = 1 - max(0, (price - budget) / budget)
  preference_score  = brand_weight * 0.4 + platform_weight * 0.2 + rating_norm * 0.4
  value_score       = rating / 5.0 * (1 - price / budget_per_slot)^0.5
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np

from shopping_agent.common.constants import (
    MAX_PLANS_OUTPUT,
    SCORE_WEIGHT_CONSTRAINT,
    SCORE_WEIGHT_PREFERENCE,
    SCORE_WEIGHT_VALUE,
)
from shopping_agent.common.exceptions import InfeasibleConstraintError, PlanningError
from shopping_agent.common.types import (
    CandidatePlan,
    PlanItem,
    Product,
    ShoppingTask,
    TradeoffNote,
    UserProfile,
)
from shopping_agent.rl.pomdp import PlanningAction, PlanningActionType, PlanningState
from shopping_agent.rl.policy import RLPlanningPolicy


class ConstraintAwarePlanner:
    """
    约束感知规划器，支持启发式和 RL 两种规划模式。

    RL 模式下，策略 π_φ 在每个坑位选品时做出高层决策：
      0 → SELECT_BEST_MATCH   （综合偏好+评分）
      1 → SELECT_BUDGET_OPT   （性价比最优）
      2 → SELECT_SAFE         （高评分+有库存）
    """

    def __init__(self, use_rl: bool = False):
        self._use_rl = use_rl
        self._rl_policy: Optional[RLPlanningPolicy] = None
        if use_rl:
            self._rl_policy = RLPlanningPolicy()

    # ------------------------------------------------------------------
    # 模式控制
    # ------------------------------------------------------------------

    def set_rl_mode(self, enabled: bool) -> None:
        self._use_rl = enabled
        if enabled and self._rl_policy is None:
            self._rl_policy = RLPlanningPolicy()

    def set_rl_policy(self, policy: RLPlanningPolicy) -> None:
        self._rl_policy = policy
        self._use_rl = True

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def plan(
        self,
        task: ShoppingTask,
        candidate_graph: dict[str, list[dict[str, Any]]],
        user_profile: Optional[UserProfile] = None,
        retrieved_products: Optional[list[Product]] = None,
    ) -> list[CandidatePlan]:
        """
        生成多套候选方案，按综合得分降序返回。

        参数：
          retrieved_products — 已检索到的商品列表（由 orchestrator 通过 state 传入）
                               若为 None 则从 candidate_graph 中重建（兼容旧接口）
        """
        hard = task.get_hard_constraints()
        budget = hard.get("budget_total")

        # 1. 按品类分组候选商品
        slot_candidates = self._group_by_category(
            task, retrieved_products or [], budget
        )

        if not any(slot_candidates.values()):
            raise InfeasibleConstraintError(
                f"在约束条件下无法找到可行方案。"
                f"预算={budget}，品类={task.categories}"
            )

        # 2. 生成方案
        if self._use_rl and self._rl_policy is not None:
            plans = self._plan_with_rl(task, slot_candidates, budget, user_profile)
        else:
            plans = self._plan_heuristic(task, slot_candidates, budget, user_profile)

        if not plans:
            raise PlanningError("无法在约束范围内生成有效购物方案。")

        plans.sort(key=lambda p: p.overall_score, reverse=True)
        return plans[:MAX_PLANS_OUTPUT]

    def plan_rl_step(
        self,
        task: ShoppingTask,
        budget_used: float,
        filled_slots: int,
        constraint_sat: float,
        pref_match: float,
        explore: bool = True,
    ) -> tuple[PlanningAction, float, PlanningState]:
        """RL 单步规划决策（供 trainer 逐步调用）。"""
        assert self._rl_policy is not None
        budget = task.get_hard_constraints().get("budget_total") or 5000.0
        state = PlanningState(
            budget_total=budget,
            budget_used=budget_used,
            total_slots=len(task.categories),
            filled_slots=filled_slots,
            constraint_sat_partial=constraint_sat,
            preference_match_partial=pref_match,
            incompatible_risk=0.0,
        )
        action, log_prob = self._rl_policy.decide(state, explore=explore)
        return action, log_prob, state

    # ------------------------------------------------------------------
    # 启发式规划
    # ------------------------------------------------------------------

    def _plan_heuristic(
        self,
        task: ShoppingTask,
        slot_candidates: dict[str, list[Product]],
        budget: Optional[float],
        user_profile: Optional[UserProfile],
    ) -> list[CandidatePlan]:
        """
        生成三档方案：
          Tier 0 — 主推方案：综合得分最高
          Tier 1 — 省钱方案：在满足约束的前提下选最低价
          Tier 2 — 高评方案：选评分最高（可能超预算，由 verifier 处理）
        """
        plans = []

        # Tier 0: 主推
        plan0 = self._build_scored_plan(
            task, slot_candidates, budget, user_profile,
            strategy="best_composite", tier_name="主推方案"
        )
        if plan0:
            plans.append(plan0)

        # Tier 1: 省钱
        plan1 = self._build_scored_plan(
            task, slot_candidates, budget, user_profile,
            strategy="cheapest", tier_name="省钱方案"
        )
        if plan1 and (not plans or plan1.plan_id != plans[0].plan_id):
            plans.append(plan1)

        # Tier 2: 高评
        plan2 = self._build_scored_plan(
            task, slot_candidates, budget, user_profile,
            strategy="highest_rated", tier_name="高评方案"
        )
        if plan2 and all(p.plan_id != plan2.plan_id for p in plans):
            plans.append(plan2)

        return plans

    def _build_scored_plan(
        self,
        task: ShoppingTask,
        slot_candidates: dict[str, list[Product]],
        budget: Optional[float],
        user_profile: Optional[UserProfile],
        strategy: str,
        tier_name: str,
    ) -> Optional[CandidatePlan]:
        """
        按指定策略为每个坑位选一个商品，组合为一套方案。
        """
        budget = budget or 5000.0
        per_slot_budget = budget / max(len(task.categories), 1)

        items: list[PlanItem] = []
        total_price = 0.0
        total_constraint_score = 0.0
        total_preference_score = 0.0
        total_value_score = 0.0

        for category in task.categories:
            candidates = slot_candidates.get(category, [])
            if not candidates:
                return None  # 任何坑位无候选则放弃此方案

            product, scores = self._select_product(
                candidates, strategy, per_slot_budget, budget - total_price, user_profile
            )
            if product is None:
                return None

            total_price += product.final_price
            total_constraint_score += scores["constraint"]
            total_preference_score += scores["preference"]
            total_value_score += scores["value"]

            # 找同品类其他候选作为备选
            alternatives = [p for p in candidates if p.product_id != product.product_id][:2]

            items.append(PlanItem(
                bundle_slot=category,
                product=product,
                reason=self._build_reason(product, strategy, scores),
                alternatives=alternatives,
            ))

        if not items:
            return None

        n = len(items)
        avg_constraint = total_constraint_score / n
        avg_preference = total_preference_score / n
        avg_value = total_value_score / n

        tradeoff_notes = self._build_tradeoff_notes(tier_name, avg_constraint, total_price, budget)

        return CandidatePlan(
            plan_id=str(uuid.uuid4()),
            task_id=task.task_id,
            items=items,
            total_price=round(total_price, 2),
            total_discount=round(sum(p.product.coupon_discount for p in items), 2),
            constraint_score=round(avg_constraint, 3),
            preference_score=round(avg_preference, 3),
            value_score=round(avg_value, 3),
            tradeoff_notes=tradeoff_notes,
            explanation=self._build_explanation(tier_name, items, total_price, budget),
        )

    def _select_product(
        self,
        candidates: list[Product],
        strategy: str,
        per_slot_budget: float,
        remaining_budget: float,
        user_profile: Optional[UserProfile],
    ) -> tuple[Optional[Product], dict[str, float]]:
        """
        按策略从候选中选一个商品，返回 (product, scores)。
        scores = {constraint: float, preference: float, value: float}
        """
        if not candidates:
            return None, {}

        # 预算内候选优先
        affordable = [p for p in candidates if p.final_price <= remaining_budget]
        pool = affordable if affordable else candidates

        if strategy == "cheapest":
            product = min(pool, key=lambda p: p.final_price)
        elif strategy == "highest_rated":
            product = max(pool, key=lambda p: p.rating or 0)
        else:  # best_composite
            product = max(
                pool,
                key=lambda p: self._composite_score(p, per_slot_budget, user_profile),
            )

        scores = {
            "constraint": self._constraint_score(product, per_slot_budget),
            "preference": self._preference_score(product, user_profile),
            "value": self._value_score(product, per_slot_budget),
        }
        return product, scores

    # ------------------------------------------------------------------
    # RL 规划
    # ------------------------------------------------------------------

    def _plan_with_rl(
        self,
        task: ShoppingTask,
        slot_candidates: dict[str, list[Product]],
        budget: Optional[float],
        user_profile: Optional[UserProfile],
    ) -> list[CandidatePlan]:
        """用 RL 策略生成一套最优方案，并附加启发式 backup 方案。"""
        budget = budget or 5000.0
        per_slot_budget = budget / max(len(task.categories), 1)
        items: list[PlanItem] = []
        budget_used = 0.0

        for slot_idx, category in enumerate(task.categories):
            candidates = slot_candidates.get(category, [])
            state = PlanningState(
                budget_total=budget,
                budget_used=budget_used,
                total_slots=len(task.categories),
                filled_slots=slot_idx,
                constraint_sat_partial=1.0 if budget_used <= budget else 0.5,
                preference_match_partial=0.7,
                incompatible_risk=0.0,
            )
            action, _ = self._rl_policy.decide(state, explore=False)  # type: ignore
            product = self._select_by_rl_action(action, candidates, budget - budget_used, user_profile)

            if product is None:
                return self._plan_heuristic(task, slot_candidates, budget, user_profile)

            budget_used += product.final_price
            scores = {
                "constraint": self._constraint_score(product, per_slot_budget),
                "preference": self._preference_score(product, user_profile),
                "value": self._value_score(product, per_slot_budget),
            }
            items.append(PlanItem(
                bundle_slot=category,
                product=product,
                reason=f"RL策略: {self._action_desc(action)}",
                alternatives=[p for p in candidates if p.product_id != product.product_id][:2],
            ))

        rl_plan = self._assemble_plan(task, items, budget, "RL最优方案")
        backup = self._plan_heuristic(task, slot_candidates, budget, user_profile)
        return ([rl_plan] if rl_plan else []) + backup

    def _select_by_rl_action(
        self,
        action: PlanningAction,
        candidates: list[Product],
        remaining_budget: float,
        user_profile: Optional[UserProfile],
    ) -> Optional[Product]:
        if not candidates:
            return None
        affordable = [p for p in candidates if p.final_price <= remaining_budget] or candidates
        if action.action_id == 0:
            pref_brands = set(user_profile.brand_weights.keys()) if user_profile else set()
            return max(affordable, key=lambda p: (p.rating or 0) + (0.5 if p.brand in pref_brands else 0.0))
        elif action.action_id == 1:
            return max(affordable, key=lambda p: (p.rating or 0) / max(p.final_price, 1.0))
        else:
            return max(affordable, key=lambda p: p.rating or 0)

    def _action_desc(self, action: PlanningAction) -> str:
        return {0: "最佳匹配", 1: "性价比优先", 2: "高评安全"}.get(action.action_id, "?")

    # ------------------------------------------------------------------
    # 评分函数
    # ------------------------------------------------------------------

    def _composite_score(
        self, p: Product, per_slot_budget: float, user_profile: Optional[UserProfile]
    ) -> float:
        return (
            SCORE_WEIGHT_CONSTRAINT * self._constraint_score(p, per_slot_budget)
            + SCORE_WEIGHT_PREFERENCE * self._preference_score(p, user_profile)
            + SCORE_WEIGHT_VALUE * self._value_score(p, per_slot_budget)
        )

    @staticmethod
    def _constraint_score(p: Product, per_slot_budget: float) -> float:
        """预算满足度（越贵越低，超预算线性惩罚）。"""
        if per_slot_budget <= 0:
            return 1.0
        ratio = p.final_price / per_slot_budget
        if ratio <= 1.0:
            return 1.0
        return max(0.0, 1.0 - (ratio - 1.0))

    @staticmethod
    def _preference_score(p: Product, user_profile: Optional[UserProfile]) -> float:
        """偏好匹配度：品牌权重 + 平台权重 + 评分归一化。"""
        brand_score = 0.5
        platform_score = 0.5
        if user_profile:
            brand_score = user_profile.brand_weights.get(p.brand, 0.5)
            platform_score = user_profile.platform_preferences.get(p.platform, 0.5)

        rating_norm = (p.rating or 3.0) / 5.0
        return 0.4 * brand_score + 0.2 * platform_score + 0.4 * rating_norm

    @staticmethod
    def _value_score(p: Product, per_slot_budget: float) -> float:
        """性价比：评分高、价格低则分高。"""
        rating_norm = (p.rating or 3.0) / 5.0
        if per_slot_budget <= 0:
            return rating_norm
        price_ratio = min(p.final_price / per_slot_budget, 2.0)
        return rating_norm * max(0.0, 1.0 - price_ratio * 0.3)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _group_by_category(
        self,
        task: ShoppingTask,
        products: list[Product],
        budget: Optional[float],
    ) -> dict[str, list[Product]]:
        """按品类将商品列表分组（软匹配）。"""
        from shopping_agent.data.loader import ProductCatalog
        result: dict[str, list[Product]] = {cat: [] for cat in task.categories}
        for product in products:
            for cat in task.categories:
                if ProductCatalog._category_match(product.category, cat):
                    result[cat].append(product)
                    break
        return result

    def _assemble_plan(
        self, task: ShoppingTask, items: list[PlanItem], budget: float, tier_name: str
    ) -> Optional[CandidatePlan]:
        total = sum(i.product.final_price for i in items)
        if not items:
            return None
        n = len(items)
        per_slot = budget / max(n, 1)
        constraint_score = sum(self._constraint_score(i.product, per_slot) for i in items) / n
        preference_score = sum(self._preference_score(i.product, None) for i in items) / n
        value_score = sum(self._value_score(i.product, per_slot) for i in items) / n
        return CandidatePlan(
            plan_id=str(uuid.uuid4()),
            task_id=task.task_id,
            items=items,
            total_price=round(total, 2),
            total_discount=round(sum(i.product.coupon_discount for i in items), 2),
            constraint_score=round(constraint_score, 3),
            preference_score=round(preference_score, 3),
            value_score=round(value_score, 3),
            tradeoff_notes=self._build_tradeoff_notes(tier_name, constraint_score, total, budget),
            explanation=self._build_explanation(tier_name, items, total, budget),
        )

    @staticmethod
    def _build_reason(product: Product, strategy: str, scores: dict) -> str:
        strategy_desc = {"best_composite": "综合最优", "cheapest": "价格最低", "highest_rated": "评分最高"}
        desc = strategy_desc.get(strategy, "推荐")
        return (
            f"{desc}选择：{product.brand} {product.title[:20]}，"
            f"售价¥{product.final_price:.0f}，评分{product.rating}"
        )

    @staticmethod
    def _build_tradeoff_notes(
        tier_name: str, constraint_score: float, total: float, budget: float
    ) -> list[TradeoffNote]:
        notes = []
        if total > budget:
            notes.append(TradeoffNote(
                dimension="price",
                note=f"总价¥{total:.0f}超出预算¥{budget:.0f}（超出{(total-budget)/budget*100:.1f}%）",
                severity="warning",
            ))
        if "省钱" in tier_name:
            notes.append(TradeoffNote(
                dimension="price",
                note="选择了各品类中价格最低的商品，部分评分略低于主推方案",
                severity="info",
            ))
        if "高评" in tier_name:
            notes.append(TradeoffNote(
                dimension="performance",
                note="优先选择高评分商品，价格可能略高",
                severity="info",
            ))
        return notes

    @staticmethod
    def _build_explanation(
        tier_name: str, items: list[PlanItem], total: float, budget: float
    ) -> str:
        lines = [f"【{tier_name}】总计¥{total:.0f}（预算¥{budget:.0f}）"]
        for item in items:
            lines.append(
                f"  · {item.bundle_slot}: {item.product.brand} {item.product.title[:20]}"
                f" ¥{item.product.final_price:.0f} (★{item.product.rating})"
            )
        return "\n".join(lines)
