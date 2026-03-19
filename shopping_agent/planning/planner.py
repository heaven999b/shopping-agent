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
    BundlePlan,
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
    ) -> list[BundlePlan]:
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
    ) -> list[BundlePlan]:
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
    ) -> Optional[BundlePlan]:
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
        total_persona_score = 0.0

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
            total_persona_score += scores["persona"]

            # 找同品类其他候选作为备选
            alternatives = [p for p in candidates if p.product_id != product.product_id][:2]

            items.append(PlanItem(
                bundle_slot=category,
                product=product,
                reason=self._build_reason(product, strategy, scores),
                persona_reason=self._build_persona_reason(product, user_profile),
                alternatives=alternatives,
            ))

        if not items:
            return None

        n = len(items)
        avg_constraint = total_constraint_score / n
        avg_preference = total_preference_score / n
        avg_value = total_value_score / n
        avg_persona = total_persona_score / n
        style_coherence = self._style_coherence_score(items)
        scenario_fit = self._scenario_fit_score(task, items, user_profile)
        bundle_completeness = self._bundle_completeness_score(task, items)
        slot_coverage = self._slot_coverage(task, items)
        compatibility_score = self._compatibility_score(items)
        budget_allocation = self._budget_allocation(items, budget)
        phased_options = self._build_phased_purchase_options(
            task, items, budget, user_profile
        )
        long_term_fit = self._long_term_fit_score(task, items, user_profile)
        phased_purchase_score = self._phased_purchase_score(
            phased_options, total_price, budget, user_profile
        )

        tradeoff_notes = self._build_tradeoff_notes(
            tier_name,
            avg_constraint,
            total_price,
            budget,
            avg_persona,
            user_profile,
        )
        tradeoff_notes.extend(
            self._build_bundle_tradeoff_notes(
                task,
                style_coherence,
                scenario_fit,
                total_price,
                budget,
                phased_options,
            )
        )

        return BundlePlan(
            plan_id=str(uuid.uuid4()),
            task_id=task.task_id,
            items=items,
            total_price=round(total_price, 2),
            total_discount=round(sum(p.product.coupon_discount for p in items), 2),
            constraint_score=round(avg_constraint, 3),
            preference_score=round(avg_preference, 3),
            value_score=round(avg_value, 3),
            persona_alignment_score=round(avg_persona, 3),
            style_coherence_score=round(style_coherence, 3),
            scenario_fit_score=round(scenario_fit, 3),
            bundle_completeness_score=round(bundle_completeness, 3),
            slot_coverage=slot_coverage,
            compatibility_score=round(compatibility_score, 3),
            long_term_fit_score=round(long_term_fit, 3),
            phased_purchase_score=round(phased_purchase_score, 3),
            tradeoff_notes=tradeoff_notes,
            explanation=self._build_explanation(
                tier_name, items, total_price, budget, avg_persona, user_profile
            ),
            persona_summary=self._build_persona_summary(items, user_profile),
            bundle_type="bundle_plan" if len(task.categories) > 1 else "single_item",
            bundle_objective=self._bundle_objective(task),
            budget_allocation=budget_allocation,
            phased_purchase_options=phased_options,
            phased_upgrade_plan=phased_options,
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
        pool = self._filter_owned_duplicates(pool, user_profile) or pool

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
            "persona": self._persona_alignment_score(product, user_profile),
        }
        return product, scores

    @staticmethod
    def _filter_owned_duplicates(
        candidates: list[Product],
        user_profile: Optional[UserProfile],
    ) -> list[Product]:
        if user_profile is None:
            return candidates
        owned_ids = {item.get("product_id") for item in getattr(user_profile, "owned_items", [])}
        if not owned_ids:
            return candidates
        filtered = [product for product in candidates if product.product_id not in owned_ids]
        return filtered

    # ------------------------------------------------------------------
    # RL 规划
    # ------------------------------------------------------------------

    def _plan_with_rl(
        self,
        task: ShoppingTask,
        slot_candidates: dict[str, list[Product]],
        budget: Optional[float],
        user_profile: Optional[UserProfile],
    ) -> list[BundlePlan]:
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
                "persona": self._persona_alignment_score(product, user_profile),
            }
            items.append(PlanItem(
                bundle_slot=category,
                product=product,
                reason=f"RL策略: {self._action_desc(action)}",
                persona_reason=self._build_persona_reason(product, user_profile),
                alternatives=[p for p in candidates if p.product_id != product.product_id][:2],
            ))

        rl_plan = self._assemble_plan(task, items, budget, "RL最优方案", user_profile)
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
            + 0.15 * self._persona_alignment_score(p, user_profile)
            + 0.12 * self._growth_alignment_score(p, user_profile)
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

    @staticmethod
    def _persona_alignment_score(
        p: Product, user_profile: Optional[UserProfile]
    ) -> float:
        if user_profile is None:
            return 0.0

        tags = p.persona_tags or {}
        score = 0.0

        identity_pref = getattr(user_profile, "identity_goal", {})
        for tag in tags.get("identity_fit", []):
            score += identity_pref.get(tag, 0.0) * 0.22

        aesthetic_pref = getattr(user_profile, "aesthetic_preference", {})
        for tag in tags.get("style_signal", []):
            score += aesthetic_pref.get(tag, 0.0) * 0.18

        brand_pref = getattr(user_profile, "brand_orientation", {})
        symbolic_value = tags.get("symbolic_value", "utilitarian")
        if symbolic_value == "taste_signaling":
            score += brand_pref.get("brand_signal", 0.0) * 0.16
        elif symbolic_value == "utilitarian":
            score += brand_pref.get("function_first", 0.0) * 0.14

        budget_profile = getattr(user_profile, "budget_sensitivity_profile", {})
        if p.final_price >= 1800:
            score += budget_profile.get("premium", 0.0) * 0.12
        elif p.final_price <= 800:
            score += budget_profile.get("strict", 0.0) * 0.12

        stability = getattr(user_profile, "persona_stability", 0.5)
        return min(1.0, score * max(0.4, min(1.0, stability)))

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
        self,
        task: ShoppingTask,
        items: list[PlanItem],
        budget: float,
        tier_name: str,
        user_profile: Optional[UserProfile] = None,
    ) -> Optional[BundlePlan]:
        total = sum(i.product.final_price for i in items)
        if not items:
            return None
        n = len(items)
        per_slot = budget / max(n, 1)
        constraint_score = sum(self._constraint_score(i.product, per_slot) for i in items) / n
        preference_score = sum(self._preference_score(i.product, user_profile) for i in items) / n
        value_score = sum(self._value_score(i.product, per_slot) for i in items) / n
        persona_score = sum(
            self._persona_alignment_score(i.product, user_profile) for i in items
        ) / n
        style_coherence = self._style_coherence_score(items)
        scenario_fit = self._scenario_fit_score(task, items, user_profile)
        bundle_completeness = self._bundle_completeness_score(task, items)
        slot_coverage = self._slot_coverage(task, items)
        compatibility_score = self._compatibility_score(items)
        budget_allocation = self._budget_allocation(items, budget)
        phased_options = self._build_phased_purchase_options(
            task, items, budget, user_profile
        )
        long_term_fit = self._long_term_fit_score(task, items, user_profile)
        phased_purchase_score = self._phased_purchase_score(
            phased_options, total, budget, user_profile
        )
        return BundlePlan(
            plan_id=str(uuid.uuid4()),
            task_id=task.task_id,
            items=items,
            total_price=round(total, 2),
            total_discount=round(sum(i.product.coupon_discount for i in items), 2),
            constraint_score=round(constraint_score, 3),
            preference_score=round(preference_score, 3),
            value_score=round(value_score, 3),
            persona_alignment_score=round(persona_score, 3),
            style_coherence_score=round(style_coherence, 3),
            scenario_fit_score=round(scenario_fit, 3),
            bundle_completeness_score=round(bundle_completeness, 3),
            slot_coverage=slot_coverage,
            compatibility_score=round(compatibility_score, 3),
            long_term_fit_score=round(long_term_fit, 3),
            phased_purchase_score=round(phased_purchase_score, 3),
            tradeoff_notes=self._build_tradeoff_notes(
                tier_name,
                constraint_score,
                total,
                budget,
                persona_score,
                user_profile,
            ),
            explanation=self._build_explanation(
                tier_name, items, total, budget, persona_score, user_profile
            ),
            persona_summary=self._build_persona_summary(items, user_profile),
            bundle_type="bundle_plan" if len(task.categories) > 1 else "single_item",
            bundle_objective=self._bundle_objective(task),
            budget_allocation=budget_allocation,
            phased_purchase_options=phased_options,
            phased_upgrade_plan=phased_options,
        )

    @staticmethod
    def _build_reason(product: Product, strategy: str, scores: dict) -> str:
        strategy_desc = {"best_composite": "综合最优", "cheapest": "价格最低", "highest_rated": "评分最高"}
        desc = strategy_desc.get(strategy, "推荐")
        return (
            f"{desc}选择：{product.brand} {product.title[:20]}，"
            f"售价¥{product.final_price:.0f}，评分{product.rating}，"
            f"画像匹配{scores.get('persona', 0.0):.0%}"
        )

    @staticmethod
    def _build_tradeoff_notes(
        tier_name: str,
        constraint_score: float,
        total: float,
        budget: float,
        persona_score: float,
        user_profile: Optional[UserProfile],
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
        if user_profile and persona_score >= 0.55:
            notes.append(TradeoffNote(
                dimension="persona",
                note="这套方案与当前用户画像较为一致，优先保留了风格表达和身份匹配。",
                severity="info",
            ))
        elif user_profile and persona_score < 0.3:
            notes.append(TradeoffNote(
                dimension="persona",
                note="这套方案在功能和预算上更稳妥，但对当前画像表达的贴合度相对一般。",
                severity="info",
            ))
        return notes

    @staticmethod
    def _build_explanation(
        tier_name: str,
        items: list[PlanItem],
        total: float,
        budget: float,
        persona_score: float,
        user_profile: Optional[UserProfile],
    ) -> str:
        lines = [f"【{tier_name}】总计¥{total:.0f}（预算¥{budget:.0f}）"]
        if user_profile:
            lines.append(f"画像匹配度：{persona_score:.0%}")
        for item in items:
            lines.append(
                f"  · {item.bundle_slot}: {item.product.brand} {item.product.title[:20]}"
                f" ¥{item.product.final_price:.0f} (★{item.product.rating})"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_persona_reason(
        product: Product,
        user_profile: Optional[UserProfile],
    ) -> str:
        if user_profile is None:
            return ""

        tags = product.persona_tags or {}
        reasons: list[str] = []

        identity_pref = getattr(user_profile, "identity_goal", {})
        identity_fit = [
            tag for tag in tags.get("identity_fit", [])
            if identity_pref.get(tag, 0.0) >= 0.4
        ]
        if identity_fit:
            reasons.append(f"更贴近你当前偏好的{identity_fit[0]}路线")

        aesthetic_pref = getattr(user_profile, "aesthetic_preference", {})
        style_fit = [
            tag for tag in tags.get("style_signal", [])
            if aesthetic_pref.get(tag, 0.0) >= 0.4
        ]
        if style_fit:
            reasons.append(f"风格上更偏{style_fit[0]}")

        brand_pref = getattr(user_profile, "brand_orientation", {})
        symbolic_value = tags.get("symbolic_value")
        if symbolic_value == "taste_signaling" and brand_pref.get("brand_signal", 0.0) >= 0.5:
            reasons.append("品牌辨识度更强")
        elif symbolic_value == "utilitarian" and brand_pref.get("function_first", 0.0) >= 0.5:
            reasons.append("更偏实用导向")

        if not reasons:
            return ""
        return "；".join(reasons[:2])

    def _build_persona_summary(
        self,
        items: list[PlanItem],
        user_profile: Optional[UserProfile],
    ) -> str:
        if user_profile is None or not items:
            return ""

        reasons = [item.persona_reason for item in items if item.persona_reason]
        if reasons:
            return "；".join(reasons[:2])

        strongest_identity = self._top_persona_label(getattr(user_profile, "identity_goal", {}))
        strongest_style = self._top_persona_label(
            getattr(user_profile, "aesthetic_preference", {})
        )
        summary_parts = []
        if strongest_identity:
            summary_parts.append(f"整体更靠近{strongest_identity}定位")
        if strongest_style:
            summary_parts.append(f"风格上偏{strongest_style}")
        return "；".join(summary_parts)

    @staticmethod
    def _top_persona_label(weights: dict[str, float]) -> str:
        if not weights:
            return ""
        return max(weights.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _bundle_objective(task: ShoppingTask) -> str:
        if task.task_type.value == "bundle":
            return f"为{ '、'.join(task.categories) }生成一套可协同购买的组合方案"
        if len(task.categories) > 1:
            return f"为{ '、'.join(task.categories) }做组合补齐"
        return f"为{task.categories[0] if task.categories else '当前需求'}选择最合适单品"

    @staticmethod
    def _slot_coverage(task: ShoppingTask, items: list[PlanItem]) -> dict[str, bool]:
        filled_slots = {item.bundle_slot for item in items}
        return {category: category in filled_slots for category in task.categories}

    @staticmethod
    def _compatibility_score(items: list[PlanItem]) -> float:
        if len(items) <= 1:
            return 1.0

        slot_count = len({item.bundle_slot for item in items})
        product_count = len({item.product.product_id for item in items})
        slot_ratio = slot_count / max(len(items), 1)
        product_ratio = product_count / max(len(items), 1)

        delivery_days = [
            item.product.logistics.delivery_days
            for item in items
            if item.product.logistics and item.product.logistics.delivery_days is not None
        ]
        delivery_score = 1.0
        if len(delivery_days) >= 2:
            spread = max(delivery_days) - min(delivery_days)
            delivery_score = max(0.5, 1.0 - spread * 0.08)

        return min(1.0, 0.45 * slot_ratio + 0.35 * product_ratio + 0.2 * delivery_score)

    @staticmethod
    def _budget_allocation(items: list[PlanItem], budget: float) -> dict[str, float]:
        if budget <= 0:
            return {}
        allocation = {}
        for item in items:
            allocation[item.bundle_slot] = round(item.product.final_price / budget, 3)
        return allocation

    @staticmethod
    def _growth_alignment_score(
        product: Product,
        user_profile: Optional[UserProfile],
    ) -> float:
        if user_profile is None:
            return 0.0

        score = 0.0
        owned_ids = {item.get("product_id") for item in getattr(user_profile, "owned_items", [])}
        owned_categories = {item.get("category") for item in getattr(user_profile, "owned_items", [])}
        if product.product_id in owned_ids:
            score -= 0.6
        if product.category in owned_categories:
            score -= 0.08

        active_setups = getattr(user_profile, "active_setups", {})
        for setup in active_setups.values():
            missing = setup.get("missing", [])
            next_upgrade = setup.get("next_best_upgrade")
            if product.category in missing or product.product_id in missing:
                score += 0.18
            if next_upgrade and (product.category == next_upgrade or product.product_id == next_upgrade):
                score += 0.22

        cadence = getattr(user_profile, "purchase_rhythm", {}).get("cadence")
        if cadence == "incremental" and product.final_price <= 1000:
            score += 0.08
        elif cadence == "upgrade_oriented" and product.final_price >= 1500:
            score += 0.08

        return max(-0.6, min(1.0, score))

    @staticmethod
    def _style_coherence_score(items: list[PlanItem]) -> float:
        if len(items) <= 1:
            return 1.0
        tag_sets = [
            set(item.product.persona_tags.get("style_signal", []))
            for item in items
            if item.product.persona_tags
        ]
        if not tag_sets:
            return 0.5
        shared = set.intersection(*tag_sets) if len(tag_sets) > 1 else tag_sets[0]
        union = set.union(*tag_sets)
        if not union:
            return 0.5
        return max(0.2, len(shared) / len(union))

    @staticmethod
    def _scenario_fit_score(
        task: ShoppingTask,
        items: list[PlanItem],
        user_profile: Optional[UserProfile],
    ) -> float:
        if not items:
            return 0.0
        score = 0.45 if len({item.bundle_slot for item in items}) == len(task.categories) else 0.2
        if any("商务" in need or "办公" in need for need in task.implicit_needs):
            professional_hits = sum(
                1
                for item in items
                if "professional" in item.product.persona_tags.get("identity_fit", [])
            )
            score += 0.35 * (professional_hits / len(items))
        elif user_profile and user_profile.identity_goal:
            target = max(user_profile.identity_goal.items(), key=lambda item: item[1])[0]
            hits = sum(
                1
                for item in items
                if target in item.product.persona_tags.get("identity_fit", [])
            )
            score += 0.25 * (hits / len(items))
        return min(1.0, score)

    @staticmethod
    def _bundle_completeness_score(task: ShoppingTask, items: list[PlanItem]) -> float:
        if not task.categories:
            return 1.0
        covered = {item.bundle_slot for item in items}
        return len(covered) / len(task.categories)

    def _long_term_fit_score(
        self,
        task: ShoppingTask,
        items: list[PlanItem],
        user_profile: Optional[UserProfile],
    ) -> float:
        if not items:
            return 0.0
        if user_profile is None:
            return 0.5

        scores = [max(0.0, self._growth_alignment_score(item.product, user_profile)) for item in items]
        score = sum(scores) / len(scores) if scores else 0.0

        active_setups = getattr(user_profile, "active_setups", {})
        for category in task.categories:
            setup = active_setups.get(f"{category}_setup")
            if not setup:
                continue
            if setup.get("next_best_upgrade"):
                score += 0.08
            if setup.get("missing"):
                score += 0.05

        stage_values = {"starter": 0.58, "growing": 0.72, "refinement": 0.82}
        if user_profile.upgrade_stage:
            avg_stage = sum(stage_values.get(v, 0.6) for v in user_profile.upgrade_stage.values()) / len(
                user_profile.upgrade_stage
            )
            score = (score + avg_stage) / 2

        return min(1.0, score)

    def _build_phased_purchase_options(
        self,
        task: ShoppingTask,
        items: list[PlanItem],
        budget: float,
        user_profile: Optional[UserProfile],
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        sorted_items = sorted(
            items,
            key=lambda item: (
                self._persona_alignment_score(item.product, user_profile),
                item.product.rating or 0.0,
                -item.product.final_price,
            ),
            reverse=True,
        )
        phase_one_items = [sorted_items[0]]
        later_items = sorted_items[1:]
        conservative_item = min(items, key=lambda item: item.product.final_price)

        return [
            {
                "route": "一步到位",
                "goal": "一次性完成当前组合需求",
                "slots": [item.bundle_slot for item in items],
                "budget": round(sum(item.product.final_price for item in items), 2),
            },
            {
                "route": "分阶段升级",
                "goal": "先补核心件，再提升整体一致性",
                "phase_1_slots": [item.bundle_slot for item in phase_one_items],
                "phase_2_slots": [item.bundle_slot for item in later_items],
                "phase_1_budget": round(sum(item.product.final_price for item in phase_one_items), 2),
                "phase_2_budget": round(sum(item.product.final_price for item in later_items), 2),
            },
            {
                "route": "保守路线",
                "goal": "先用最低风险方式建立基础组合",
                "slots": [conservative_item.bundle_slot],
                "budget": round(conservative_item.product.final_price, 2),
            },
        ]

    @staticmethod
    def _phased_purchase_score(
        phased_options: list[dict[str, Any]],
        total: float,
        budget: float,
        user_profile: Optional[UserProfile],
    ) -> float:
        if not phased_options:
            return 0.0
        if budget <= 0:
            return 0.7

        pressure = total / budget
        score = 1.0 if pressure <= 0.85 else max(0.2, 1.0 - (pressure - 0.85) * 1.8)
        if pressure > 0.9 and len(phased_options) >= 2:
            phase_one_budget = phased_options[1].get("phase_1_budget", total)
            score = max(score, min(1.0, budget / max(phase_one_budget, 1.0)) * 0.7)

        cadence = getattr(user_profile, "purchase_rhythm", {}).get("cadence") if user_profile else None
        if cadence == "incremental" and len(phased_options) >= 2:
            score = min(1.0, score + 0.1)
        elif cadence == "upgrade_oriented" and pressure <= 1.0:
            score = min(1.0, score + 0.05)

        return max(0.0, min(1.0, score))

    @staticmethod
    def _build_bundle_tradeoff_notes(
        task: ShoppingTask,
        style_coherence: float,
        scenario_fit: float,
        total: float,
        budget: float,
        phased_options: list[dict[str, Any]],
    ) -> list[TradeoffNote]:
        notes: list[TradeoffNote] = []
        if len(task.categories) > 1 and style_coherence < 0.45:
            notes.append(TradeoffNote(
                dimension="style",
                note="当前组合在风格统一性上偏一般，更适合先补齐功能核心件再做整体升级。",
                severity="info",
            ))
        if len(task.categories) > 1 and scenario_fit < 0.55:
            notes.append(TradeoffNote(
                dimension="scenario",
                note="这套组合能满足核心需求，但与当前使用场景的贴合度还有提升空间。",
                severity="info",
            ))
        if total > budget * 0.85 and phased_options:
            phase = phased_options[1]
            notes.append(TradeoffNote(
                dimension="phasing",
                note=(
                    f"如果想降低一次性预算压力，可以先走“{phase['route']}”，"
                    f"首阶段预算约¥{phase.get('phase_1_budget', 0):.0f}。"
                ),
                severity="info",
            ))
        return notes
