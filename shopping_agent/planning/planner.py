"""
ConstraintAwarePlanner — 约束感知多目标规划器。

支持两种模式：
  - 启发式模式（默认，作为 baseline）：枚举三档方案（主推/省钱/升级）
  - RL 模式（Option A）：π_φ(a|s) 学习如何在候选商品图上逐步选品

流程：
  1. 任务分解：将 bundle 任务拆成各坑位的独立子问题
  2. 候选筛选：从候选图中按硬约束过滤
  3. 预算分配：根据品类优先级分配预算
  4. 多目标优化：在约束满足度、偏好匹配度、性价比三维上优化
  5. 生成多套方案（最多 MAX_PLANS_OUTPUT 套）
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np

from shopping_agent.common.constants import (
    MAX_CANDIDATES_PER_SLOT,
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
      0 → SELECT_BEST_MATCH   （按偏好+评分选最优候选）
      1 → SELECT_BUDGET_OPT   （按性价比选最优候选）
      2 → SELECT_SAFE         （按评分+库存选最保守候选）

    这一高层动作空间使 RL 在小数据条件下也能有效学习，
    后续可扩展为指向具体商品 ID 的 pointer network。
    """

    def __init__(self, use_rl: bool = False):
        self._use_rl = use_rl
        self._rl_policy: Optional[RLPlanningPolicy] = None
        if use_rl:
            self._rl_policy = RLPlanningPolicy()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def set_rl_mode(self, enabled: bool) -> None:
        """切换 RL / 启发式模式（消融实验用）。"""
        self._use_rl = enabled
        if enabled and self._rl_policy is None:
            self._rl_policy = RLPlanningPolicy()

    def set_rl_policy(self, policy: RLPlanningPolicy) -> None:
        """注入已训练的 RL 策略（由 trainer 传入）。"""
        self._rl_policy = policy
        self._use_rl = True

    def plan(
        self,
        task: ShoppingTask,
        candidate_graph: dict[str, list[dict[str, Any]]],
        user_profile: Optional[UserProfile] = None,
    ) -> list[CandidatePlan]:
        """
        生成多套候选方案，按综合得分降序返回。
        """
        hard = task.get_hard_constraints()
        budget = hard.get("budget_total")

        # 1. 从候选图提取各坑位的候选商品
        slot_candidates = self._extract_slot_candidates(
            task, candidate_graph, budget
        )

        if not slot_candidates:
            raise InfeasibleConstraintError(
                f"在约束条件下无法找到可行方案。"
                f"预算={budget}，品类={task.categories}"
            )

        # 2. 生成方案
        if self._use_rl and self._rl_policy is not None:
            plans = self._plan_with_rl(task, slot_candidates, budget, user_profile)
        else:
            plans = self._enumerate_plans(task, slot_candidates, budget, user_profile)

        if not plans:
            raise PlanningError("无法在约束范围内生成有效购物方案。")

        # 3. 按综合得分排序，返回 Top-N
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
        """
        RL 单步规划决策（供 trainer 逐步调用）。

        返回 (action, log_prob, state)
        """
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
    # RL 规划（生产用途：用学到的策略生成一套最优方案）
    # ------------------------------------------------------------------

    def _plan_with_rl(
        self,
        task: ShoppingTask,
        slot_candidates: dict[str, list[Product]],
        budget: Optional[float],
        user_profile: Optional[UserProfile],
    ) -> list[CandidatePlan]:
        """
        用 RL 策略生成单套最优方案（推理时 explore=False）。
        同时保留启发式 backup 方案共同返回，以满足 MAX_PLANS_OUTPUT。
        """
        budget = budget or 5000.0
        items = []
        budget_used = 0.0

        for slot_idx, (slot, candidates) in enumerate(slot_candidates.items()):
            state = PlanningState(
                budget_total=budget,
                budget_used=budget_used,
                total_slots=len(task.categories),
                filled_slots=slot_idx,
                constraint_sat_partial=1.0 if budget_used <= budget else 0.5,
                preference_match_partial=0.7,
                incompatible_risk=0.0,
            )
            action, _ = self._rl_policy.decide(state, explore=False)  # type: ignore[union-attr]

            product = self._select_by_action(action, candidates, budget - budget_used, user_profile)
            if product is None:
                # 无候选则用 stub
                product = self._make_stub_product(slot, budget, user_profile, tier=0)

            budget_used += product.final_price
            items.append(PlanItem(
                bundle_slot=slot,
                product=product,
                reason=f"RL策略决策: {self._action_desc(action)}",
            ))

        rl_plan = self._build_plan_from_items(task, items, budget, tier_name="RL最优方案")
        backup_plans = self._enumerate_plans(task, slot_candidates, budget, user_profile)

        return ([rl_plan] if rl_plan else []) + backup_plans

    def _select_by_action(
        self,
        action: PlanningAction,
        candidates: list[Product],
        remaining_budget: float,
        user_profile: Optional[UserProfile],
    ) -> Optional[Product]:
        """根据 RL 高层动作从候选列表中选择一个商品。"""
        if not candidates:
            return None

        affordable = [p for p in candidates if p.final_price <= remaining_budget]
        if not affordable:
            affordable = candidates  # 预算不足时不过滤，让 verifier 处理

        action_id = action.action_id

        if action_id == 0:  # SELECT_BEST_MATCH：综合评分+偏好
            pref_brands = set()
            if user_profile and user_profile.brand_weights:
                pref_brands = set(user_profile.brand_weights.keys())
            return max(
                affordable,
                key=lambda p: p.rating + (0.5 if p.brand in pref_brands else 0.0),
            )
        elif action_id == 1:  # SELECT_BUDGET_OPT：性价比（评分/价格）
            return max(
                affordable,
                key=lambda p: p.rating / max(p.final_price, 1.0),
            )
        else:  # SELECT_SAFE：高评分+充足库存
            return max(affordable, key=lambda p: p.rating)

    def _action_desc(self, action: PlanningAction) -> str:
        descs = {0: "最佳匹配", 1: "性价比优先", 2: "保守安全"}
        return descs.get(action.action_id, "未知动作")

    # ------------------------------------------------------------------
    # 启发式规划（原有逻辑，向后兼容）
    # ------------------------------------------------------------------

    def _extract_slot_candidates(
        self,
        task: ShoppingTask,
        candidate_graph: dict[str, list[dict[str, Any]]],
        budget: Optional[float],
    ) -> dict[str, list[Product]]:
        """
        从候选图的节点中提取各品类坑位的候选商品。
        生产环境应维护 product_id → Product 的索引。
        此处简化返回空结构，由 _enumerate_plans 用 stub 填充。
        """
        return {cat: [] for cat in task.categories}

    def _enumerate_plans(
        self,
        task: ShoppingTask,
        slot_candidates: dict[str, list[Product]],
        budget: Optional[float],
        user_profile: Optional[UserProfile],
    ) -> list[CandidatePlan]:
        """枚举三档方案（主推/省钱/升级）。"""
        plans = []
        for i in range(MAX_PLANS_OUTPUT):
            plan = self._build_mock_plan(task, budget, user_profile, tier=i)
            if plan:
                plans.append(plan)
        return plans

    def _build_mock_plan(
        self,
        task: ShoppingTask,
        budget: Optional[float],
        user_profile: Optional[UserProfile],
        tier: int = 0,
    ) -> Optional[CandidatePlan]:
        """构建 stub 方案（占位实现；生产环境从真实候选组合中选优）。"""
        import random
        from shopping_agent.common.types import LogisticsInfo, ProductAttribute

        budget = budget or 5000.0
        tier_multipliers = [1.0, 0.85, 1.15]
        tier_multiplier = tier_multipliers[tier % 3]
        tier_names = ["主推方案", "省钱方案", "升级方案"]

        items = []
        total = 0.0
        per_item_budget = (budget * tier_multiplier) / max(len(task.categories), 1)

        for cat in task.categories:
            brand = (
                list(user_profile.brand_weights.keys())[0]
                if (user_profile and user_profile.brand_weights)
                else "Sony"
            )
            price = round(per_item_budget * random.uniform(0.8, 0.95), 2)
            total += price

            product = self._make_stub_product(cat, per_item_budget, user_profile, tier)
            items.append(PlanItem(
                bundle_slot=cat,
                product=product,
                reason=f"综合评分高，{tier_names[tier % 3]}性价比",
            ))

        if total > budget * 1.05:
            return None

        return self._build_plan_from_items(task, items, budget, tier_names[tier % 3])

    def _make_stub_product(
        self,
        category: str,
        budget_per_slot: float,
        user_profile: Optional[UserProfile],
        tier: int = 0,
    ) -> Product:
        """生成 stub 商品（无真实候选时使用）。"""
        import random
        from shopping_agent.common.types import LogisticsInfo, ProductAttribute

        tier_names = ["主推方案", "省钱方案", "升级方案"]
        brand = (
            list(user_profile.brand_weights.keys())[0]
            if (user_profile and user_profile.brand_weights)
            else "Sony"
        )
        price = round(budget_per_slot * random.uniform(0.8, 0.95), 2)

        return Product(
            product_id=str(uuid.uuid4()),
            title=f"{brand} {category} {tier_names[tier % 3]} 推荐款",
            platform="JD",
            price=price,
            original_price=round(price * 1.1, 2),
            coupon_discount=round(price * 0.03, 2),
            category=category,
            brand=brand,
            attributes=[ProductAttribute(name="颜色", value="黑色")],
            in_stock=True,
            logistics=LogisticsInfo(
                platform="JD",
                delivery_days=2 if tier == 0 else 3,
                delivery_fee=0.0,
                supports_return=True,
                return_days=7,
                is_official_store=True,
            ),
            rating=round(random.uniform(4.3, 4.9), 1),
            review_count=random.randint(1000, 30000),
            sales_volume=random.randint(5000, 100000),
        )

    def _build_plan_from_items(
        self,
        task: ShoppingTask,
        items: list[PlanItem],
        budget: float,
        tier_name: str = "推荐方案",
    ) -> Optional[CandidatePlan]:
        """从 items 构建 CandidatePlan。"""
        total = sum(item.product.final_price for item in items)
        if total > budget * 1.05:
            return None

        constraint_score = 1.0 if total <= budget else max(0.0, 1.0 - (total - budget) / budget)
        preference_score = 0.75
        value_score = 0.8

        tradeoff_notes = []
        if "省钱" in tier_name:
            tradeoff_notes.append(TradeoffNote(
                dimension="price", note="比主推方案更省钱，但部分商品评分略低", severity="info"
            ))
        elif "升级" in tier_name:
            tradeoff_notes.append(TradeoffNote(
                dimension="performance", note="升级配置，预算超出约15%，性能更强", severity="warning"
            ))

        return CandidatePlan(
            plan_id=str(uuid.uuid4()),
            task_id=task.task_id,
            items=items,
            total_price=round(total, 2),
            total_discount=round(total * 0.03, 2),
            constraint_score=constraint_score,
            preference_score=preference_score,
            value_score=value_score,
            tradeoff_notes=tradeoff_notes,
            explanation=f"这是{tier_name}，综合考虑了您的预算和品类需求。",
        )
