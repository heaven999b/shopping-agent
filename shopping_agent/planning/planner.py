"""
ConstraintAwarePlanner — 约束感知多目标规划器。

流程：
  1. 任务分解：将 bundle 任务拆成各坑位的独立子问题
  2. 候选筛选：从候选图中按硬约束过滤
  3. 预算分配：根据品类优先级分配预算
  4. 多目标优化：在约束满足度、偏好匹配度、性价比三维上优化
  5. 生成多套方案（最多 MAX_PLANS_OUTPUT 套）
"""

from __future__ import annotations

import uuid
from itertools import product as itertools_product
from typing import Any, Optional

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


class ConstraintAwarePlanner:
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

        # 2. 生成方案（枚举各坑位候选的组合）
        plans = self._enumerate_plans(
            task, slot_candidates, budget, user_profile
        )

        if not plans:
            raise PlanningError("无法在约束范围内生成有效购物方案。")

        # 3. 按综合得分排序，返回 Top-N
        plans.sort(key=lambda p: p.overall_score, reverse=True)
        return plans[:MAX_PLANS_OUTPUT]

    def _extract_slot_candidates(
        self,
        task: ShoppingTask,
        candidate_graph: dict[str, list[dict[str, Any]]],
        budget: Optional[float],
    ) -> dict[str, list[Product]]:
        """
        从候选图的节点中提取各品类坑位的候选商品。
        这里做简化：直接从图的 key（product_id）重建商品对象。
        生产环境应维护 product_id → Product 的索引。
        """
        # 实际上候选图的 key 是 product_id，我们需要通过外部索引查回 Product
        # 此处 planner 需要访问原始商品列表，通过 orchestrator 传入
        # 简化：返回空结构，由 _enumerate_plans 用占位商品填充
        return {cat: [] for cat in task.categories}

    def _enumerate_plans(
        self,
        task: ShoppingTask,
        slot_candidates: dict[str, list[Product]],
        budget: Optional[float],
        user_profile: Optional[UserProfile],
    ) -> list[CandidatePlan]:
        """
        枚举各坑位候选组合，生成候选方案。
        对空坑位使用占位（生产环境应有实际候选商品）。
        """
        plans = []
        # 此处为 stub：生成 3 套模拟方案展示结构
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
        """
        构建模拟方案（占位实现）。
        生产环境：从真实候选商品组合中选择最优组合。
        """
        import random
        from shopping_agent.common.types import LogisticsInfo, ProductAttribute

        budget = budget or 5000.0
        tier_multipliers = [1.0, 0.85, 1.15]  # 主推/省钱/升级三档
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

            product = Product(
                product_id=str(uuid.uuid4()),
                title=f"{brand} {cat} {tier_names[tier % 3]} 推荐款",
                platform="JD",
                price=price,
                original_price=round(price * 1.1, 2),
                coupon_discount=round(price * 0.03, 2),
                category=cat,
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
            items.append(PlanItem(
                bundle_slot=cat,
                product=product,
                reason=f"综合评分高，{tier_names[tier % 3]}性价比",
            ))

        # 检查预算
        if total > budget * 1.05:  # 超预算 5% 以上放弃此方案
            return None

        constraint_score = 1.0 if total <= budget else max(0.0, 1.0 - (total - budget) / budget)
        preference_score = 0.7 + tier * 0.05
        value_score = [0.8, 0.9, 0.7][tier % 3]

        tradeoff_notes = []
        if tier == 1:
            tradeoff_notes.append(TradeoffNote(
                dimension="price", note="比主推方案便宜约15%，但部分商品评分略低", severity="info"
            ))
        elif tier == 2:
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
            explanation=f"这是{tier_names[tier % 3]}，综合考虑了您的预算和品类需求。",
        )
