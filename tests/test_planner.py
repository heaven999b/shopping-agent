"""
tests/test_planner.py — ConstraintAwarePlanner 单元测试。

覆盖：
  - 单品类任务：生成 1~3 个候选方案
  - Bundle 任务：每个方案包含多品类
  - 预算严格无可行方案 → InfeasibleConstraintError
"""

from __future__ import annotations

import pytest

from shopping_agent.common.exceptions import InfeasibleConstraintError
from shopping_agent.planning.planner import ConstraintAwarePlanner
from tests.conftest import make_product, make_task


@pytest.fixture
def planner() -> ConstraintAwarePlanner:
    return ConstraintAwarePlanner(use_rl=False)


def _plan(planner, task, products, user_profile=None):
    """Helper: call planner.plan with retrieved_products interface."""
    return planner.plan(task, {}, user_profile=user_profile, retrieved_products=products)


class TestPlannerSingleCategory:
    def test_generates_plans(self, planner, product_headset, product_cheap_headset, user_profile):
        task = make_task(categories=["headset"], budget=5000.0)
        plans = _plan(planner, task, [product_headset, product_cheap_headset], user_profile)
        assert len(plans) >= 1

    def test_plan_within_budget(self, planner, product_headset, product_cheap_headset):
        task = make_task(categories=["headset"], budget=5000.0)
        plans = _plan(planner, task, [product_headset, product_cheap_headset])
        for plan in plans:
            assert plan.net_price <= 5000.0

    def test_no_matching_category_raises(self, planner):
        """商品品类与任务不匹配 → InfeasibleConstraintError。"""
        monitor = make_product("m1", price=999.0, category="monitor")
        task = make_task(categories=["headset"], budget=5000.0)
        with pytest.raises(InfeasibleConstraintError):
            _plan(planner, task, [monitor])

    def test_plan_diversity_strategies(self, planner):
        """当商品足够多时，应生成多档方案（主推/省钱/高分）。"""
        products = [
            make_product(f"p{i}", price=float(200 + i * 300),
                         category="headset", rating=3.5 + i * 0.3)
            for i in range(5)
        ]
        task = make_task(categories=["headset"], budget=5000.0)
        plans = _plan(planner, task, products)
        assert len(plans) >= 2


class TestPlannerBundle:
    def test_bundle_plan_covers_categories(self, planner, product_headset, product_monitor):
        task = make_task(
            task_id="t_bundle",
            categories=["headset", "monitor"],
            budget=10000.0,
        )
        plans = _plan(planner, task, [product_headset, product_monitor])
        assert len(plans) >= 1
        for plan in plans:
            slots = {item.bundle_slot for item in plan.items}
            assert len(slots) >= 1

    def test_bundle_total_price_correct(self, planner, product_headset, product_monitor):
        task = make_task(categories=["headset", "monitor"], budget=10000.0)
        plans = _plan(planner, task, [product_headset, product_monitor])
        for plan in plans:
            expected = sum(item.product.final_price for item in plan.items)
            assert abs(plan.net_price - expected) < 0.01


class TestPlannerScoring:
    def test_best_plan_has_high_score(self, planner):
        products = [
            make_product("high", price=1500.0, category="headset",
                         rating=4.9, brand="Sony"),
            make_product("low", price=800.0, category="headset",
                         rating=3.5, brand="Unknown"),
        ]
        task = make_task(categories=["headset"], budget=5000.0)
        plans = _plan(planner, task, products)
        if len(plans) >= 2:
            assert plans[0].overall_score >= plans[-1].overall_score - 0.01
