"""
tests/test_planner.py — ConstraintAwarePlanner 单元测试。

覆盖：
  - 单品类任务：生成 1~3 个候选方案
  - Bundle 任务：每个方案包含多品类
  - 预算严格无可行方案 → InfeasibleConstraintError
"""

from __future__ import annotations

import pytest

from shopping_agent.common.types import BundlePlan, PlanItem, TaskType
from shopping_agent.common.exceptions import InfeasibleConstraintError
from shopping_agent.planning.explainer import Explainer
from shopping_agent.planning.bundle_planner import BundlePlanner
from shopping_agent.planning.planner import ConstraintAwarePlanner
from tests.conftest import make_product, make_task


@pytest.fixture
def planner() -> ConstraintAwarePlanner:
    return BundlePlanner(use_rl=False)


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

    def test_bundle_plan_emits_bundle_metadata(self, planner, product_headset, product_monitor, user_profile):
        product_headset.persona_tags = {
            "style_signal": ["clean", "premium"],
            "identity_fit": ["professional"],
            "symbolic_value": "taste_signaling",
        }
        product_monitor.persona_tags = {
            "style_signal": ["clean", "premium"],
            "identity_fit": ["professional"],
            "symbolic_value": "taste_signaling",
        }
        task = make_task(
            task_id="t_bundle_meta",
            task_type=TaskType.BUNDLE,
            categories=["headset", "monitor"],
            budget=6000.0,
            raw_query="帮我配一套办公室桌搭，耳机和显示器都要",
        )
        task.implicit_needs = ["办公桌搭", "商务"]
        plans = _plan(planner, task, [product_headset, product_monitor], user_profile)

        assert plans[0].bundle_type == "bundle_plan"
        assert isinstance(plans[0], BundlePlan)
        assert plans[0].bundle_objective
        assert set(plans[0].budget_allocation.keys()) == {"headset", "monitor"}
        assert plans[0].style_coherence_score > 0.5
        assert plans[0].bundle_completeness_score == pytest.approx(1.0, abs=1e-4)
        assert plans[0].required_slots == ["headset", "monitor"]
        assert plans[0].filled_slots == ["headset", "monitor"]
        assert plans[0].slot_coverage == {"headset": True, "monitor": True}
        assert plans[0].compatibility_score >= 0.8
        assert plans[0].relation_coverage_score >= 0.25
        assert plans[0].bundle_decision_score > 0.45
        assert plans[0].overall_score == plans[0].bundle_decision_score
        assert plans[0].phased_upgrade_plan == plans[0].phased_purchase_options
        assert plans[0].phased_purchase_options

    def test_bundle_compatibility_uses_relation_graph(self, planner, user_profile):
        keyboard = make_product("k1", title="办公键盘", category="keyboard", price=699.0, brand="Logitech")
        mouse = make_product("m1", title="办公鼠标", category="mouse", price=399.0, brand="Logitech")
        keyboard.persona_tags = {"style_signal": ["clean"], "identity_fit": ["professional"]}
        mouse.persona_tags = {"style_signal": ["clean"], "identity_fit": ["professional"]}
        task = make_task(
            task_id="t_bundle_graph",
            task_type=TaskType.BUNDLE,
            categories=["keyboard", "mouse"],
            budget=3000.0,
            raw_query="帮我配一套办公键鼠",
        )
        graph = {
            "k1": [
                {"target_id": "m1", "edge_type": "COMPLEMENT", "weight": 0.9},
                {"target_id": "m1", "edge_type": "STYLE_MATCH", "weight": 0.8},
            ],
            "m1": [
                {"target_id": "k1", "edge_type": "COMPLEMENT", "weight": 0.9},
            ],
        }

        plans = planner.plan(task, graph, user_profile=user_profile, retrieved_products=[keyboard, mouse])

        assert plans[0].compatibility_score > 0.85
        assert plans[0].relation_coverage_score > 0.75
        assert plans[0].bundle_decision_score > 0.5

    def test_partial_bundle_is_penalized_by_relation_and_coverage(self, planner):
        monitor = make_product("m1", title="办公显示器", category="monitor", price=1999.0, brand="LG")
        lamp = make_product("l1", title="桌面台灯", category="desk_lamp", price=499.0, brand="BenQ")
        monitor.persona_tags = {"style_signal": ["clean"], "identity_fit": ["professional"]}
        lamp.persona_tags = {"style_signal": ["clean"], "identity_fit": ["professional"]}
        task = make_task(
            task_id="t_partial_bundle",
            task_type=TaskType.BUNDLE,
            categories=["monitor", "desk_lamp", "headset"],
            budget=6000.0,
            raw_query="帮我配一套桌搭，显示器、灯和耳机都要",
        )

        partial_plan = planner._assemble_plan(
            task,
            [
                PlanItem(bundle_slot="monitor", product=monitor, reason="partial"),
                PlanItem(bundle_slot="desk_lamp", product=lamp, reason="partial"),
            ],
            candidate_graph={},
            budget=6000.0,
            tier_name="partial",
        )

        assert partial_plan is not None
        assert partial_plan.bundle_completeness_score == pytest.approx(2 / 3, abs=1e-3)
        assert partial_plan.compatibility_score <= 0.8
        assert partial_plan.relation_coverage_score == pytest.approx(0.25, abs=1e-4)
        assert partial_plan.bundle_decision_score < 0.4



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

    def test_persona_alignment_prefers_professional_route(self, planner, user_profile):
        professional = make_product(
            "pro",
            title="索尼 专业办公降噪耳机",
            price=1799.0,
            category="headset",
            brand="Sony",
            rating=4.7,
        )
        professional.persona_tags = {
            "identity_fit": ["professional"],
            "style_signal": ["clean", "premium"],
            "symbolic_value": "taste_signaling",
        }

        playful = make_product(
            "fun",
            title="潮流炫彩电竞耳机",
            price=1699.0,
            category="headset",
            brand="Unknown",
            rating=4.8,
        )
        playful.persona_tags = {
            "identity_fit": ["gamer"],
            "style_signal": ["playful"],
            "symbolic_value": "utilitarian",
        }

        task = make_task(categories=["headset"], budget=5000.0)
        plans = _plan(planner, task, [professional, playful], user_profile)

        assert plans[0].items[0].product.product_id == "pro"
        assert plans[0].persona_alignment_score > 0.3
        assert "professional" in plans[0].persona_summary or "clean" in plans[0].persona_summary


class TestExplainerPersona:
    def test_explainer_mentions_persona_fit(self, planner, user_profile):
        explainer = Explainer()
        product = make_product(
            "pro",
            title="索尼 专业办公降噪耳机",
            price=1799.0,
            category="headset",
            brand="Sony",
            rating=4.7,
        )
        product.persona_tags = {
            "identity_fit": ["professional"],
            "style_signal": ["clean", "premium"],
            "symbolic_value": "taste_signaling",
        }

        task = make_task(categories=["headset"], budget=5000.0)
        plan = _plan(planner, task, [product], user_profile)[0]
        text = explainer.explain(plan, task, style_hint="detailed")

        assert "画像" in text
        assert "professional" in text or "clean" in text

    def test_explainer_mentions_bundle_budget_and_phases(self, planner, product_headset, product_monitor, user_profile):
        explainer = Explainer()
        product_headset.persona_tags = {
            "style_signal": ["clean", "premium"],
            "identity_fit": ["professional"],
            "symbolic_value": "taste_signaling",
        }
        product_monitor.persona_tags = {
            "style_signal": ["clean", "premium"],
            "identity_fit": ["professional"],
            "symbolic_value": "taste_signaling",
        }
        task = make_task(
            task_type=TaskType.BUNDLE,
            categories=["headset", "monitor"],
            budget=6000.0,
            raw_query="帮我配一套办公桌搭",
        )
        task.implicit_needs = ["办公"]
        plan = _plan(planner, task, [product_headset, product_monitor], user_profile)[0]
        text = explainer.explain(plan, task, style_hint="detailed")

        assert "预算分配" in text
        assert "分阶段购买建议" in text
