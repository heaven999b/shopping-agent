"""
tests/test_verifier.py — VerificationPipeline 单元测试。

覆盖：
  - BudgetChecker: PASS / WARN / BLOCK
  - ConstraintChecker: 缺货 BLOCK、时效 BLOCK、全通过 PASS
  - FreshnessChecker: 过期商品 STALE
  - RiskChecker: 高金额 WARN
  - VerificationPipeline: 汇总 passed 逻辑
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from shopping_agent.common.types import VerificationStatus
from shopping_agent.verifier.pipeline import (
    BudgetChecker,
    ConstraintChecker,
    FreshnessChecker,
    RiskChecker,
    VerificationPipeline,
)
from tests.conftest import make_plan, make_product, make_task


class TestBudgetChecker:
    checker = BudgetChecker()

    def test_pass_within_budget(self):
        plan = make_plan(products=[make_product(price=1000.0)])
        task = make_task(budget=2000.0)
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.PASS

    def test_warn_slightly_over(self):
        """超出 3%（在 5% 警告区间内）→ WARN。"""
        plan = make_plan(products=[make_product(price=2060.0)], total_price=2060.0)
        task = make_task(budget=2000.0)
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.WARN

    def test_block_over_budget(self):
        """超出 10% → BLOCK。"""
        plan = make_plan(products=[make_product(price=2200.0)], total_price=2200.0)
        task = make_task(budget=2000.0)
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.BLOCK

    def test_no_budget_constraint(self):
        """无预算约束 → PASS。"""
        plan = make_plan(products=[make_product(price=99999.0)], total_price=99999.0)
        task = make_task(budget=None)
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.PASS


class TestConstraintChecker:
    checker = ConstraintChecker()

    def test_out_of_stock_block(self, product_out_of_stock):
        plan = make_plan(products=[product_out_of_stock])
        task = make_task(budget=5000.0)
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.BLOCK
        assert "无货" in result.message

    def test_all_good(self, product_headset):
        plan = make_plan(products=[product_headset])
        task = make_task(budget=5000.0)
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.PASS


class TestFreshnessChecker:
    checker = FreshnessChecker()

    def test_fresh_product_pass(self, product_headset):
        plan = make_plan(products=[product_headset])
        task = make_task()
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.PASS

    def test_stale_product(self):
        stale = make_product(ttl_seconds=1)
        # 手动让 fetched_at 过期
        stale.fetched_at = datetime.now() - timedelta(seconds=60)
        plan = make_plan(products=[stale])
        task = make_task()
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.STALE


class TestRiskChecker:
    checker = RiskChecker()

    def test_high_amount_warn(self):
        from shopping_agent.common.constants import HIGH_RISK_AMOUNT_THRESHOLD
        expensive = make_product(price=HIGH_RISK_AMOUNT_THRESHOLD + 1)
        plan = make_plan(products=[expensive], total_price=HIGH_RISK_AMOUNT_THRESHOLD + 1)
        task = make_task()
        result = self.checker.check(plan, task)
        assert result.status == VerificationStatus.WARN

    def test_normal_amount_pass(self, product_headset):
        plan = make_plan(products=[product_headset])
        task = make_task()
        result = self.checker.check(plan, task)
        # 1999 应该远低于高风险阈值
        assert result.status == VerificationStatus.PASS


class TestVerificationPipeline:
    pipeline = VerificationPipeline()

    def test_full_pass(self, product_headset):
        plan = make_plan(products=[product_headset])
        task = make_task(budget=5000.0)
        report = self.pipeline.verify(plan, task, parallel=False)
        assert report.passed is True
        assert len(report.blocking_issues) == 0

    def test_block_over_budget(self):
        expensive = make_product(price=5000.0)
        plan = make_plan(products=[expensive], total_price=5000.0)
        task = make_task(budget=2000.0)
        report = self.pipeline.verify(plan, task, parallel=False)
        assert report.passed is False
        assert len(report.blocking_issues) >= 1

    def test_parallel_vs_serial_same_result(self, product_headset):
        plan_s = make_plan(plan_id="s", products=[product_headset])
        plan_p = make_plan(plan_id="p", products=[product_headset])
        task = make_task(budget=5000.0)
        report_s = self.pipeline.verify(plan_s, task, parallel=False)
        report_p = self.pipeline.verify(plan_p, task, parallel=True)
        assert report_s.passed == report_p.passed
