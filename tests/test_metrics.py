"""
tests/test_metrics.py — MetricsComputer 单元测试。

覆盖：
  - 空结果列表
  - success_rate 计算
  - avg_plan_diversity 计算
  - graph_recovery_rate 计算
  - avg_constraint_hit_rate 计算
"""

from __future__ import annotations

import pytest

from shopping_agent.evaluation.metrics import MetricsComputer, TaskResult


def make_result(
    task_id: str = "t001",
    success: bool = True,
    has_result: bool = True,
    budget_satisfied: bool = True,
    constraint_hit_rate: float = 1.0,
    overall_score: float = 0.8,
    plan_diversity: float = 0.1,
    graph_recovery_used: bool = False,
    clarification_turns: int = 0,
    latency_ms: float = 100.0,
    budget_ratio: float = 0.8,
    parser_category_match: float = 1.0,
    plan_category_match: float = 1.0,
    clarification_alignment: float = 1.0,
    feasibility_alignment: float = 1.0,
    intent_resolution_score: float = 1.0,
    execution_readiness_score: float = 1.0,
    phase_coverage_score: float = 1.0,
    failure_bucket: str | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        query="test query",
        success=success,
        has_result=has_result,
        budget_satisfied=budget_satisfied,
        constraint_hit_rate=constraint_hit_rate,
        overall_score=overall_score,
        plan_diversity=plan_diversity,
        graph_recovery_used=graph_recovery_used,
        clarification_turns=clarification_turns,
        latency_ms=latency_ms,
        budget_ratio=budget_ratio,
        parser_category_match=parser_category_match,
        plan_category_match=plan_category_match,
        clarification_alignment=clarification_alignment,
        feasibility_alignment=feasibility_alignment,
        intent_resolution_score=intent_resolution_score,
        execution_readiness_score=execution_readiness_score,
        phase_coverage_score=phase_coverage_score,
        failure_bucket=failure_bucket,
    )


class TestMetricsComputer:
    computer = MetricsComputer()

    def test_empty_results(self):
        metrics = self.computer.compute([])
        assert metrics["num_tasks"] == 0
        assert "success_rate" not in metrics  # 空时只有 num_tasks

    def test_success_rate(self):
        results = [
            make_result("t1", success=True),
            make_result("t2", success=True),
            make_result("t3", success=False),
        ]
        metrics = self.computer.compute(results)
        assert metrics["num_tasks"] == 3
        assert metrics["success_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_constraint_hit_rate(self):
        results = [
            make_result("t1", constraint_hit_rate=1.0),
            make_result("t2", constraint_hit_rate=0.5),
        ]
        metrics = self.computer.compute(results)
        assert metrics["avg_constraint_hit_rate"] == pytest.approx(0.75, abs=1e-4)

    def test_plan_diversity(self):
        results = [
            make_result("t1", plan_diversity=0.2),
            make_result("t2", plan_diversity=0.4),
        ]
        metrics = self.computer.compute(results)
        assert metrics["avg_plan_diversity"] == pytest.approx(0.3, abs=1e-4)

    def test_graph_recovery_rate(self):
        results = [
            make_result("t1", graph_recovery_used=True),
            make_result("t2", graph_recovery_used=False),
            make_result("t3", graph_recovery_used=True),
        ]
        metrics = self.computer.compute(results)
        assert metrics["graph_recovery_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_all_failed(self):
        results = [make_result(f"t{i}", success=False) for i in range(5)]
        metrics = self.computer.compute(results)
        assert metrics["success_rate"] == 0.0

    def test_all_succeeded(self):
        results = [make_result(f"t{i}", success=True) for i in range(5)]
        metrics = self.computer.compute(results)
        assert metrics["success_rate"] == 1.0

    def test_compare(self):
        baseline = self.computer.compute([make_result("t1", success=True, overall_score=0.7)])
        experiment = self.computer.compute([make_result("t2", success=True, overall_score=0.9)])
        comparison = self.computer.compare(baseline, experiment)
        assert "success_rate" in comparison
        assert comparison["avg_overall_score"]["delta"] == pytest.approx(0.2, abs=1e-3)

    def test_diagnostic_metrics(self):
        results = [
            make_result(
                "t1",
                parser_category_match=1.0,
                plan_category_match=0.5,
                clarification_alignment=1.0,
                feasibility_alignment=1.0,
                intent_resolution_score=0.9,
                execution_readiness_score=0.7,
                phase_coverage_score=0.8,
                failure_bucket="success",
            ),
            make_result(
                "t2",
                parser_category_match=0.5,
                plan_category_match=1.0,
                clarification_alignment=0.0,
                feasibility_alignment=1.0,
                intent_resolution_score=0.4,
                execution_readiness_score=0.8,
                phase_coverage_score=0.6,
                failure_bucket="clarification_failure",
            ),
        ]
        metrics = self.computer.compute(results)
        assert metrics["avg_parser_category_match"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["avg_plan_category_match"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["clarification_alignment_rate"] == pytest.approx(0.5, abs=1e-4)
        assert metrics["avg_intent_resolution_score"] == pytest.approx(0.65, abs=1e-4)
        assert metrics["avg_execution_readiness_score"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["failure_bucket_breakdown"]["clarification_failure"] == 1
