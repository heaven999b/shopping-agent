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
    plan_persona_alignment_score: float = 0.0,
    persona_reason_coverage: float = 0.0,
    style_coherence_score: float = 0.0,
    bundle_completeness_score: float = 0.0,
    compatibility_score: float = 0.0,
    relation_coverage_score: float = 0.0,
    bundle_decision_score: float = 0.0,
    long_term_fit_score: float = 0.0,
    phased_purchase_score: float = 0.0,
    drift_expected: bool = False,
    drift_detected: bool = False,
    drift_alignment_score: float = 0.0,
    failure_bucket: str | None = None,
    task_family: str = "general",
    bundle_success: bool | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        query="test query",
        task_family=task_family,
        original_task_family=task_family,
        success=success,
        bundle_success=success if bundle_success is None else bundle_success,
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
        plan_persona_alignment_score=plan_persona_alignment_score,
        persona_reason_coverage=persona_reason_coverage,
        style_coherence_score=style_coherence_score,
        bundle_completeness_score=bundle_completeness_score,
        compatibility_score=compatibility_score,
        relation_coverage_score=relation_coverage_score,
        bundle_decision_score=bundle_decision_score,
        long_term_fit_score=long_term_fit_score,
        phased_purchase_score=phased_purchase_score,
        drift_expected=drift_expected,
        drift_detected=drift_detected,
        drift_alignment_score=drift_alignment_score,
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
                task_family="bundle",
                parser_category_match=1.0,
                plan_category_match=0.5,
                clarification_alignment=1.0,
                feasibility_alignment=1.0,
                intent_resolution_score=0.9,
                execution_readiness_score=0.7,
                phase_coverage_score=0.8,
                plan_persona_alignment_score=0.6,
                persona_reason_coverage=1.0,
                style_coherence_score=0.9,
                bundle_completeness_score=1.0,
                compatibility_score=0.8,
                relation_coverage_score=0.7,
                bundle_decision_score=0.75,
                long_term_fit_score=0.8,
                phased_purchase_score=0.7,
                drift_expected=True,
                drift_detected=True,
                drift_alignment_score=1.0,
                failure_bucket="success",
            ),
            make_result(
                "t2",
                task_family="phased_purchase",
                parser_category_match=0.5,
                plan_category_match=1.0,
                clarification_alignment=0.0,
                feasibility_alignment=1.0,
                intent_resolution_score=0.4,
                execution_readiness_score=0.8,
                phase_coverage_score=0.6,
                plan_persona_alignment_score=0.2,
                persona_reason_coverage=0.5,
                style_coherence_score=0.4,
                bundle_completeness_score=0.6,
                compatibility_score=0.5,
                relation_coverage_score=0.2,
                bundle_decision_score=0.45,
                long_term_fit_score=0.5,
                phased_purchase_score=0.4,
                drift_expected=False,
                drift_detected=False,
                drift_alignment_score=1.0,
                failure_bucket="clarification_failure",
            ),
        ]
        metrics = self.computer.compute(results)
        assert metrics["avg_parser_category_match"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["avg_plan_category_match"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["clarification_alignment_rate"] == pytest.approx(0.5, abs=1e-4)
        assert metrics["avg_intent_resolution_score"] == pytest.approx(0.65, abs=1e-4)
        assert metrics["avg_execution_readiness_score"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["avg_plan_persona_alignment_score"] == pytest.approx(0.4, abs=1e-4)
        assert metrics["avg_persona_reason_coverage"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["avg_style_coherence_score"] == pytest.approx(0.65, abs=1e-4)
        assert metrics["avg_bundle_completeness_score"] == pytest.approx(0.8, abs=1e-4)
        assert metrics["avg_compatibility_score"] == pytest.approx(0.65, abs=1e-4)
        assert metrics["avg_relation_coverage_score"] == pytest.approx(0.45, abs=1e-4)
        assert metrics["avg_bundle_decision_score"] == pytest.approx(0.6, abs=1e-4)
        assert metrics["avg_long_term_fit_score"] == pytest.approx(0.65, abs=1e-4)
        assert metrics["avg_phased_purchase_score"] == pytest.approx(0.55, abs=1e-4)
        assert metrics["drift_detection_rate"] == pytest.approx(0.5, abs=1e-4)
        assert metrics["avg_drift_alignment_score"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["failure_bucket_breakdown"]["clarification_failure"] == 1
        assert metrics["task_family_summary"]["bundle"]["num_tasks"] == 1
        assert metrics["bundle_summary"]["num_tasks"] == 2
        assert metrics["bundle_summary"]["bundle_success_rate"] == pytest.approx(1.0, abs=1e-4)

    def test_bundle_summary_uses_original_task_family(self):
        results = [
            make_result(
                "t1",
                task_family="single",
                bundle_success=True,
            ),
            make_result(
                "t2",
                task_family="single",
                bundle_success=False,
            ),
        ]
        results[0].original_task_family = "bundle"
        results[1].original_task_family = "upgrade_path"

        metrics = self.computer.compute(results)

        assert metrics["bundle_summary"]["num_tasks"] == 2
        assert metrics["bundle_summary"]["bundle_success_rate"] == pytest.approx(0.5, abs=1e-4)
