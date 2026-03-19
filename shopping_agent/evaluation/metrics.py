"""
评估指标计算模块。

定义了 TaskResult（单次任务执行结果）和 MetricsComputer（指标计算器）。

核心指标：
  - success_rate          任务成功率（有推荐结果且预算满足）
  - budget_satisfaction   预算满足率（推荐总价 ≤ 预算）
  - constraint_hit_rate   硬约束命中率（推荐商品满足必要属性的比例）
  - avg_overall_score     平均综合得分
  - avg_turns             平均澄清轮数
  - coverage              有推荐结果的任务占比（系统覆盖率）
  - plan_diversity        方案间价格/得分的变异系数（越高越好）
  - revision_rate         需要图修复的方案比例（越低越好）
  - graph_recovery_rate   图修复成功比例（有修复尝试时）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskResult:
    """
    单次任务执行结果，由 BenchmarkRunner 填写。
    """
    task_id: str
    query: str

    # 执行状态
    success: bool = False           # 有推荐结果且满足预算
    has_result: bool = False        # 是否有任何推荐结果（覆盖率）
    error: Optional[str] = None     # 若抛异常记录错误信息

    # 方案指标
    budget_satisfied: bool = False  # 总价 ≤ 预算
    budget_ratio: float = 0.0       # total_price / budget（越接近 1 越好）
    constraint_hit_rate: float = 0.0  # 必要属性命中率
    overall_score: float = 0.0      # 最优方案综合得分
    num_plans: int = 0              # 生成方案数

    # 过程指标
    clarification_turns: int = 0    # 澄清轮数
    latency_ms: float = 0.0         # 端到端耗时
    workflow_steps_completed: int = 0

    # 方案多样性（多套方案时才有意义）
    plan_diversity: float = 0.0     # 各方案价格的变异系数（std/mean），0~1+
    plan_score_variance: float = 0.0  # 各方案综合得分方差

    # 图修复标记
    graph_recovery_used: bool = False  # 此任务是否触发了图修复
    verifier_skipped: bool = False     # 消融实验中跳过了 verifier

    # 诊断指标
    task_family: str = "general"
    parser_category_match: float = 0.0
    plan_category_match: float = 0.0
    clarification_expected: bool = False
    clarification_alignment: float = 0.0
    feasibility_expected: bool = True
    feasibility_alignment: float = 0.0
    intent_resolution_score: float = 0.0
    execution_readiness_score: float = 0.0
    phase_coverage_score: float = 0.0
    plan_persona_alignment_score: float = 0.0
    persona_reason_coverage: float = 0.0
    style_coherence_score: float = 0.0
    bundle_completeness_score: float = 0.0
    long_term_fit_score: float = 0.0
    phased_purchase_score: float = 0.0
    drift_expected: bool = False
    drift_detected: bool = False
    drift_alignment_score: float = 0.0
    ended_in_error: bool = False
    error_type: Optional[str] = None
    failure_bucket: Optional[str] = None

    # 方案明细（用于 per-item 分析）
    plan_items: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "success": self.success,
            "has_result": self.has_result,
            "error": self.error,
            "budget_satisfied": self.budget_satisfied,
            "budget_ratio": round(self.budget_ratio, 3),
            "constraint_hit_rate": round(self.constraint_hit_rate, 3),
            "overall_score": round(self.overall_score, 3),
            "num_plans": self.num_plans,
            "clarification_turns": self.clarification_turns,
            "latency_ms": round(self.latency_ms, 1),
            "workflow_steps_completed": self.workflow_steps_completed,
            "plan_diversity": round(self.plan_diversity, 4),
            "plan_score_variance": round(self.plan_score_variance, 4),
            "graph_recovery_used": self.graph_recovery_used,
            "parser_category_match": round(self.parser_category_match, 3),
            "task_family": self.task_family,
            "plan_category_match": round(self.plan_category_match, 3),
            "clarification_expected": self.clarification_expected,
            "clarification_alignment": round(self.clarification_alignment, 3),
            "feasibility_expected": self.feasibility_expected,
            "feasibility_alignment": round(self.feasibility_alignment, 3),
            "intent_resolution_score": round(self.intent_resolution_score, 3),
            "execution_readiness_score": round(self.execution_readiness_score, 3),
            "phase_coverage_score": round(self.phase_coverage_score, 3),
            "plan_persona_alignment_score": round(self.plan_persona_alignment_score, 3),
            "persona_reason_coverage": round(self.persona_reason_coverage, 3),
            "style_coherence_score": round(self.style_coherence_score, 3),
            "bundle_completeness_score": round(self.bundle_completeness_score, 3),
            "long_term_fit_score": round(self.long_term_fit_score, 3),
            "phased_purchase_score": round(self.phased_purchase_score, 3),
            "drift_expected": self.drift_expected,
            "drift_detected": self.drift_detected,
            "drift_alignment_score": round(self.drift_alignment_score, 3),
            "ended_in_error": self.ended_in_error,
            "error_type": self.error_type,
            "failure_bucket": self.failure_bucket,
        }


class MetricsComputer:
    """
    计算一批 TaskResult 的聚合指标。
    """

    def compute(self, results: list[TaskResult]) -> dict[str, Any]:
        """
        返回聚合指标 dict，包含：
          - 任务级统计
          - 质量指标均值
          - 过程指标均值
        """
        if not results:
            return {"num_tasks": 0}

        n = len(results)
        n_success = sum(1 for r in results if r.success)
        n_has_result = sum(1 for r in results if r.has_result)
        n_budget_ok = sum(1 for r in results if r.budget_satisfied)

        avg_constraint_hit = (
            sum(r.constraint_hit_rate for r in results) / n
        )
        avg_score = sum(r.overall_score for r in results) / n
        avg_turns = sum(r.clarification_turns for r in results) / n
        avg_latency = sum(r.latency_ms for r in results) / n
        avg_workflow_steps = sum(r.workflow_steps_completed for r in results) / n
        avg_parser_category_match = sum(r.parser_category_match for r in results) / n
        avg_plan_category_match = sum(r.plan_category_match for r in results) / n
        clarification_alignment_rate = sum(r.clarification_alignment for r in results) / n
        feasibility_alignment_rate = sum(r.feasibility_alignment for r in results) / n
        avg_intent_resolution = sum(r.intent_resolution_score for r in results) / n
        avg_execution_readiness = sum(r.execution_readiness_score for r in results) / n
        avg_phase_coverage = sum(r.phase_coverage_score for r in results) / n
        avg_plan_persona_alignment = sum(r.plan_persona_alignment_score for r in results) / n
        avg_persona_reason_coverage = sum(r.persona_reason_coverage for r in results) / n
        avg_style_coherence = sum(r.style_coherence_score for r in results) / n
        avg_bundle_completeness = sum(r.bundle_completeness_score for r in results) / n
        avg_long_term_fit = sum(r.long_term_fit_score for r in results) / n
        avg_phased_purchase = sum(r.phased_purchase_score for r in results) / n
        drift_detection_rate = sum(1 for r in results if r.drift_detected) / n
        avg_drift_alignment = sum(r.drift_alignment_score for r in results) / n

        # 预算利用率（仅计算有结果的任务）
        results_with_result = [r for r in results if r.has_result]
        avg_budget_ratio = (
            sum(r.budget_ratio for r in results_with_result) / len(results_with_result)
            if results_with_result else 0.0
        )

        # 方案多样性（取各任务的平均 plan_diversity）
        diversity_vals = [r.plan_diversity for r in results if r.plan_diversity > 0]
        avg_plan_diversity = sum(diversity_vals) / len(diversity_vals) if diversity_vals else 0.0

        # 图修复使用率
        n_graph_recovery = sum(1 for r in results if r.graph_recovery_used)
        graph_recovery_rate = n_graph_recovery / n

        # 错误分析
        errors = [r.error for r in results if r.error]
        error_rate = len(errors) / n
        ended_in_error_rate = sum(1 for r in results if r.ended_in_error) / n
        error_type_breakdown: dict[str, int] = {}
        failure_bucket_breakdown: dict[str, int] = {}
        task_family_summary: dict[str, dict[str, Any]] = {}
        for r in results:
            if r.error_type:
                error_type_breakdown[r.error_type] = error_type_breakdown.get(r.error_type, 0) + 1
            if r.failure_bucket:
                failure_bucket_breakdown[r.failure_bucket] = (
                    failure_bucket_breakdown.get(r.failure_bucket, 0) + 1
                )
            family_bucket = task_family_summary.setdefault(
                r.task_family,
                {
                    "num_tasks": 0,
                    "successes": 0,
                    "coverage_hits": 0,
                    "avg_overall_score_total": 0.0,
                    "avg_drift_alignment_total": 0.0,
                    "avg_bundle_completeness_total": 0.0,
                    "avg_long_term_fit_total": 0.0,
                },
            )
            family_bucket["num_tasks"] += 1
            family_bucket["successes"] += int(r.success)
            family_bucket["coverage_hits"] += int(r.has_result)
            family_bucket["avg_overall_score_total"] += r.overall_score
            family_bucket["avg_drift_alignment_total"] += r.drift_alignment_score
            family_bucket["avg_bundle_completeness_total"] += r.bundle_completeness_score
            family_bucket["avg_long_term_fit_total"] += r.long_term_fit_score

        for family, bucket in task_family_summary.items():
            count = max(1, bucket["num_tasks"])
            bucket["success_rate"] = round(bucket["successes"] / count, 4)
            bucket["coverage"] = round(bucket["coverage_hits"] / count, 4)
            bucket["avg_overall_score"] = round(bucket["avg_overall_score_total"] / count, 4)
            bucket["avg_drift_alignment_score"] = round(
                bucket["avg_drift_alignment_total"] / count, 4
            )
            bucket["avg_bundle_completeness_score"] = round(
                bucket["avg_bundle_completeness_total"] / count, 4
            )
            bucket["avg_long_term_fit_score"] = round(
                bucket["avg_long_term_fit_total"] / count, 4
            )
            del bucket["successes"]
            del bucket["coverage_hits"]
            del bucket["avg_overall_score_total"]
            del bucket["avg_drift_alignment_total"]
            del bucket["avg_bundle_completeness_total"]
            del bucket["avg_long_term_fit_total"]

        return {
            # 基础统计
            "num_tasks": n,
            "num_success": n_success,
            "num_has_result": n_has_result,
            # 核心指标
            "success_rate": round(n_success / n, 4),
            "coverage": round(n_has_result / n, 4),
            "budget_satisfaction_rate": round(n_budget_ok / n, 4),
            "avg_constraint_hit_rate": round(avg_constraint_hit, 4),
            "avg_overall_score": round(avg_score, 4),
            "avg_parser_category_match": round(avg_parser_category_match, 4),
            "avg_plan_category_match": round(avg_plan_category_match, 4),
            "clarification_alignment_rate": round(clarification_alignment_rate, 4),
            "feasibility_alignment_rate": round(feasibility_alignment_rate, 4),
            "avg_intent_resolution_score": round(avg_intent_resolution, 4),
            "avg_execution_readiness_score": round(avg_execution_readiness, 4),
            "avg_phase_coverage_score": round(avg_phase_coverage, 4),
            "avg_plan_persona_alignment_score": round(avg_plan_persona_alignment, 4),
            "avg_persona_reason_coverage": round(avg_persona_reason_coverage, 4),
            "avg_style_coherence_score": round(avg_style_coherence, 4),
            "avg_bundle_completeness_score": round(avg_bundle_completeness, 4),
            "avg_long_term_fit_score": round(avg_long_term_fit, 4),
            "avg_phased_purchase_score": round(avg_phased_purchase, 4),
            "drift_detection_rate": round(drift_detection_rate, 4),
            "avg_drift_alignment_score": round(avg_drift_alignment, 4),
            # 方案质量
            "avg_plan_diversity": round(avg_plan_diversity, 4),
            # 过程指标
            "avg_clarification_turns": round(avg_turns, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_budget_ratio": round(avg_budget_ratio, 3),
            "avg_workflow_steps_completed": round(avg_workflow_steps, 2),
            # 图修复
            "graph_recovery_rate": round(graph_recovery_rate, 4),
            # 错误
            "error_rate": round(error_rate, 4),
            "ended_in_error_rate": round(ended_in_error_rate, 4),
            "error_type_breakdown": error_type_breakdown,
            "failure_bucket_breakdown": failure_bucket_breakdown,
            "task_family_summary": task_family_summary,
            "errors": errors[:5],  # 最多显示 5 个错误
        }

    def compare(
        self,
        baseline: dict[str, Any],
        experiment: dict[str, Any],
        label_baseline: str = "baseline",
        label_exp: str = "experiment",
    ) -> dict[str, Any]:
        """
        对比两组指标，计算相对提升（用于 RL vs 启发式 消融实验）。
        """
        metric_keys = [
            "success_rate", "coverage", "budget_satisfaction_rate",
            "avg_constraint_hit_rate", "avg_overall_score",
            "avg_plan_diversity", "avg_clarification_turns",
            "avg_budget_ratio", "graph_recovery_rate",
            "avg_parser_category_match", "avg_plan_category_match",
            "clarification_alignment_rate", "feasibility_alignment_rate",
            "avg_intent_resolution_score", "avg_execution_readiness_score",
            "avg_plan_persona_alignment_score", "avg_persona_reason_coverage",
            "avg_style_coherence_score", "avg_bundle_completeness_score",
            "avg_long_term_fit_score", "avg_phased_purchase_score",
            "drift_detection_rate", "avg_drift_alignment_score",
        ]
        comparison = {}
        for key in metric_keys:
            b = baseline.get(key, 0.0)
            e = experiment.get(key, 0.0)
            delta = e - b
            relative = (delta / b * 100) if b != 0 else float("inf")
            comparison[key] = {
                label_baseline: b,
                label_exp: e,
                "delta": round(delta, 4),
                "relative_pct": round(relative, 2),
            }
        return comparison
