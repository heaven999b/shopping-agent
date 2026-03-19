"""
Pipeline-level evaluation metrics.

Computes per-stage metrics, U-score (Understanding), and E-score (Execution)
for each pipeline run, enabling quadrant analysis (U × E).
"""
from __future__ import annotations

from ..models import (
    BenchmarkTask, EScore, PipelineResult, StageMetrics, TaskEvaluation, UScore,
)


# ── Constraint extraction helpers ───────────────────────────────────────────

def _constraint_set(task: BenchmarkTask) -> set[str]:
    """Canonical set of constraint keys for a task."""
    keys = set()
    if task.constraints.budget_max is not None:
        keys.add("budget_max")
    if task.constraints.budget_min is not None:
        keys.add("budget_min")
    for attr in task.constraints.required_attributes:
        keys.add(f"attr:{attr}")
    for cat in task.constraints.categories:
        keys.add(f"category:{cat}")
    return keys


def constraint_extraction_f1(task: BenchmarkTask, result: PipelineResult) -> float:
    """
    Compute F1 between ground-truth constraint keys and extracted constraint keys.

    Compares task.constraints (ground truth) against
    clarification.extracted_constraints (agent output).
    """
    ground_truth = _constraint_set(task)
    if not ground_truth:
        # If there are no ground-truth constraints, full marks if agent extracted nothing
        return 1.0 if not result.clarification.extracted_constraints else 0.5

    extracted = set(result.clarification.extracted_constraints.keys())
    # Normalise extracted keys (might have "budget_max", "budget", "price", etc.)
    normalised: set[str] = set()
    for k in extracted:
        k_lower = k.lower().replace(" ", "_")
        if "budget" in k_lower or "price" in k_lower or "cost" in k_lower:
            normalised.add("budget_max")
        else:
            normalised.add(f"attr:{k_lower}")

    tp = len(ground_truth & normalised)
    fp = len(normalised - ground_truth)
    fn = len(ground_truth - normalised)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Retrieval metrics ───────────────────────────────────────────────────────

def retrieval_recall(task: BenchmarkTask, result: PipelineResult) -> float:
    """Fraction of acceptable products that were retrieved."""
    acceptable = set(task.acceptable_product_ids)
    if not acceptable:
        # Impossible task: recall is 1.0 if nothing was retrieved (correct), 0 otherwise
        return 1.0 if not result.retrieval.retrieved_product_ids else 0.0
    retrieved = set(result.retrieval.retrieved_product_ids)
    return len(acceptable & retrieved) / len(acceptable)


def retrieval_precision(task: BenchmarkTask, result: PipelineResult) -> float:
    """Fraction of retrieved products that are actually acceptable."""
    acceptable = set(task.acceptable_product_ids)
    retrieved = set(result.retrieval.retrieved_product_ids)
    if not retrieved:
        return 0.0
    if not acceptable:
        # Impossible task: precision should be 0 (retrieved something when nothing is valid)
        return 0.0
    return len(acceptable & retrieved) / len(retrieved)


# ── Graph metrics ────────────────────────────────────────────────────────────

def graph_coverage(task: BenchmarkTask, result: PipelineResult) -> float:
    """Fraction of ground-truth constraints covered in the graph output."""
    coverage_dict = result.graph.constraint_coverage
    if not coverage_dict:
        return 0.5  # unknown
    covered = sum(1 for v in coverage_dict.values() if v)
    return covered / len(coverage_dict)


# ── Planning metrics ─────────────────────────────────────────────────────────

def planning_precision(task: BenchmarkTask, result: PipelineResult) -> float:
    """Fraction of recommended products that are acceptable."""
    acceptable = set(task.acceptable_product_ids)
    recommended = set(result.planning.recommended_product_ids)
    if not recommended:
        return 0.0
    if not acceptable:
        return 0.0
    return len(acceptable & recommended) / len(recommended)


def planning_constraint_sat(task: BenchmarkTask, result: PipelineResult) -> float:
    """Fraction of constraints reported as satisfied in the planning output."""
    sat = result.planning.constraint_satisfaction
    if not sat:
        return 0.5  # unknown
    satisfied = sum(1 for v in sat.values() if v)
    return satisfied / len(sat)


# ── Verifier metrics ─────────────────────────────────────────────────────────

def verifier_detection_rate(task: BenchmarkTask, result: PipelineResult) -> float:
    """
    When the plan is actually invalid (verifier found violations or planner failed),
    measure the detection rate.

    Returns:
        1.0 if violations found and verifier caught them, or plan is valid and verifier said so.
        0.0 if verifier missed violations.
        0.5 if no violations expected and none found.
    """
    has_violations = len(result.verifier.violations) > 0
    plan_is_invalid = not result.verifier.is_valid

    # Ground-truth: plan should be invalid if no acceptable products exist
    # or if recommended products aren't in acceptable set
    acceptable = set(task.acceptable_product_ids)
    recommended = set(result.planning.recommended_product_ids)
    truly_invalid = (not acceptable) or (recommended and not recommended & acceptable)

    if truly_invalid and plan_is_invalid:
        return 1.0  # Correctly detected invalidity
    elif truly_invalid and not plan_is_invalid:
        return 0.0  # Missed the violation
    elif not truly_invalid and not plan_is_invalid:
        return 1.0  # Correctly validated
    else:
        return 0.5  # False alarm (said invalid when actually ok)


# ── Main metric computation ──────────────────────────────────────────────────

def compute_task_evaluation(
    task: BenchmarkTask,
    result: PipelineResult,
) -> TaskEvaluation:
    """Compute all metrics for a single pipeline result."""
    # Stage metrics
    sm = StageMetrics(
        clarification_accuracy=float(
            result.clarification.needs_clarification == task.needs_clarification
        ),
        constraint_extraction_f1=constraint_extraction_f1(task, result),
        retrieval_recall=retrieval_recall(task, result),
        retrieval_precision=retrieval_precision(task, result),
        graph_coverage=graph_coverage(task, result),
        planning_precision=planning_precision(task, result),
        planning_constraint_sat=planning_constraint_sat(task, result),
        verifier_detection_rate=verifier_detection_rate(task, result),
    )

    u = UScore.compute(
        clarification_decision=sm.clarification_accuracy,
        constraint_extraction=sm.constraint_extraction_f1,
    )
    e = EScore.compute(
        retrieval_recall=sm.retrieval_recall,
        retrieval_precision=sm.retrieval_precision,
        planning_precision=sm.planning_precision,
        planning_constraint_sat=sm.planning_constraint_sat,
        verifier_detection_rate=sm.verifier_detection_rate,
    )

    # Success = E-score above 0.5 and verifier approved
    success = e.total >= 0.5 and result.verifier.is_valid

    eval_obj = TaskEvaluation(
        task_id=task.id,
        difficulty=task.difficulty,
        category=task.category,
        u_score=u,
        e_score=e,
        stage_metrics=sm,
        success=success,
        run_index=result.run_index,
    )
    eval_obj.quadrant = eval_obj.determine_quadrant()
    return eval_obj
