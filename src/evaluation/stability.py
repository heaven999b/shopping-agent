"""
Stability Analysis – pass^k metrics.

Runs each task k times and computes:
    - success_rate: fraction of runs that succeeded
    - score variance: variance of U/E scores across runs
    - violation_rate_variance: variance of verifier violation counts
    - pass_at_k: probability that at least one of k runs succeeds
"""
from __future__ import annotations

import math
from collections import defaultdict

from ..models import StabilityResult, TaskEvaluation


def compute_stability(
    task_id: str,
    evaluations: list[TaskEvaluation],
) -> StabilityResult:
    """
    Compute stability metrics for a single task across k runs.

    Args:
        task_id: The task identifier
        evaluations: List of TaskEvaluation objects, one per run
    """
    k = len(evaluations)
    if k == 0:
        return StabilityResult(task_id=task_id, k=0, success_rate=0.0,
                               u_score_mean=0.0, u_score_variance=0.0,
                               e_score_mean=0.0, e_score_variance=0.0,
                               violation_counts=[], violation_rate_variance=0.0,
                               pass_at_k=0.0)

    successes = sum(1 for e in evaluations if e.success)
    success_rate = successes / k

    u_scores = [e.u_score.total for e in evaluations]
    e_scores = [e.e_score.total for e in evaluations]

    u_mean = sum(u_scores) / k
    e_mean = sum(e_scores) / k

    u_var = sum((s - u_mean) ** 2 for s in u_scores) / k
    e_var = sum((s - e_mean) ** 2 for s in e_scores) / k

    # Violation counts from verifier (proxy for instability of error detection)
    violation_counts: list[int] = []
    for e in evaluations:
        # We don't have the raw PipelineResult here, so we use verifier_detection_rate as proxy
        # A rate < 1.0 implies some violations were missed; we encode this as 0 or 1
        if e.stage_metrics.verifier_detection_rate < 1.0:
            violation_counts.append(1)
        else:
            violation_counts.append(0)

    viol_mean = sum(violation_counts) / k
    viol_var = sum((v - viol_mean) ** 2 for v in violation_counts) / k

    # pass^k = P(at least one success in k independent runs)
    # = 1 - (1 - p)^k   where p = success_rate
    pass_at_k = 1.0 - (1.0 - success_rate) ** k if success_rate < 1.0 else 1.0

    return StabilityResult(
        task_id=task_id,
        k=k,
        success_rate=round(success_rate, 4),
        u_score_mean=round(u_mean, 4),
        u_score_variance=round(u_var, 4),
        e_score_mean=round(e_mean, 4),
        e_score_variance=round(e_var, 4),
        violation_counts=violation_counts,
        violation_rate_variance=round(viol_var, 4),
        pass_at_k=round(pass_at_k, 4),
    )


def group_evaluations_by_task(
    evaluations: list[TaskEvaluation],
) -> dict[str, list[TaskEvaluation]]:
    """Group a flat list of evaluations by task_id."""
    groups: dict[str, list[TaskEvaluation]] = defaultdict(list)
    for e in evaluations:
        groups[e.task_id].append(e)
    return dict(groups)


def compute_all_stability(
    evaluations: list[TaskEvaluation],
) -> list[StabilityResult]:
    """Compute stability results for all tasks."""
    groups = group_evaluations_by_task(evaluations)
    results = []
    for task_id, evals in groups.items():
        evals_sorted = sorted(evals, key=lambda e: e.run_index)
        results.append(compute_stability(task_id, evals_sorted))
    return results
