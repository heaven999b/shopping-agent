"""
Evaluator – orchestrates evaluation across the full benchmark.

Runs the pipeline for each task (optionally k times for stability),
computes all metrics, classifies errors, and produces a summary report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..models import (
    BenchmarkTask, Constraint, EvaluationSummary, PipelineResult,
    Product, StageMetrics, TaskEvaluation,
)
from ..agent.pipeline import ShoppingAgentPipeline
from .metrics import compute_task_evaluation
from .error_taxonomy import classify_errors
from .stability import compute_all_stability


def load_catalog(path: str | Path) -> list[Product]:
    with open(path) as f:
        return [Product(**p) for p in json.load(f)]


def load_tasks(path: str | Path) -> list[BenchmarkTask]:
    tasks = []
    with open(path) as f:
        raw = json.load(f)
    for item in raw:
        # Convert raw constraints dict to Constraint model
        c_raw = item.get("constraints", {})
        constraint = Constraint(
            budget_max=c_raw.get("budget_max"),
            budget_min=c_raw.get("budget_min"),
            categories=c_raw.get("categories", []),
            required_attributes=c_raw.get("required_attributes", {}),
            excluded_brands=c_raw.get("excluded_brands", []),
            quantity=c_raw.get("quantity", 1),
        )
        task = BenchmarkTask(
            id=item["id"],
            user_query=item["user_query"],
            constraints=constraint,
            needs_clarification=item["needs_clarification"],
            acceptable_product_ids=item["acceptable_product_ids"],
            difficulty=item["difficulty"],
            category=item["category"],
            description=item.get("description", ""),
        )
        tasks.append(task)
    return tasks


class Evaluator:
    """
    Full benchmark evaluator.

    Usage:
        evaluator = Evaluator(pipeline, catalog, tasks)
        summary = evaluator.run(k=1, on_result=callback)
    """

    def __init__(
        self,
        pipeline: ShoppingAgentPipeline,
        catalog: list[Product],
        tasks: list[BenchmarkTask],
    ):
        self.pipeline = pipeline
        self.catalog = catalog
        self.tasks = tasks

    def run(
        self,
        k: int = 1,
        task_ids: list[str] | None = None,
        on_result: Callable[[BenchmarkTask, PipelineResult, TaskEvaluation], None] | None = None,
    ) -> EvaluationSummary:
        """
        Run evaluation.

        Args:
            k: Number of runs per task (for pass^k stability analysis)
            task_ids: If provided, only evaluate these task IDs
            on_result: Optional callback invoked after each task/run completes
        """
        tasks = (
            [t for t in self.tasks if t.id in task_ids]
            if task_ids else self.tasks
        )

        all_evaluations: list[TaskEvaluation] = []
        all_results: list[PipelineResult] = []

        for task in tasks:
            for run_idx in range(k):
                result = self.pipeline.run(task, self.catalog, run_index=run_idx)
                evaluation = compute_task_evaluation(task, result)
                evaluation.error_types = classify_errors(task, result, evaluation)
                evaluation.run_index = run_idx

                all_results.append(result)
                all_evaluations.append(evaluation)

                if on_result:
                    on_result(task, result, evaluation)

        return self._summarise(all_evaluations, k)

    def _summarise(
        self,
        evaluations: list[TaskEvaluation],
        k: int,
    ) -> EvaluationSummary:
        if not evaluations:
            return EvaluationSummary(
                n_tasks=0, n_runs_per_task=k,
                avg_u_score=0.0, avg_e_score=0.0,
                quadrant_distribution={}, error_type_counts={},
                difficulty_breakdown={}, avg_stage_metrics=StageMetrics(),
            )

        n = len(evaluations)
        avg_u = sum(e.u_score.total for e in evaluations) / n
        avg_e = sum(e.e_score.total for e in evaluations) / n

        # Quadrant distribution
        quad_dist: dict[str, int] = {"HH": 0, "HL": 0, "LH": 0, "LL": 0}
        for e in evaluations:
            quad_dist[e.quadrant] = quad_dist.get(e.quadrant, 0) + 1

        # Error type counts
        error_counts: dict[str, int] = {}
        for e in evaluations:
            for err in e.error_types:
                error_counts[err] = error_counts.get(err, 0) + 1

        # Difficulty breakdown
        difficulty_groups: dict[str, list[TaskEvaluation]] = {}
        for e in evaluations:
            difficulty_groups.setdefault(e.difficulty, []).append(e)

        difficulty_breakdown: dict[str, dict[str, float]] = {}
        for diff, evals in difficulty_groups.items():
            difficulty_breakdown[diff] = {
                "avg_u_score": round(sum(e.u_score.total for e in evals) / len(evals), 4),
                "avg_e_score": round(sum(e.e_score.total for e in evals) / len(evals), 4),
                "success_rate": round(sum(1 for e in evals if e.success) / len(evals), 4),
                "n_tasks": len(set(e.task_id for e in evals)),
            }

        # Average stage metrics
        def avg_field(field: str) -> float:
            vals = [getattr(e.stage_metrics, field) for e in evaluations]
            return round(sum(vals) / len(vals), 4)

        avg_sm = StageMetrics(
            clarification_accuracy=avg_field("clarification_accuracy"),
            constraint_extraction_f1=avg_field("constraint_extraction_f1"),
            retrieval_recall=avg_field("retrieval_recall"),
            retrieval_precision=avg_field("retrieval_precision"),
            graph_coverage=avg_field("graph_coverage"),
            planning_precision=avg_field("planning_precision"),
            planning_constraint_sat=avg_field("planning_constraint_sat"),
            verifier_detection_rate=avg_field("verifier_detection_rate"),
        )

        # Stability (pass^k)
        stability = compute_all_stability(evaluations) if k > 1 else []

        n_tasks = len(set(e.task_id for e in evaluations))

        return EvaluationSummary(
            n_tasks=n_tasks,
            n_runs_per_task=k,
            avg_u_score=round(avg_u, 4),
            avg_e_score=round(avg_e, 4),
            quadrant_distribution=quad_dist,
            error_type_counts=error_counts,
            difficulty_breakdown=difficulty_breakdown,
            avg_stage_metrics=avg_sm,
            stability_results=stability,
        )
