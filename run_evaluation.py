#!/usr/bin/env python3
"""
Shopping Agent Evaluation CLI

Usage:
    # Full evaluation (requires ANTHROPIC_API_KEY)
    python run_evaluation.py --tasks benchmark/tasks.json --catalog benchmark/catalog.json

    # Dry-run with mock agent (no API calls)
    python run_evaluation.py --mock

    # Pass^k stability analysis (k=5 runs per task)
    python run_evaluation.py --mock --k 5

    # Evaluate specific tasks
    python run_evaluation.py --mock --task-ids task_001 task_002 task_021

    # Save results to JSON
    python run_evaluation.py --mock --output results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box

from src.agent.pipeline import ShoppingAgentPipeline
from src.evaluation.evaluator import Evaluator, load_catalog, load_tasks
from src.evaluation.error_taxonomy import error_type_description
from src.models import BenchmarkTask, EvaluationSummary, PipelineResult, TaskEvaluation

console = Console()

BENCHMARK_DIR = Path(__file__).parent / "benchmark"


# ── Display helpers ──────────────────────────────────────────────────────────

def _quadrant_emoji(q: str) -> str:
    return {"HH": "✅", "HL": "⚠️ ", "LH": "🔀", "LL": "❌"}.get(q, "?")


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    elif score >= 0.4:
        return "yellow"
    return "red"


def print_task_result(
    task: BenchmarkTask,
    result: PipelineResult,
    evaluation: TaskEvaluation,
) -> None:
    u = evaluation.u_score.total
    e = evaluation.e_score.total
    quad = evaluation.quadrant
    errors = evaluation.error_types

    u_str = f"[{_score_color(u)}]{u:.2f}[/]"
    e_str = f"[{_score_color(e)}]{e:.2f}[/]"
    success_str = "[green]SUCCESS[/]" if evaluation.success else "[red]FAIL[/]"

    error_str = ", ".join(errors) if errors else "—"
    console.print(
        f"  [{task.difficulty:6s}] {task.id} | "
        f"U={u_str} E={e_str} | "
        f"{_quadrant_emoji(quad)} {quad} | "
        f"{success_str} | errors: {error_str}"
    )


def print_stage_detail(result: PipelineResult) -> None:
    console.print(f"    Clarification: needs_clarification={result.clarification.needs_clarification}")
    console.print(f"    Retrieval: {len(result.retrieval.retrieved_product_ids)} products retrieved")
    console.print(f"    Graph: {len(result.graph.ranked_nodes)} nodes ranked")
    console.print(f"    Planning: recommended={result.planning.recommended_product_ids}")
    console.print(f"    Verifier: valid={result.verifier.is_valid}, violations={result.verifier.violations}")


def print_summary(summary: EvaluationSummary) -> None:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]EVALUATION SUMMARY[/bold cyan]",
        border_style="cyan",
    ))

    # Overall scores
    u = summary.avg_u_score
    e = summary.avg_e_score
    console.print(f"\n[bold]Overall Scores[/bold] (across {summary.n_tasks} tasks, {summary.n_runs_per_task} run(s) each)")
    console.print(f"  Avg U-score (Understanding): [{_score_color(u)}]{u:.4f}[/]")
    console.print(f"  Avg E-score (Execution):     [{_score_color(e)}]{e:.4f}[/]")

    # Quadrant distribution
    console.print("\n[bold]Quadrant Distribution (Understanding × Execution)[/bold]")
    quad = summary.quadrant_distribution
    total = sum(quad.values())
    for q in ["HH", "HL", "LH", "LL"]:
        count = quad.get(q, 0)
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 5)
        console.print(f"  {_quadrant_emoji(q)} {q}: {count:3d} ({pct:5.1f}%)  {bar}")

    # Per-stage metrics
    sm = summary.avg_stage_metrics
    console.print("\n[bold]Per-Stage Metrics[/bold]")
    stage_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    stage_table.add_column("Stage", style="cyan")
    stage_table.add_column("Metric", style="white")
    stage_table.add_column("Score", justify="right")

    def score_cell(v: float) -> str:
        return f"[{_score_color(v)}]{v:.4f}[/]"

    stage_table.add_row("Clarification", "Accuracy",         score_cell(sm.clarification_accuracy))
    stage_table.add_row("Clarification", "Constraint F1",    score_cell(sm.constraint_extraction_f1))
    stage_table.add_row("Retrieval",     "Recall",           score_cell(sm.retrieval_recall))
    stage_table.add_row("Retrieval",     "Precision",        score_cell(sm.retrieval_precision))
    stage_table.add_row("Graph",         "Coverage",         score_cell(sm.graph_coverage))
    stage_table.add_row("Planning",      "Precision",        score_cell(sm.planning_precision))
    stage_table.add_row("Planning",      "Constraint Sat.",  score_cell(sm.planning_constraint_sat))
    stage_table.add_row("Verifier",      "Detection Rate",   score_cell(sm.verifier_detection_rate))
    console.print(stage_table)

    # Difficulty breakdown
    console.print("[bold]Difficulty Breakdown[/bold]")
    diff_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    diff_table.add_column("Difficulty", style="cyan")
    diff_table.add_column("Tasks", justify="right")
    diff_table.add_column("Avg U", justify="right")
    diff_table.add_column("Avg E", justify="right")
    diff_table.add_column("Success Rate", justify="right")

    for diff in ["easy", "medium", "hard"]:
        if diff in summary.difficulty_breakdown:
            db = summary.difficulty_breakdown[diff]
            diff_table.add_row(
                diff,
                str(int(db["n_tasks"])),
                score_cell(db["avg_u_score"]),
                score_cell(db["avg_e_score"]),
                score_cell(db["success_rate"]),
            )
    console.print(diff_table)

    # Error taxonomy
    if summary.error_type_counts:
        console.print("[bold]Error Taxonomy[/bold]")
        err_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        err_table.add_column("Error Type", style="red")
        err_table.add_column("Count", justify="right")
        err_table.add_column("Description", style="dim")

        for err_type, count in sorted(summary.error_type_counts.items(), key=lambda x: -x[1]):
            err_table.add_row(err_type, str(count), error_type_description(err_type)[:60])
        console.print(err_table)

    # Stability (pass^k)
    if summary.stability_results:
        console.print(f"\n[bold]Stability Analysis (k={summary.n_runs_per_task})[/bold]")
        stab_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        stab_table.add_column("Task ID", style="cyan")
        stab_table.add_column("Success Rate", justify="right")
        stab_table.add_column("E-score Mean", justify="right")
        stab_table.add_column("E-score Var.", justify="right")
        stab_table.add_column("pass^k", justify="right")

        for sr in sorted(summary.stability_results, key=lambda x: x.success_rate):
            stab_table.add_row(
                sr.task_id,
                score_cell(sr.success_rate),
                score_cell(sr.e_score_mean),
                f"{sr.e_score_variance:.4f}",
                score_cell(sr.pass_at_k),
            )
        console.print(stab_table)


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--catalog", default=str(BENCHMARK_DIR / "catalog.json"),
              help="Path to catalog JSON", show_default=True)
@click.option("--tasks", default=str(BENCHMARK_DIR / "tasks.json"),
              help="Path to tasks JSON", show_default=True)
@click.option("--mock", is_flag=True, default=False,
              help="Use mock agent (no API calls, for testing)")
@click.option("--k", default=1, show_default=True,
              help="Number of runs per task (for pass^k stability analysis)")
@click.option("--task-ids", multiple=True, metavar="TASK_ID",
              help="Specific task IDs to evaluate (default: all). Can be repeated.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show per-stage detail for each task")
@click.option("--output", default=None,
              help="Save summary JSON to this path")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY",
              help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)")
def main(
    catalog: str,
    tasks: str,
    mock: bool,
    k: int,
    task_ids: tuple[str, ...],
    verbose: bool,
    output: str | None,
    api_key: str | None,
) -> None:
    """Shopping Agent Evaluation Framework (tSCOPE-inspired)."""

    if not mock and not api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set. Use --mock for testing.")
        sys.exit(1)

    # Load data
    console.print(f"Loading catalog from [cyan]{catalog}[/cyan]")
    product_catalog = load_catalog(catalog)
    console.print(f"Loading tasks from [cyan]{tasks}[/cyan]")
    task_list = load_tasks(tasks)

    mode = "[yellow]MOCK[/yellow]" if mock else "[green]LIVE (claude-opus-4-6)[/green]"
    console.print(f"\nMode: {mode} | Tasks: {len(task_list)} | k={k}\n")

    # Build pipeline and evaluator
    pipeline = ShoppingAgentPipeline(api_key=api_key, mock=mock)
    evaluator = Evaluator(pipeline, product_catalog, task_list)

    # Track results
    results_count = [0]
    start_time = time.time()

    def on_result(task: BenchmarkTask, result: PipelineResult, evaluation: TaskEvaluation) -> None:
        results_count[0] += 1
        print_task_result(task, result, evaluation)
        if verbose:
            print_stage_detail(result)

    # Run evaluation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        n_total = len(task_ids or [t.id for t in task_list]) * k
        prog_task = progress.add_task("Evaluating...", total=n_total)

        original_on_result = on_result

        def on_result_with_progress(task, result, evaluation):
            original_on_result(task, result, evaluation)
            progress.advance(prog_task)

        summary = evaluator.run(
            k=k,
            task_ids=list(task_ids) if task_ids else None,
            on_result=on_result_with_progress,
        )

    elapsed = time.time() - start_time
    console.print(f"\n[dim]Completed {results_count[0]} evaluations in {elapsed:.1f}s[/dim]")

    print_summary(summary)

    if output:
        out_path = Path(output)
        with open(out_path, "w") as f:
            json.dump(summary.model_dump(), f, indent=2)
        console.print(f"\n[green]Results saved to {out_path}[/green]")


if __name__ == "__main__":
    main()
