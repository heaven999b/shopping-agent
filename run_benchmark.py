#!/usr/bin/env python3
"""
Standalone benchmark runner.

用法：
  python run_benchmark.py                    # 全量启发式评测
  python run_benchmark.py --rl               # RL-enhanced 模式评测
  python run_benchmark.py --compare          # 两种模式对比
  python run_benchmark.py --tasks t001,t002  # 指定任务
  python run_benchmark.py --save             # 保存报告到 logs/
  python run_benchmark.py --verbose 0        # 静默模式（只输出 summary）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

# 确保项目根目录在 PYTHONPATH 中
sys.path.insert(0, str(Path(__file__).parent))

from shopping_agent.evaluation.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="SHOP-PLAN Benchmark Runner")
    parser.add_argument("--rl", action="store_true", help="使用 RL-enhanced 策略（默认启发式）")
    parser.add_argument("--compare", action="store_true", help="同时评测两种模式并输出对比")
    parser.add_argument("--tasks", default=None, help="逗号分隔的 task_id（默认全量）")
    parser.add_argument("--save", action="store_true", help="将报告保存到 logs/")
    parser.add_argument(
        "--verbose", type=int, default=1, help="详细程度：0=只显示summary，1=逐任务（默认）"
    )
    parser.add_argument("--output-dir", default="logs", help="报告保存目录")
    parser.add_argument(
        "--baseline-suite",
        action="store_true",
        help="运行 full/naive-llm/no-clarification/no-constraint/no-memory/single-item/no-bundle-scoring 等 baseline 对比",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "e2e"],
        default="pipeline",
        help="评测模式：pipeline=模块级，e2e=端到端公开入口",
    )
    args = parser.parse_args()

    task_ids = args.tasks.split(",") if args.tasks else None
    verbose = args.verbose >= 1

    if args.baseline_suite:
        suite = [
            ("full_agent", dict(use_rl=False, baseline_profile="full_agent")),
            (
                "naive_llm_agent",
                dict(
                    use_rl=False,
                    baseline_profile="naive_llm_agent",
                    disable_graph=True,
                    disable_verifier=True,
                    disable_clarification=True,
                ),
            ),
            (
                "naive_retrieval",
                dict(
                    use_rl=False,
                    baseline_profile="naive_retrieval",
                    disable_graph=True,
                    disable_verifier=True,
                    disable_clarification=True,
                ),
            ),
            (
                "no_clarification",
                dict(
                    use_rl=False,
                    baseline_profile="no_clarification",
                    disable_clarification=True,
                ),
            ),
            ("no_constraint", dict(use_rl=False, baseline_profile="no_constraint")),
            ("constraint_only", dict(use_rl=False, baseline_profile="constraint_only")),
            ("no_memory", dict(use_rl=False, baseline_profile="no_memory")),
            ("single_item", dict(use_rl=False, baseline_profile="single_item")),
            (
                "no_bundle_scoring",
                dict(use_rl=False, baseline_profile="no_bundle_scoring"),
            ),
        ]
        reports = {}
        for name, kwargs in suite:
            print("\n" + "=" * 60)
            print(f"  Running baseline: {name}")
            print("=" * 60)
            runner = BenchmarkRunner(
                output_dir=args.output_dir,
                benchmark_mode=args.mode,
                **kwargs,
            )
            reports[name] = runner.run(task_ids=task_ids, verbose=verbose)
        _print_baseline_suite(reports)
        if args.save:
            out_dir = Path(args.output_dir)
            json_path = out_dir / "baseline_suite.json"
            md_path = out_dir / "baseline_suite.md"
            csv_path = out_dir / "baseline_suite.csv"
            main_results_path = out_dir / "main_results.md"
            task_distribution_json = out_dir / "task_distribution.json"
            task_distribution_md = out_dir / "task_distribution.md"
            json_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2))
            md_path.write_text(_baseline_suite_markdown(reports), encoding="utf-8")
            _write_baseline_suite_csv(csv_path, reports)
            main_results_path.write_text(
                _main_results_markdown(reports),
                encoding="utf-8",
            )
            distribution = _task_distribution_summary(reports)
            task_distribution_json.write_text(
                json.dumps(distribution, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            task_distribution_md.write_text(
                _task_distribution_markdown(distribution),
                encoding="utf-8",
            )
            print(f"\nbaseline suite 已保存: {json_path}")
            print(f"markdown 表格已保存: {md_path}")
            print(f"csv 表格已保存: {csv_path}")
            print(f"主结果摘要已保存: {main_results_path}")
            print(f"任务分布已保存: {task_distribution_json}")
            print(f"任务分布说明已保存: {task_distribution_md}")
        return

    if args.compare:
        # 消融对比
        print("=" * 60)
        print("  Running heuristic baseline...")
        print("=" * 60)
        runner_h = BenchmarkRunner(
            use_rl=False,
            output_dir=args.output_dir,
            benchmark_mode=args.mode,
        )
        report_h = runner_h.run(task_ids=task_ids, verbose=verbose)

        print("\n" + "=" * 60)
        print("  Running RL-enhanced mode...")
        print("=" * 60)
        runner_r = BenchmarkRunner(
            use_rl=True,
            output_dir=args.output_dir,
            benchmark_mode=args.mode,
        )
        report_r = runner_r.run(task_ids=task_ids, verbose=verbose)

        comparison = runner_h.compare(report_h, report_r)
        _print_comparison(comparison)

        if args.save:
            runner_h.save_report(report_h)
            runner_r.save_report(report_r)
            path = Path(args.output_dir) / "ablation_comparison.json"
            path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
            print(f"\n对比报告已保存: {path}")

    else:
        runner = BenchmarkRunner(
            use_rl=args.rl,
            output_dir=args.output_dir,
            benchmark_mode=args.mode,
        )
        report = runner.run(task_ids=task_ids, verbose=verbose)

        if args.save:
            saved_path = runner.save_report(report)
            print(f"报告已保存: {saved_path}")


def _print_comparison(comparison: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  Ablation Comparison: heuristic → RL-enhanced")
    print("=" * 70)


def _print_baseline_suite(reports: dict[str, dict]) -> None:
    print("\n" + "=" * 88)
    print("  Baseline Suite")
    print("=" * 88)


def _baseline_suite_rows(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        metrics = report["metrics"]
        bundle_summary = metrics.get("bundle_summary", {})
        avg_budget_ratio = metrics.get("avg_budget_ratio", 0.0)
        cost_band = (
            "low"
            if avg_budget_ratio < 0.55
            else "mid"
            if avg_budget_ratio < 0.85
            else "high"
        )
        regret_risk = max(
            0.0, avg_budget_ratio - metrics.get("budget_satisfaction_rate", 0.0)
        )
        rows.append(
            {
                "method": name,
                "all_task_success_rate": round(metrics.get("success_rate", 0.0), 4),
                "coverage": round(metrics.get("coverage", 0.0), 4),
                "budget_satisfaction_rate": round(
                    metrics.get("budget_satisfaction_rate", 0.0), 4
                ),
                "bundle_success_rate": round(
                    bundle_summary.get("bundle_success_rate", 0.0), 4
                ),
                "bundle_task_count": int(bundle_summary.get("num_tasks", 0)),
                "cost_band": cost_band,
                "bundle_score": round(
                    bundle_summary.get(
                        "avg_bundle_decision_score",
                        metrics.get("avg_bundle_decision_score", 0.0),
                    ),
                    4,
                ),
                "bundle_completeness": round(
                    bundle_summary.get(
                        "avg_bundle_completeness_score",
                        metrics.get("avg_bundle_completeness_score", 0.0),
                    ),
                    4,
                ),
                "compatibility": round(
                    bundle_summary.get(
                        "avg_compatibility_score",
                        metrics.get("avg_compatibility_score", 0.0),
                    ),
                    4,
                ),
                "relation_coverage": round(
                    bundle_summary.get(
                        "avg_relation_coverage_score",
                        metrics.get("avg_relation_coverage_score", 0.0),
                    ),
                    4,
                ),
                "long_term_fit": round(
                    bundle_summary.get(
                        "avg_long_term_fit_score",
                        metrics.get("avg_long_term_fit_score", 0.0),
                    ),
                    4,
                ),
                "phased_purchase": round(
                    bundle_summary.get(
                        "avg_phased_purchase_score",
                        metrics.get("avg_phased_purchase_score", 0.0),
                    ),
                    4,
                ),
                "regret_risk": round(regret_risk, 4),
            }
        )
    return rows


def _baseline_suite_markdown(reports: dict[str, dict]) -> str:
    rows = _baseline_suite_rows(reports)
    lines = [
        "## All-Task Summary",
        "",
        "| Method | AllTaskSuccess | Coverage | BudgetSatisfaction | Cost |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['all_task_success_rate']:.2%} | {row['coverage']:.2%} | "
            f"{row['budget_satisfaction_rate']:.2%} | {row['cost_band']} |"
        )
    lines.extend(
        [
            "",
            "## Bundle-Only Summary",
            "",
            "| Method | BundleTasks | BundleSuccess | BundleScore | BundleCompleteness | Compatibility | RelationCoverage | LongTermFit | PhasedPurchase | RegretRisk |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['bundle_task_count']} | {row['bundle_success_rate']:.2%} | "
            f"{row['bundle_score']:.4f} | {row['bundle_completeness']:.4f} | "
            f"{row['compatibility']:.4f} | {row['relation_coverage']:.4f} | "
            f"{row['long_term_fit']:.4f} | {row['phased_purchase']:.4f} | {row['regret_risk']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _write_baseline_suite_csv(path: Path, reports: dict[str, dict]) -> None:
    rows = _baseline_suite_rows(reports)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "all_task_success_rate",
                "coverage",
                "budget_satisfaction_rate",
                "bundle_success_rate",
                "bundle_task_count",
                "cost_band",
                "bundle_score",
                "bundle_completeness",
                "compatibility",
                "relation_coverage",
                "long_term_fit",
                "phased_purchase",
                "regret_risk",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _task_distribution_summary(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = reports.get("full_agent") or next(iter(reports.values()))
    per_task = reference.get("per_task", [])
    family_counts = Counter(
        (task.get("original_task_family") or task.get("task_family") or "general")
        for task in per_task
    )
    type_counts = Counter(task.get("task_type") or "unknown" for task in per_task)
    difficulty_counts = Counter(
        task.get("task_difficulty") or "unknown" for task in per_task
    )
    clarification_needed = sum(
        1 for task in per_task if task.get("clarification_needed")
    )
    bundle_like = sum(
        1
        for task in per_task
        if (task.get("original_task_family") or task.get("task_family"))
        in {"bundle", "upgrade_path", "phased_purchase", "bundle_noise"}
    )
    return {
        "num_tasks": len(per_task),
        "bundle_like_tasks": bundle_like,
        "clarification_needed_tasks": clarification_needed,
        "task_family_counts": dict(family_counts),
        "task_type_counts": dict(type_counts),
        "task_difficulty_counts": dict(difficulty_counts),
    }


def _task_distribution_markdown(summary: dict[str, object]) -> str:
    family_counts = cast(dict[str, int], summary["task_family_counts"])
    type_counts = cast(dict[str, int], summary["task_type_counts"])
    difficulty_counts = cast(dict[str, int], summary["task_difficulty_counts"])

    lines = [
        "# Task Distribution",
        "",
        f"- NumTasks: {summary['num_tasks']}",
        f"- BundleLikeTasks: {summary['bundle_like_tasks']}",
        f"- ClarificationNeededTasks: {summary['clarification_needed_tasks']}",
        "",
        "## Task Family Counts",
        "",
        "| Family | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(family_counts.items()):
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Task Type Counts",
            "",
            "| Type | Count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(type_counts.items()):
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Difficulty Counts",
            "",
            "| Difficulty | Count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(difficulty_counts.items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def _main_results_markdown(reports: dict[str, dict[str, Any]]) -> str:
    rows = {row["method"]: row for row in _baseline_suite_rows(reports)}
    full = rows["full_agent"]

    def _delta(other: str, key: str) -> float:
        return float(full[key]) - float(rows[other][key])

    lines = [
        "# Main Results",
        "",
        "## Primary Bundle Comparisons",
        "",
        "| Comparison | Metric | FullAgent | Baseline | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    comparisons = [
        ("naive_llm_agent", "bundle_success_rate", "BundleSuccess"),
        ("no_memory", "bundle_success_rate", "BundleSuccess"),
        ("single_item", "bundle_success_rate", "BundleSuccess"),
        ("naive_llm_agent", "bundle_score", "BundleScore"),
        ("no_memory", "long_term_fit", "LongTermFit"),
        ("naive_llm_agent", "relation_coverage", "RelationCoverage"),
    ]
    for baseline, key, label in comparisons:
        if baseline not in rows:
            continue
        lines.append(
            f"| full_agent vs {baseline} | {label} | "
            f"{float(full[key]):.4f} | {float(rows[baseline][key]):.4f} | {_delta(baseline, key):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `All-Task Summary` 用于展示总体覆盖和预算满足率。",
            "- `Bundle-Only Summary` 是主结果表，优先用于说明组合规划、长期记忆与 phased purchase 的贡献。",
            "- 当前 benchmark 为固定任务集与确定性 pipeline 评测，默认报告绝对差值而非显著性检验。",
            "- 若需要显著性描述，下一步应基于多 seed / 多次重跑输出均值与波动范围。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
