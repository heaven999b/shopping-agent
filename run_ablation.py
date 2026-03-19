#!/usr/bin/env python3
"""
消融实验脚本 — 对比四种配置的系统性能。

消融组：
  A_baseline   — 无图构建 + 无 Verifier（最简规划）
  B_verifier   — 无图构建 + 有 Verifier
  C_graph      — 有图构建 + 有 Verifier
  D_full       — 有图构建 + 有 Verifier（完整系统，同 C，基线非 RL）

用法：
  python run_ablation.py                        # 全量消融（52 tasks × 3 组）
  python run_ablation.py --quick               # 快速模式（12 个任务）
  python run_ablation.py --task-ids t001 t021  # 只跑指定任务
  python run_ablation.py --save                # 保存详细结果到 logs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from shopping_agent.evaluation.benchmark import BenchmarkRunner
from shopping_agent.evaluation.metrics import MetricsComputer


ABLATION_CONDITIONS = [
    {
        "name": "A_baseline",
        "description": "无候选图 + 无 Verifier",
        "use_rl": False,
        "disable_graph": True,
        "disable_verifier": True,
    },
    {
        "name": "B_verifier",
        "description": "无候选图 + 有 Verifier",
        "use_rl": False,
        "disable_graph": True,
        "disable_verifier": False,
    },
    {
        "name": "C_graph_verifier",
        "description": "有候选图 + 有 Verifier（完整系统）",
        "use_rl": False,
        "disable_graph": False,
        "disable_verifier": False,
    },
]

QUICK_TASK_IDS = [
    "t001", "t003", "t007",
    "t014", "t015", "t016",
    "t021", "t025", "t034", "t037",
    "t042", "t043",
]


def run_ablation(
    task_ids: Optional[list[str]] = None,
    save: bool = False,
    quick: bool = False,
    output_dir: str = "logs",
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if quick and task_ids is None:
        task_ids = QUICK_TASK_IDS

    reports = {}
    print(f"\n{'='*65}")
    print(f"消融实验  {ts}")
    print(f"{'='*65}")

    for cond in ABLATION_CONDITIONS:
        print(f"\n[{cond['name']}] {cond['description']}")
        print("-" * 55)

        runner = BenchmarkRunner(
            use_rl=cond["use_rl"],
            output_dir=output_dir,
            disable_graph=cond["disable_graph"],
            disable_verifier=cond["disable_verifier"],
        )
        report = runner.run(task_ids=task_ids, verbose=True)
        report["condition"] = cond["name"]
        report["description"] = cond["description"]
        reports[cond["name"]] = report

        if save:
            path = runner.save_report(
                report,
                filename=f"ablation_{cond['name']}_{ts}.json",
            )
            print(f"  → 保存至 {path}")

    print_comparison_table(reports)

    if save:
        summary_path = Path(output_dir) / f"ablation_summary_{ts}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v["metrics"] for k, v in reports.items()},
                f, ensure_ascii=False, indent=2,
            )
        print(f"\n汇总已保存: {summary_path}")

    return reports


def print_comparison_table(reports: dict) -> None:
    metrics_spec = [
        ("success_rate",            "Success Rate",    ".1%"),
        ("coverage",                "Coverage",        ".1%"),
        ("budget_satisfaction_rate","Budget Sat.",     ".1%"),
        ("avg_constraint_hit_rate", "Constraint Hit",  ".1%"),
        ("avg_overall_score",       "Avg Score",       ".4f"),
        ("avg_plan_diversity",      "Plan Diversity",  ".4f"),
        ("graph_recovery_rate",     "Graph Recovery",  ".1%"),
        ("avg_latency_ms",          "Latency (ms)",    ".0f"),
    ]

    cond_names = list(reports.keys())
    col_w = 20

    print(f"\n{'='*75}")
    print("消融实验对比表   (* 标记各行最优值)")
    print(f"{'='*75}")
    header = f"{'Metric':<22}" + "".join(f"{n:>{col_w}}" for n in cond_names)
    print(header)
    print("-" * (22 + col_w * len(cond_names)))

    for key, label, fmt in metrics_spec:
        row = f"{label:<22}"
        vals = [reports[c]["metrics"].get(key, 0.0) for c in cond_names]
        best = min(vals) if "latency" in key.lower() else max(vals)
        for v in vals:
            cell = format(v, fmt)
            if v == best and len(set(vals)) > 1:
                cell = f"*{cell}"
            row += f"{cell:>{col_w}}"
        print(row)

    print("-" * (22 + col_w * len(cond_names)))

    # 相对提升：Full vs Baseline
    first, last = cond_names[0], cond_names[-1]
    if first != last:
        mc = MetricsComputer()
        comp = mc.compare(
            reports[first]["metrics"],
            reports[last]["metrics"],
            label_baseline=first,
            label_exp=last,
        )
        print(f"\n[{last} vs {first} 相对提升]")
        for key, label, _ in metrics_spec[:6]:
            if key in comp:
                pct = comp[key]["relative_pct"]
                sign = "+" if pct >= 0 else ""
                print(f"  {label:<24}: {sign}{pct:.1f}%")

    print(f"{'='*75}\n")


def main():
    parser = argparse.ArgumentParser(description="Shopping Agent 消融实验")
    parser.add_argument("--task-ids", nargs="+", help="只运行指定 task_id")
    parser.add_argument("--save", action="store_true", help="保存详细结果到 logs/")
    parser.add_argument("--quick", action="store_true", help="快速模式（12 个任务）")
    parser.add_argument("--output-dir", default="logs", help="输出目录")
    args = parser.parse_args()

    run_ablation(
        task_ids=args.task_ids,
        save=args.save,
        quick=args.quick,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
