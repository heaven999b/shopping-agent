#!/usr/bin/env python3
"""
Standalone benchmark runner.

用法：
  python run_benchmark.py                    # 全量启发式评测
  python run_benchmark.py --rl               # RL 模式评测
  python run_benchmark.py --compare          # 两种模式对比
  python run_benchmark.py --tasks t001,t002  # 指定任务
  python run_benchmark.py --save             # 保存报告到 logs/
  python run_benchmark.py --verbose 0        # 静默模式（只输出 summary）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 确保项目根目录在 PYTHONPATH 中
sys.path.insert(0, str(Path(__file__).parent))

from shopping_agent.evaluation.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="SHOP-PLAN Benchmark Runner")
    parser.add_argument("--rl", action="store_true", help="使用 RL 策略（默认启发式）")
    parser.add_argument("--compare", action="store_true", help="同时评测两种模式并输出对比")
    parser.add_argument("--tasks", default=None, help="逗号分隔的 task_id（默认全量）")
    parser.add_argument("--save", action="store_true", help="将报告保存到 logs/")
    parser.add_argument("--verbose", type=int, default=1, help="详细程度：0=只显示summary，1=逐任务（默认）")
    parser.add_argument("--output-dir", default="logs", help="报告保存目录")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "e2e"],
        default="pipeline",
        help="评测模式：pipeline=模块级，e2e=端到端公开入口",
    )
    args = parser.parse_args()

    task_ids = args.tasks.split(",") if args.tasks else None
    verbose = args.verbose >= 1

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
        print("  Running RL mode...")
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
    print(f"  Ablation Comparison: heuristic → RL")
    print("=" * 70)
    header = f"  {'Metric':<38} {'Heuristic':>10} {'RL':>10} {'Delta':>12}"
    print(header)
    print("  " + "-" * 66)
    for metric, vals in comparison.items():
        keys = list(vals.keys())
        baseline_key = keys[0]
        exp_key = keys[1]
        b = vals[baseline_key]
        e = vals[exp_key]
        delta = vals["delta"]
        rel = vals["relative_pct"]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<38} {b:>10.4f} {e:>10.4f} {sign}{delta:.4f} ({sign}{rel:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
