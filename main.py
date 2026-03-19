"""
SHOP-PLAN Shopping Agent — 命令行入口

用法：
  # 交互模式（对话购物）
  python main.py chat
  python main.py chat --user-id "user_001" --rl

  # 评测模式（跑 benchmark）
  python main.py benchmark
  python main.py benchmark --rl --save
  python main.py benchmark --quick --save
  python main.py benchmark --compare   # 同时跑启发式和 RL-enhanced，输出对比
  python main.py benchmark --baseline-suite

  # RL-enhanced 策略训练
  python main.py train --iters 200 --episodes 32

  # 数据目录检查
  python main.py catalog

  # 一键 demo（无需 API Key）
  python main.py demo

  # 计划工作台操作
  python main.py plan --session-id <id> --show
  python main.py plan --session-id <id> --accept --route "分阶段升级"
  python main.py plan --session-id <id> --advance-phase
  python main.py plan --session-id <id> --complete

环境变量：
  ANTHROPIC_API_KEY  对话/LLM 功能必须设置；纯 benchmark 不需要
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

from shopping_agent.agent.orchestrator import ShoppingAgentOrchestrator
from shopping_agent.common.types import FeedbackSignal


# ---------------------------------------------------------------------------
# 子命令：chat
# ---------------------------------------------------------------------------

def cmd_chat(args) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("错误：对话模式需要设置环境变量 ANTHROPIC_API_KEY")
        sys.exit(1)

    user_id = args.user_id or f"user_{uuid.uuid4().hex[:8]}"
    orchestrator = ShoppingAgentOrchestrator(use_rl=args.rl)
    session_id = None

    print("=" * 60)
    print("  SHOP-PLAN 购物智能体")
    mode = "RL-enhanced" if args.rl else "启发式"
    print(f"  模式: {mode}")
    print("  输入购物需求，输入 'quit' 退出，'feedback' 给出反馈")
    print("=" * 60)
    print(f"用户ID: {user_id}\n")

    while True:
        try:
            user_input = input("您: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower() == "feedback" and session_id:
            _handle_feedback(orchestrator, session_id)
            continue

        try:
            if session_id is None:
                result = orchestrator.run(user_id=user_id, user_input=user_input)
                session_id = result["session_id"]
            else:
                result = orchestrator.continue_session(
                    session_id=session_id, user_input=user_input
                )

            print(f"\nAgent: {result['message']}\n")

            if result.get("plan"):
                plan = result["plan"]
                print(f"  [方案 {plan['plan_id'][:8]}] 总价: ¥{plan['net_price']:.2f} "
                      f"得分: {plan['overall_score']:.0%}")
                for item in plan["items"]:
                    print(f"  - {item['slot']}: {item['product']} ¥{item['price']:.2f}")
                print()

            if result.get("current_step") == "done" and not result.get("needs_input"):
                print("─" * 40)
                print("本次任务已完成。您可以开始新的购物需求。")
                print("─" * 40)
                session_id = None

        except Exception as e:
            print(f"\n[系统错误] {e}\n")


# ---------------------------------------------------------------------------
# 子命令：benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args) -> None:
    from shopping_agent.evaluation.benchmark import BenchmarkRunner
    from shopping_agent.data.loader import load_benchmark_tasks

    task_ids = args.tasks.split(",") if args.tasks else None
    if args.quick and not task_ids:
        task_ids = [task["task_id"] for task in load_benchmark_tasks()[:5]]

    if args.baseline_suite:
        import subprocess
        cmd = [
            sys.executable,
            "run_benchmark.py",
            "--baseline-suite",
            "--mode",
            "e2e" if args.quick else "pipeline",
        ]
        if args.save:
            cmd.append("--save")
        subprocess.run(cmd, check=True)
        return

    if args.compare:
        # 同时跑两种模式并对比
        print("Running heuristic baseline...")
        runner_base = BenchmarkRunner(use_rl=False, output_dir="logs")
        report_base = runner_base.run(verbose=True)

        print("\nRunning RL-enhanced mode...")
        runner_rl = BenchmarkRunner(use_rl=True, output_dir="logs")
        report_rl = runner_rl.run(verbose=True)

        comparison = runner_base.compare(report_base, report_rl)
        print("\n" + "=" * 60)
        print("Ablation Comparison (heuristic vs RL-enhanced)")
        print("=" * 60)
        for metric, vals in comparison.items():
            delta_str = f"{vals['delta']:+.4f} ({vals['relative_pct']:+.1f}%)"
            print(f"  {metric:<35} {vals['heuristic']:.4f} → {vals['rl']:.4f}  {delta_str}")

        if args.save:
            runner_base.save_report(report_base)
            runner_rl.save_report(report_rl)
            path = "logs/ablation_comparison.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            print(f"\n对比结果已保存: {path}")
        return

    # 单模式
    runner = BenchmarkRunner(use_rl=args.rl, output_dir="logs")
    report = runner.run(task_ids=task_ids, verbose=True)

    if args.save:
        runner.save_report(report)


# ---------------------------------------------------------------------------
# 子命令：train
# ---------------------------------------------------------------------------

def cmd_train(args) -> None:
    print(f"Starting RL-enhanced policy training: {args.iters} iterations × {args.episodes} episodes")
    orchestrator = ShoppingAgentOrchestrator()
    logs = orchestrator.train_rl(
        num_iterations=args.iters,
        episodes_per_iter=args.episodes,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"\nTraining complete. {len(logs)} iteration logs recorded.")
    if logs:
        last = logs[-1]
        print(f"Final eval: success_rate={last.get('eval_success_rate', 'N/A')}, "
              f"avg_turns={last.get('eval_avg_turns', 'N/A')}")

    path = "logs/rl_training_log.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"训练日志已保存: {path}")


# ---------------------------------------------------------------------------
# 子命令：catalog
# ---------------------------------------------------------------------------

def cmd_catalog(args) -> None:
    from shopping_agent.data.loader import ProductCatalog

    catalog = ProductCatalog.load()
    print(f"商品目录加载完成：共 {catalog.count()} 个商品")
    print(f"品类列表: {catalog.categories()}")

    if args.search:
        results = catalog.search(keyword=args.search, top_k=5)
        print(f"\n搜索 '{args.search}' 的结果（Top 5）：")
        for p in results:
            print(f"  [{p.product_id}] {p.brand} {p.title[:30]} ¥{p.final_price:.0f} ★{p.rating}")


def cmd_demo(args) -> None:
    from shopping_agent.data.loader import load_benchmark_tasks
    from shopping_agent.evaluation.benchmark import BenchmarkRunner

    task_ids = [task["task_id"] for task in load_benchmark_tasks()[:2]]
    runner = BenchmarkRunner(output_dir="logs", benchmark_mode="e2e")
    report = runner.run(task_ids=task_ids, verbose=True)
    saved = runner.save_report(report, filename="demo_report.json")
    print(f"\nDemo 完成，结果已保存: {saved}")


# ---------------------------------------------------------------------------
# 子命令：plan
# ---------------------------------------------------------------------------

def cmd_plan(args) -> None:
    orchestrator = ShoppingAgentOrchestrator()

    if args.show:
        workspace = orchestrator.get_workspace(args.session_id)
        _print_workspace(workspace)
        return

    if args.accept:
        workspace = orchestrator.update_plan_status(
            args.session_id,
            action="accept",
            route=args.route or "一步到位",
        )
        print("已接受当前方案。")
        _print_workspace(workspace)
        return

    if args.advance_phase:
        workspace = orchestrator.update_plan_status(
            args.session_id,
            action="advance_phase",
        )
        print("已推进到下一阶段。")
        _print_workspace(workspace)
        return

    if args.complete:
        workspace = orchestrator.update_plan_status(
            args.session_id,
            action="complete",
        )
        print("已标记当前计划完成。")
        _print_workspace(workspace)
        return

    if args.archive:
        workspace = orchestrator.update_plan_status(
            args.session_id,
            action="archive",
        )
        print("已归档当前计划。")
        _print_workspace(workspace)
        return

    print("请至少指定一个动作：--show / --accept / --advance-phase / --complete / --archive")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _handle_feedback(orchestrator: ShoppingAgentOrchestrator, session_id: str):
    print("\n请选择反馈类型：")
    print("  1. 满意（加购/下单）")
    print("  2. 不满意（拒绝推荐）")
    print("  3. 取消")

    choice = input("选择(1/2/3): ").strip()
    if choice == "1":
        orchestrator.record_feedback(session_id, FeedbackSignal.EXPLICIT_POSITIVE)
        print("感谢反馈！已记录正向信号。\n")
    elif choice == "2":
        orchestrator.record_feedback(session_id, FeedbackSignal.EXPLICIT_NEGATIVE)
        print("感谢反馈！已记录负向信号，下次推荐会调整。\n")
    else:
        print("已取消。\n")


def _print_workspace(workspace: dict) -> None:
    print("=" * 60)
    print(f"Plan Workspace: {workspace['title']}")
    print(f"状态: {workspace['status']} | 阶段: {workspace['lifecycle_stage']}")
    if workspace.get("objective"):
        print(f"目标: {workspace['objective']}")
    print("-" * 60)
    for artifact in workspace.get("artifacts", []):
        print(f"[{artifact['artifact_type']}] {artifact['title']}")
        if artifact.get("summary"):
            print(f"  {artifact['summary']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SHOP-PLAN Shopping Agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # chat
    p_chat = subparsers.add_parser("chat", help="交互对话模式")
    p_chat.add_argument("--user-id", default=None)
    p_chat.add_argument("--rl", action="store_true", help="使用 RL-enhanced 策略")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="自动评测模式")
    p_bench.add_argument("--rl", action="store_true", help="使用 RL-enhanced 策略")
    p_bench.add_argument("--compare", action="store_true", help="同时跑两种模式并对比")
    p_bench.add_argument("--baseline-suite", action="store_true", help="运行 full/naive/constraint-only/single-item 对比")
    p_bench.add_argument("--save", action="store_true", help="保存报告到 logs/")
    p_bench.add_argument("--tasks", default=None, help="逗号分隔的 task_id 列表（默认全量）")
    p_bench.add_argument("--quick", action="store_true", help="使用小任务子集快速验证闭环")

    # train
    p_train = subparsers.add_parser("train", help="RL-enhanced 策略训练")
    p_train.add_argument("--iters", type=int, default=200)
    p_train.add_argument("--episodes", type=int, default=32)
    p_train.add_argument("--eval-interval", type=int, default=20)
    p_train.add_argument("--checkpoint-dir", default="checkpoints")

    # catalog
    p_cat = subparsers.add_parser("catalog", help="查看商品目录")
    p_cat.add_argument("--search", default=None, help="关键词搜索")

    # demo
    subparsers.add_parser("demo", help="运行一键 demo（小规模 e2e benchmark）")

    # plan
    p_plan = subparsers.add_parser("plan", help="查看或推进计划工作台")
    p_plan.add_argument("--session-id", required=True, help="目标 session_id")
    p_plan.add_argument("--show", action="store_true", help="显示当前 workspace")
    p_plan.add_argument("--accept", action="store_true", help="接受当前计划")
    p_plan.add_argument("--route", default=None, help="接受时指定路线，例如 分阶段升级")
    p_plan.add_argument("--advance-phase", action="store_true", help="推进到下一阶段")
    p_plan.add_argument("--complete", action="store_true", help="标记计划完成")
    p_plan.add_argument("--archive", action="store_true", help="归档计划")

    args = parser.parse_args()

    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "catalog":
        cmd_catalog(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "plan":
        cmd_plan(args)
    else:
        # 默认行为：兼容旧版（无子命令时进入 chat）
        args.user_id = None
        args.rl = False
        cmd_chat(args)


if __name__ == "__main__":
    main()
