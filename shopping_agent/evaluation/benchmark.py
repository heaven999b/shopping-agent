"""
BenchmarkRunner — 自动化评测框架。

加载 data/tasks.json，对每个 benchmark task 运行完整 agent pipeline，
收集 TaskResult，调用 MetricsComputer 计算聚合指标。

用法：
    runner = BenchmarkRunner()
    report = runner.run()
    runner.save_report(report)

    # 消融对比（启发式 vs RL）
    runner_rl = BenchmarkRunner(use_rl=True)
    report_rl = runner_rl.run()
    comparison = runner.compare(report, report_rl)
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from shopping_agent.agent.orchestrator import ShoppingAgentOrchestrator
from shopping_agent.common.types import (
    Constraint,
    ConstraintSeverity,
    ShoppingTask,
    TaskType,
)
from shopping_agent.data.loader import load_benchmark_tasks
from shopping_agent.evaluation.metrics import MetricsComputer, TaskResult

import uuid


def _task_from_dict(d: dict) -> ShoppingTask:
    """将 tasks.json 中的 dict 转为 ShoppingTask 对象。"""
    constraints = []
    for key, spec in d.get("constraints", {}).items():
        severity = (
            ConstraintSeverity.HARD
            if spec.get("severity", "hard") == "hard"
            else ConstraintSeverity.SOFT
        )
        constraints.append(Constraint(key=key, value=spec["value"], severity=severity))

    task_type_map = {
        "single": TaskType.SINGLE,
        "bundle": TaskType.BUNDLE,
        "comparison": TaskType.COMPARISON,
        "gift": TaskType.GIFT,
        "replenish": TaskType.REPLENISH,
    }

    return ShoppingTask(
        task_id=d["task_id"],
        task_type=task_type_map.get(d.get("task_type", "single"), TaskType.SINGLE),
        categories=d["categories"],
        raw_query=d["query"],
        constraints=constraints,
        uncertainty_slots=d.get("uncertainty_slots", {}),
        uncertainty_score=len(d.get("uncertainty_slots", {})) * 0.3,
    )


def _check_required_attrs(plan_items: list[dict], required_attrs: list[str]) -> float:
    """
    检查推荐方案是否满足 expected.required_attrs。
    返回命中率（0~1）。
    """
    if not required_attrs:
        return 1.0
    hits = 0
    for item in plan_items:
        attrs = item.get("attributes", {})
        for attr in required_attrs:
            if attrs.get(attr) is True or attrs.get(attr) == "true":
                hits += 1
    return hits / (len(required_attrs) * max(len(plan_items), 1))


class BenchmarkRunner:
    """
    Benchmark 运行器。

    参数：
      use_rl         — 是否使用 RL 策略（True = Option A，False = 启发式 baseline）
      user_id        — 评测用虚拟用户 ID
      tasks_path     — tasks.json 路径（None 则自动查找）
      output_dir     — 报告保存目录
    """

    def __init__(
        self,
        use_rl: bool = False,
        user_id: str = "benchmark_user",
        tasks_path: Optional[str] = None,
        output_dir: str = "logs",
        # ── 消融开关 ──
        disable_graph: bool = False,
        disable_verifier: bool = False,
        disable_clarification: bool = False,
    ):
        self.orchestrator = ShoppingAgentOrchestrator(use_rl=use_rl)
        self.user_id = user_id
        self.tasks_path = tasks_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = MetricsComputer()
        self._use_rl = use_rl
        # 消融开关
        self._disable_graph = disable_graph
        self._disable_verifier = disable_verifier
        self._disable_clarification = disable_clarification

    def run(
        self,
        task_ids: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        运行全量 benchmark，返回报告 dict。

        参数：
          task_ids — 若指定则只运行这些 task（None = 全量）
          verbose  — 是否打印逐任务结果
        """
        raw_tasks = load_benchmark_tasks(self.tasks_path)
        if task_ids:
            raw_tasks = [t for t in raw_tasks if t["task_id"] in task_ids]

        results: list[TaskResult] = []
        for raw in raw_tasks:
            result = self._run_single_task(raw, verbose=verbose)
            results.append(result)

        metrics = self._metrics.compute(results)
        report = {
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "mode": "rl" if self._use_rl else "heuristic",
            "num_tasks": len(results),
            "metrics": metrics,
            "per_task": [r.to_dict() for r in results],
        }

        if verbose:
            self._print_summary(metrics)

        return report

    def _run_single_task(self, raw: dict, verbose: bool = True) -> TaskResult:
        """运行单个 benchmark 任务，返回 TaskResult。"""
        task_id = raw["task_id"]
        query = raw["query"]
        expected = raw.get("expected", {})
        budget = raw.get("constraints", {}).get("budget_total", {}).get("value")

        result = TaskResult(task_id=task_id, query=query)
        t_start = time.time()

        try:
            # 构建 ShoppingTask 并直接注入 orchestrator（跳过 intent parser）
            task = _task_from_dict(raw)
            session_id = str(uuid.uuid4())

            # 直接调用内部步骤（不经过 intent parser，使用已解析的 task）
            from shopping_agent.agent.state import AgentState, WorkflowStep
            state = AgentState(session_id=session_id, user_id=self.user_id)
            state.task = task
            self.orchestrator._sessions[session_id] = state

            # 加载记忆
            state.transition(WorkflowStep.LOAD_MEMORY)
            self.orchestrator._step_load_memory(state)

            # 检索
            state.transition(WorkflowStep.RETRIEVE)
            self.orchestrator._step_retrieve(state)

            # 构建候选图（消融 C 关闭此步）
            if not self._disable_graph:
                state.transition(WorkflowStep.BUILD_GRAPH)
                self.orchestrator._step_build_graph(state)

            # 规划
            state.transition(WorkflowStep.PLAN)
            self.orchestrator._step_plan(state)

            # 校验（消融 A/baseline 关闭此步：直接选最高分方案）
            if not self._disable_verifier:
                state.transition(WorkflowStep.VERIFY)
                self.orchestrator._step_verify(state)
            else:
                # 无 verifier：直接选 overall_score 最高的方案
                if state.candidate_plans:
                    state.selected_plan = max(
                        state.candidate_plans, key=lambda p: p.overall_score
                    )
                    result.verifier_skipped = True

            # 填写结果
            result.has_result = state.selected_plan is not None
            result.num_plans = len(state.candidate_plans)

            if state.selected_plan:
                total = state.selected_plan.total_price
                result.overall_score = state.selected_plan.overall_score
                result.budget_satisfied = (budget is None or total <= budget * 1.05)
                result.budget_ratio = (total / budget) if budget else 0.0

                # 收集 plan items 的属性，用于 constraint_hit_rate
                plan_items_with_attrs = []
                for item in state.selected_plan.items:
                    plan_items_with_attrs.append({
                        "slot": item.bundle_slot,
                        "product_id": item.product.product_id,
                        "price": item.product.final_price,
                        "attributes": {a.name: a.value for a in item.product.attributes},
                    })
                result.plan_items = plan_items_with_attrs

                required_attrs = expected.get("required_attrs", [])
                result.constraint_hit_rate = _check_required_attrs(
                    plan_items_with_attrs, required_attrs
                )

                # Plan diversity: price variation coefficient across candidate plans
                all_plans = state.candidate_plans
                if len(all_plans) >= 2:
                    prices = [p.total_price for p in all_plans]
                    mean_price = sum(prices) / len(prices)
                    if mean_price > 0:
                        import math
                        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
                        result.plan_diversity = math.sqrt(variance) / mean_price

                    scores = [p.overall_score for p in all_plans]
                    score_mean = sum(scores) / len(scores)
                    result.plan_score_variance = sum(
                        (s - score_mean) ** 2 for s in scores
                    ) / len(scores)

                # Graph recovery detection
                result.graph_recovery_used = any(
                    "图修复" in attr.decision or "GraphBased" in attr.module
                    for attr in state.attribution_trace
                )

                result.success = result.has_result and result.budget_satisfied
            else:
                result.success = False

        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}"
            result.success = False
            if verbose:
                print(f"  [ERROR] {task_id}: {result.error}")

        result.latency_ms = (time.time() - t_start) * 1000

        if verbose:
            status = "✓" if result.success else "✗"
            budget_str = f"¥{result.budget_ratio * (budget or 0):.0f}/{budget}" if budget else "N/A"
            recovery_tag = " [graph-repair]" if result.graph_recovery_used else ""
            print(
                f"  [{status}] {task_id} | score={result.overall_score:.3f} "
                f"| budget={budget_str} | plans={result.num_plans} "
                f"| diversity={result.plan_diversity:.3f}"
                f"| {result.latency_ms:.0f}ms{recovery_tag}"
            )

        return result

    def save_report(self, report: dict[str, Any], filename: Optional[str] = None) -> Path:
        """保存评测报告为 JSON 文件。"""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = report.get("mode", "heuristic")
            filename = f"benchmark_{mode}_{ts}.json"

        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {path}")
        return path

    def compare(
        self,
        report_baseline: dict[str, Any],
        report_experiment: dict[str, Any],
    ) -> dict[str, Any]:
        """对比两份报告，返回消融实验结果。"""
        return self._metrics.compare(
            baseline=report_baseline["metrics"],
            experiment=report_experiment["metrics"],
            label_baseline=report_baseline.get("mode", "baseline"),
            label_exp=report_experiment.get("mode", "experiment"),
        )

    @staticmethod
    def _print_summary(metrics: dict[str, Any]) -> None:
        print("\n" + "=" * 55)
        print("Benchmark Summary")
        print("=" * 55)
        print(f"  Tasks:                {metrics['num_tasks']}")
        print(f"  Success Rate:         {metrics['success_rate']:.1%}")
        print(f"  Coverage:             {metrics['coverage']:.1%}")
        print(f"  Budget Satisfied:     {metrics['budget_satisfaction_rate']:.1%}")
        print(f"  Constraint Hit Rate:  {metrics['avg_constraint_hit_rate']:.1%}")
        print(f"  Avg Score:            {metrics['avg_overall_score']:.4f}")
        print(f"  Plan Diversity (CV):  {metrics['avg_plan_diversity']:.4f}")
        print(f"  Avg Budget Ratio:     {metrics['avg_budget_ratio']:.2f}")
        print(f"  Avg Latency:          {metrics['avg_latency_ms']:.0f} ms")
        print(f"  Graph Recovery Rate:  {metrics['graph_recovery_rate']:.1%}")
        if metrics.get("errors"):
            print(f"  Errors ({metrics['error_rate']:.1%}):  {metrics['errors'][:2]}")
        print("=" * 55)
