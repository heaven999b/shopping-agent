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
    ):
        self.orchestrator = ShoppingAgentOrchestrator(use_rl=use_rl)
        self.user_id = user_id
        self.tasks_path = tasks_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = MetricsComputer()
        self._use_rl = use_rl

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

            # 构建候选图
            state.transition(WorkflowStep.BUILD_GRAPH)
            self.orchestrator._step_build_graph(state)

            # 规划
            state.transition(WorkflowStep.PLAN)
            self.orchestrator._step_plan(state)

            # 校验
            state.transition(WorkflowStep.VERIFY)
            self.orchestrator._step_verify(state)

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
            print(
                f"  [{status}] {task_id} | score={result.overall_score:.3f} "
                f"| budget={budget_str} | plans={result.num_plans} "
                f"| {result.latency_ms:.0f}ms"
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
        print("\n" + "=" * 50)
        print("Benchmark Summary")
        print("=" * 50)
        print(f"  Tasks:              {metrics['num_tasks']}")
        print(f"  Success Rate:       {metrics['success_rate']:.1%}")
        print(f"  Coverage:           {metrics['coverage']:.1%}")
        print(f"  Budget Satisfied:   {metrics['budget_satisfaction_rate']:.1%}")
        print(f"  Constraint Hit:     {metrics['avg_constraint_hit_rate']:.1%}")
        print(f"  Avg Score:          {metrics['avg_overall_score']:.4f}")
        print(f"  Avg Latency:        {metrics['avg_latency_ms']:.0f} ms")
        print(f"  Avg Budget Ratio:   {metrics['avg_budget_ratio']:.2f}")
        if metrics.get("errors"):
            print(f"  Errors ({metrics['error_rate']:.1%}): {metrics['errors'][:2]}")
        print("=" * 50)
