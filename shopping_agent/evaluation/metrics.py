"""
评估指标计算模块。

定义了 TaskResult（单次任务执行结果）和 MetricsComputer（指标计算器）。

核心指标：
  - success_rate          任务成功率（有推荐结果且预算满足）
  - budget_satisfaction   预算满足率（推荐总价 ≤ 预算）
  - constraint_hit_rate   硬约束命中率（推荐商品满足必要属性的比例）
  - avg_overall_score     平均综合得分
  - avg_turns             平均澄清轮数
  - coverage              有推荐结果的任务占比（系统覆盖率）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskResult:
    """
    单次任务执行结果，由 BenchmarkRunner 填写。
    """
    task_id: str
    query: str

    # 执行状态
    success: bool = False           # 有推荐结果且满足预算
    has_result: bool = False        # 是否有任何推荐结果（覆盖率）
    error: Optional[str] = None     # 若抛异常记录错误信息

    # 方案指标
    budget_satisfied: bool = False  # 总价 ≤ 预算
    budget_ratio: float = 0.0       # total_price / budget（越接近 1 越好）
    constraint_hit_rate: float = 0.0  # 必要属性命中率
    overall_score: float = 0.0      # 最优方案综合得分
    num_plans: int = 0              # 生成方案数

    # 过程指标
    clarification_turns: int = 0    # 澄清轮数
    latency_ms: float = 0.0         # 端到端耗时

    # 方案明细（用于 per-item 分析）
    plan_items: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "success": self.success,
            "has_result": self.has_result,
            "error": self.error,
            "budget_satisfied": self.budget_satisfied,
            "budget_ratio": round(self.budget_ratio, 3),
            "constraint_hit_rate": round(self.constraint_hit_rate, 3),
            "overall_score": round(self.overall_score, 3),
            "num_plans": self.num_plans,
            "clarification_turns": self.clarification_turns,
            "latency_ms": round(self.latency_ms, 1),
        }


class MetricsComputer:
    """
    计算一批 TaskResult 的聚合指标。
    """

    def compute(self, results: list[TaskResult]) -> dict[str, Any]:
        """
        返回聚合指标 dict，包含：
          - 任务级统计
          - 质量指标均值
          - 过程指标均值
        """
        if not results:
            return {"num_tasks": 0}

        n = len(results)
        n_success = sum(1 for r in results if r.success)
        n_has_result = sum(1 for r in results if r.has_result)
        n_budget_ok = sum(1 for r in results if r.budget_satisfied)

        avg_constraint_hit = (
            sum(r.constraint_hit_rate for r in results) / n
        )
        avg_score = sum(r.overall_score for r in results) / n
        avg_turns = sum(r.clarification_turns for r in results) / n
        avg_latency = sum(r.latency_ms for r in results) / n

        # 预算利用率（仅计算有结果的任务）
        results_with_result = [r for r in results if r.has_result]
        avg_budget_ratio = (
            sum(r.budget_ratio for r in results_with_result) / len(results_with_result)
            if results_with_result else 0.0
        )

        # 错误分析
        errors = [r.error for r in results if r.error]
        error_rate = len(errors) / n

        return {
            # 基础统计
            "num_tasks": n,
            "num_success": n_success,
            "num_has_result": n_has_result,
            # 核心指标
            "success_rate": round(n_success / n, 4),
            "coverage": round(n_has_result / n, 4),
            "budget_satisfaction_rate": round(n_budget_ok / n, 4),
            "avg_constraint_hit_rate": round(avg_constraint_hit, 4),
            "avg_overall_score": round(avg_score, 4),
            # 过程指标
            "avg_clarification_turns": round(avg_turns, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_budget_ratio": round(avg_budget_ratio, 3),
            # 错误
            "error_rate": round(error_rate, 4),
            "errors": errors[:5],  # 最多显示 5 个错误
        }

    def compare(
        self,
        baseline: dict[str, Any],
        experiment: dict[str, Any],
        label_baseline: str = "baseline",
        label_exp: str = "experiment",
    ) -> dict[str, Any]:
        """
        对比两组指标，计算相对提升（用于 RL vs 启发式 消融实验）。
        """
        metric_keys = [
            "success_rate", "coverage", "budget_satisfaction_rate",
            "avg_constraint_hit_rate", "avg_overall_score",
            "avg_clarification_turns", "avg_budget_ratio",
        ]
        comparison = {}
        for key in metric_keys:
            b = baseline.get(key, 0.0)
            e = experiment.get(key, 0.0)
            delta = e - b
            relative = (delta / b * 100) if b != 0 else float("inf")
            comparison[key] = {
                label_baseline: b,
                label_exp: e,
                "delta": round(delta, 4),
                "relative_pct": round(relative, 2),
            }
        return comparison
