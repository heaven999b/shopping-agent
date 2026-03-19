"""shopping_agent.evaluation — 基准测试与指标计算。"""

__all__ = ["BenchmarkRunner", "MetricsComputer", "TaskResult"]


def __getattr__(name: str):
    if name == "BenchmarkRunner":
        from shopping_agent.evaluation.benchmark import BenchmarkRunner

        return BenchmarkRunner
    if name in {"MetricsComputer", "TaskResult"}:
        from shopping_agent.evaluation.metrics import MetricsComputer, TaskResult

        return {"MetricsComputer": MetricsComputer, "TaskResult": TaskResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
