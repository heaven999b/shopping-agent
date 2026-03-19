"""
tests/test_benchmark.py — BenchmarkRunner 评测模式测试。
"""

from __future__ import annotations

import json

from shopping_agent.evaluation.benchmark import BenchmarkRunner


class TestBenchmarkRunnerModes:
    def test_pipeline_mode_report_marks_mode(self, tmp_path):
        tasks = [
            {
                "task_id": "tb001",
                "query": "帮我买一个降噪耳机，预算2000以内",
                "task_type": "single",
                "categories": ["headset"],
                "constraints": {
                    "budget_total": {"value": 2000.0, "severity": "hard"},
                    "noise_cancelling": {"value": True, "severity": "hard"},
                },
                "uncertainty_slots": {},
                "expected": {"required_attrs": ["noise_cancelling"]},
            }
        ]
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
        )
        report = runner.run(verbose=False)

        assert report["benchmark_mode"] == "pipeline"
        assert report["num_tasks"] == 1

    def test_e2e_mode_handles_clarification_task(self, tmp_path):
        tasks = [
            {
                "task_id": "tb002",
                "query": "想买个耳机",
                "task_type": "single",
                "categories": ["headset"],
                "constraints": {},
                "uncertainty_slots": {
                    "budget_total": None,
                    "usage_scenario": None,
                },
                "clarification_needed": True,
                "expected": {
                    "required_attrs": [],
                    "acceptable_budget_range": [0, 99999],
                    "gold_categories": ["headset"],
                },
            }
        ]
        tasks_path = tmp_path / "tasks_e2e.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="e2e",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
        )
        report = runner.run(verbose=False)

        assert report["benchmark_mode"] == "e2e"
        assert report["num_tasks"] == 1
        assert report["per_task"][0]["has_result"] is True
        assert report["per_task"][0]["clarification_turns"] >= 1
