"""
tests/test_benchmark.py — BenchmarkRunner 评测模式测试。
"""

from __future__ import annotations

import json
from pathlib import Path

from shopping_agent.evaluation.benchmark import BenchmarkRunner
from run_benchmark import (
    _baseline_suite_markdown,
    _main_results_markdown,
    _task_distribution_markdown,
    _task_distribution_summary,
    _write_baseline_suite_csv,
)


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
        assert report["report_schema_version"] == "v2"
        assert report["seed"] == 42
        assert report["deterministic"] is True
        assert "report_sections" in report
        assert (
            report["report_sections"]["evaluation_notes"]["understanding_metrics_mode"]
            == "proxy_from_structured_tasks"
        )
        assert report["report_sections"]["evaluation_notes"]["seed"] == 42
        assert report["report_sections"]["evaluation_notes"]["deterministic"] is True
        assert report["run_id"].startswith("pipeline_full_agent_heuristic_s42")

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
        assert (
            report["report_sections"]["evaluation_notes"]["understanding_metrics_mode"]
            == "end_to_end"
        )
        assert report["per_task"][0]["has_result"] is True
        assert report["per_task"][0]["clarification_turns"] >= 1
        assert "avg_parser_category_match" in report["metrics"]
        assert "clarification_alignment_rate" in report["metrics"]
        assert "avg_intent_resolution_score" in report["metrics"]
        assert "avg_drift_alignment_score" in report["metrics"]
        assert report["per_task"][0]["failure_bucket"] == "success"

    def test_seed_and_deterministic_control_are_configurable(self, tmp_path):
        tasks = [
            {
                "task_id": "tb002b",
                "query": "帮我买个耳机",
                "task_type": "single",
                "categories": ["headset"],
                "constraints": {
                    "budget_total": {"value": 2000.0, "severity": "hard"},
                },
                "uncertainty_slots": {},
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_seed.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            seed=7,
            deterministic=False,
        )
        report = runner.run(verbose=False)

        assert report["seed"] == 7
        assert report["deterministic"] is False
        assert report["report_sections"]["evaluation_notes"]["seed"] == 7
        assert report["report_sections"]["evaluation_notes"]["deterministic"] is False

    def test_benchmark_report_includes_drift_fields(self, tmp_path):
        tasks = [
            {
                "task_id": "tb003",
                "query": "帮我买一个耳机，预算2000",
                "task_type": "single",
                "categories": ["headset"],
                "constraints": {
                    "budget_total": {"value": 2000.0, "severity": "hard"},
                },
                "uncertainty_slots": {},
                "expected": {
                    "required_attrs": [],
                    "drift_expected": False,
                },
            }
        ]
        tasks_path = tmp_path / "tasks_drift.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
        )
        report = runner.run(verbose=False)

        per_task = report["per_task"][0]
        assert "drift_expected" in per_task
        assert "drift_detected" in per_task
        assert "drift_alignment_score" in per_task
        assert "drift_detection_rate" in report["metrics"]

    def test_benchmark_report_summarizes_task_family(self, tmp_path):
        tasks = [
            {
                "task_id": "tb004",
                "query": "想买个耳机",
                "task_type": "single",
                "categories": ["headset"],
                "constraints": {},
                "uncertainty_slots": {
                    "budget_total": None,
                    "usage_scenario": None,
                },
                "clarification_needed": True,
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_family.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
        )
        report = runner.run(verbose=False)

        assert report["per_task"][0]["task_family"] == "clarification_heavy"
        assert "clarification_heavy" in report["metrics"]["task_family_summary"]
        assert "task_family_summary" in report["report_sections"]

    def test_benchmark_report_includes_bundle_and_long_term_metrics(self, tmp_path):
        tasks = [
            {
                "task_id": "tb005",
                "query": "帮我补一套办公桌搭，已经有键盘了，还差显示器和耳机，预算5000",
                "task_type": "bundle",
                "task_family": "upgrade_path",
                "categories": ["monitor", "headset"],
                "constraints": {
                    "budget_total": {"value": 5000.0, "severity": "hard"},
                },
                "user_profile_overrides": {
                    "owned_items": [
                        {
                            "product_id": "owned_keyboard",
                            "category": "keyboard",
                            "brand": "Logitech",
                        }
                    ],
                    "active_setups": {
                        "monitor_setup": {
                            "owned": ["owned_keyboard"],
                            "missing": ["monitor_arm"],
                            "style": "clean",
                            "next_best_upgrade": "monitor_arm",
                        }
                    },
                    "upgrade_stage": {"monitor_setup": "growing"},
                    "purchase_rhythm": {
                        "avg_spend": 1200.0,
                        "purchase_count": 3,
                        "cadence": "incremental",
                    },
                },
                "uncertainty_slots": {},
                "expected": {
                    "required_attrs": [],
                    "gold_categories": ["monitor", "headset"],
                },
            }
        ]
        tasks_path = tmp_path / "tasks_bundle.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
        )
        report = runner.run(verbose=False)

        per_task = report["per_task"][0]
        assert per_task["task_family"] == "upgrade_path"
        assert "avg_style_coherence_score" in report["metrics"]
        assert "avg_bundle_completeness_score" in report["metrics"]
        assert "avg_compatibility_score" in report["metrics"]
        assert "avg_bundle_decision_score" in report["metrics"]
        assert "avg_long_term_fit_score" in report["metrics"]
        assert "avg_phased_purchase_score" in report["metrics"]
        assert (
            report["metrics"]["task_family_summary"]["upgrade_path"]["num_tasks"] == 1
        )

    def test_e2e_user_profile_overrides_apply_before_planning(self, tmp_path):
        tasks = [
            {
                "task_id": "tb005b",
                "query": "帮我补一套办公桌搭，已经有键盘了，还差显示器和耳机，预算5000",
                "task_type": "bundle",
                "task_family": "upgrade_path",
                "categories": ["monitor", "headset"],
                "constraints": {
                    "budget_total": {"value": 5000.0, "severity": "hard"},
                },
                "user_profile_overrides": {
                    "owned_items": [
                        {
                            "product_id": "owned_keyboard",
                            "category": "keyboard",
                            "brand": "Logitech",
                        }
                    ],
                    "active_setups": {
                        "monitor_setup": {
                            "owned": ["owned_keyboard"],
                            "missing": ["monitor_arm"],
                            "style": "clean",
                            "next_best_upgrade": "monitor_arm",
                        }
                    },
                    "upgrade_stage": {"monitor_setup": "growing"},
                    "purchase_rhythm": {
                        "avg_spend": 1200.0,
                        "purchase_count": 3,
                        "cadence": "incremental",
                    },
                },
                "uncertainty_slots": {},
                "expected": {
                    "required_attrs": [],
                    "gold_categories": ["monitor", "headset"],
                },
            }
        ]
        tasks_path = tmp_path / "tasks_e2e_profile.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="e2e",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            baseline_profile="no_memory",
        )
        report = runner.run(verbose=False)

        assert report["per_task"][0]["original_task_family"] == "upgrade_path"
        assert report["per_task"][0]["long_term_fit_score"] == 0.0

    def test_parser_match_uses_original_gold_categories(self, tmp_path):
        tasks = [
            {
                "task_id": "tb005c",
                "query": "帮我补一套桌搭，显示器和耳机都要",
                "task_type": "bundle",
                "categories": ["monitor", "headset"],
                "constraints": {
                    "budget_total": {"value": 5000.0, "severity": "hard"},
                },
                "expected": {"gold_categories": ["monitor", "headset"]},
            }
        ]
        tasks_path = tmp_path / "tasks_parser_eval.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            baseline_profile="single_item",
        )
        report = runner.run(verbose=False)

        assert report["per_task"][0]["original_task_family"] == "bundle"
        assert report["per_task"][0]["parser_category_match"] < 1.0

    def test_save_report_writes_summary_and_metrics_artifacts(self, tmp_path):
        tasks = [
            {
                "task_id": "tb006",
                "query": "帮我买一个降噪耳机，预算2000以内",
                "task_type": "single",
                "categories": ["headset"],
                "constraints": {
                    "budget_total": {"value": 2000.0, "severity": "hard"},
                },
                "uncertainty_slots": {},
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_save.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
        )
        report = runner.run(verbose=False)
        path = runner.save_report(report, filename="benchmark_test.json")

        assert path.exists()
        assert (tmp_path / "logs" / "benchmark_test_summary.json").exists()
        assert (tmp_path / "logs" / "benchmark_test_metrics.json").exists()
        assert (tmp_path / "logs" / "benchmark_test_per_task.json").exists()

    def test_baseline_profile_is_reflected_in_report(self, tmp_path):
        tasks = [
            {
                "task_id": "tb007",
                "query": "帮我补一套桌搭，已有键盘",
                "task_type": "bundle",
                "categories": ["monitor", "headset"],
                "constraints": {
                    "budget_total": {"value": 5000.0, "severity": "hard"},
                },
                "uncertainty_slots": {},
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_baseline.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            baseline_profile="single_item",
        )
        report = runner.run(verbose=False)

        assert report["baseline_profile"] == "single_item"

    def test_single_item_baseline_preserves_original_bundle_family(self, tmp_path):
        tasks = [
            {
                "task_id": "tb007b",
                "query": "帮我补一套桌搭，已有键盘，还缺显示器和耳机",
                "task_type": "bundle",
                "categories": ["monitor", "headset"],
                "constraints": {
                    "budget_total": {"value": 5000.0, "severity": "hard"},
                },
                "uncertainty_slots": {},
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_baseline_family.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            baseline_profile="single_item",
        )
        report = runner.run(verbose=False)

        per_task = report["per_task"][0]
        assert per_task["task_family"] == "general"
        assert per_task["original_task_family"] == "bundle"
        assert report["metrics"]["bundle_summary"]["num_tasks"] == 1

    def test_no_bundle_scoring_baseline_is_reflected_in_report(self, tmp_path):
        tasks = [
            {
                "task_id": "tb008",
                "query": "帮我配一套办公桌搭，显示器和键盘都要",
                "task_type": "bundle",
                "categories": ["monitor", "keyboard"],
                "constraints": {
                    "budget_total": {"value": 5000.0, "severity": "hard"},
                },
                "uncertainty_slots": {},
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_no_bundle.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            baseline_profile="no_bundle_scoring",
        )
        report = runner.run(verbose=False)

        assert report["baseline_profile"] == "no_bundle_scoring"

    def test_no_memory_baseline_clears_long_horizon_profile(self, tmp_path):
        tasks = [
            {
                "task_id": "tb008b",
                "query": "帮我在已有桌搭基础上补一套升级方案",
                "task_type": "bundle",
                "task_family": "upgrade_path",
                "categories": ["monitor", "headset"],
                "constraints": {
                    "budget_total": {"value": 5000.0, "severity": "hard"},
                },
                "user_profile_overrides": {
                    "owned_items": [
                        {
                            "product_id": "owned_keyboard",
                            "category": "keyboard",
                            "brand": "Logitech",
                        }
                    ],
                    "active_setups": {
                        "monitor_setup": {
                            "owned": ["owned_keyboard"],
                            "missing": ["monitor"],
                            "style": "clean",
                        }
                    },
                    "upgrade_stage": {"monitor_setup": "growing"},
                    "purchase_rhythm": {
                        "avg_spend": 1200.0,
                        "purchase_count": 3,
                        "cadence": "incremental",
                    },
                    "identity_goal": {"professional": 0.8},
                },
                "uncertainty_slots": {},
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_no_memory.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            baseline_profile="no_memory",
        )
        report = runner.run(verbose=False)

        assert report["baseline_profile"] == "no_memory"
        state = next(iter(runner.orchestrator._sessions.values()))
        assert state.user_profile is not None
        assert state.user_profile.identity_goal == {"professional": 0.8}
        assert state.user_profile.owned_items == []
        assert state.user_profile.active_setups == {}
        assert state.user_profile.upgrade_stage == {}
        assert state.user_profile.purchase_rhythm == {}

    def test_no_constraint_baseline_strips_constraints_from_task(self, tmp_path):
        tasks = [
            {
                "task_id": "tb008c",
                "query": "预算 5000 内给我配显示器和键盘",
                "task_type": "bundle",
                "categories": ["monitor", "keyboard"],
                "constraints": {
                    "budget_total": {"value": 5000.0, "severity": "hard"},
                    "noise_cancelling": {"value": True, "severity": "soft"},
                },
                "uncertainty_slots": {},
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_no_constraint.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            baseline_profile="no_constraint",
        )
        runner.run(verbose=False)

        state = next(iter(runner.orchestrator._sessions.values()))
        assert state.task is not None
        assert state.task.constraints == []

    def test_no_clarification_baseline_clears_uncertainty_slots(self, tmp_path):
        tasks = [
            {
                "task_id": "tb008d",
                "query": "想配桌搭但预算和用途还没想好",
                "task_type": "bundle",
                "categories": ["monitor", "keyboard"],
                "constraints": {},
                "uncertainty_slots": {
                    "budget_total": None,
                    "usage_scenario": None,
                },
                "clarification_needed": True,
                "expected": {"required_attrs": []},
            }
        ]
        tasks_path = tmp_path / "tasks_no_clarification.json"
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        runner = BenchmarkRunner(
            benchmark_mode="pipeline",
            tasks_path=str(tasks_path),
            output_dir=str(tmp_path / "logs"),
            baseline_profile="no_clarification",
        )
        runner.run(verbose=False)

        state = next(iter(runner.orchestrator._sessions.values()))
        assert state.task is not None
        assert state.task.uncertainty_slots == {}

    def test_baseline_suite_writers_emit_markdown_and_csv(self, tmp_path):
        reports = {
            "full_agent": {
                "per_task": [
                    {
                        "task_family": "bundle",
                        "original_task_family": "bundle",
                        "task_type": "bundle",
                        "task_difficulty": "hard",
                        "clarification_needed": False,
                    }
                ],
                "metrics": {
                    "success_rate": 0.8,
                    "avg_budget_ratio": 0.72,
                    "budget_satisfaction_rate": 0.9,
                    "avg_bundle_decision_score": 0.81,
                    "avg_bundle_completeness_score": 0.95,
                    "avg_compatibility_score": 0.88,
                    "avg_long_term_fit_score": 0.76,
                },
            },
            "no_memory": {
                "per_task": [
                    {
                        "task_family": "upgrade_path",
                        "original_task_family": "upgrade_path",
                        "task_type": "bundle",
                        "task_difficulty": "medium",
                        "clarification_needed": False,
                    }
                ],
                "metrics": {
                    "success_rate": 0.74,
                    "avg_budget_ratio": 0.58,
                    "budget_satisfaction_rate": 0.88,
                    "avg_bundle_decision_score": 0.57,
                    "avg_bundle_completeness_score": 0.83,
                    "avg_compatibility_score": 0.82,
                    "avg_long_term_fit_score": 0.03,
                },
            },
            "single_item": {
                "per_task": [
                    {
                        "task_family": "clarification_heavy",
                        "original_task_family": "clarification_heavy",
                        "task_type": "single",
                        "task_difficulty": "easy",
                        "clarification_needed": True,
                    }
                ],
                "metrics": {
                    "success_rate": 0.6,
                    "avg_budget_ratio": 0.51,
                    "budget_satisfaction_rate": 0.84,
                    "avg_bundle_decision_score": 0.32,
                    "avg_bundle_completeness_score": 0.41,
                    "avg_compatibility_score": 0.35,
                    "avg_long_term_fit_score": 0.29,
                },
            },
        }

        md = _baseline_suite_markdown(reports)
        assert "## All-Task Summary" in md
        assert "## Bundle-Only Summary" in md
        assert "AllTaskSuccess" in md
        assert "BundleTasks" in md
        assert "RelationCoverage" in md
        assert "full_agent" in md
        assert "no_memory" in md

        csv_path = Path(tmp_path / "baseline_suite.csv")
        _write_baseline_suite_csv(csv_path, reports)
        text = csv_path.read_text(encoding="utf-8")
        assert (
            "method,all_task_success_rate,coverage,budget_satisfaction_rate,bundle_success_rate,bundle_task_count,cost_band,bundle_score"
            in text
        )
        assert "relation_coverage" in text
        assert "full_agent" in text
        assert "no_memory" in text

        distribution = _task_distribution_summary(reports)
        assert distribution["num_tasks"] == 1
        assert distribution["bundle_like_tasks"] == 1
        dist_md = _task_distribution_markdown(distribution)
        assert "Task Distribution" in dist_md
        assert "BundleLikeTasks" in dist_md

        main_md = _main_results_markdown(reports)
        assert "Main Results" in main_md
        assert "Primary Bundle Comparisons" in main_md
        assert "Interpretation Notes" in main_md
