from __future__ import annotations

from shopping_agent.rl.pomdp import Episode
from shopping_agent.rl.trainer import RLTrainer
from shopping_agent.rl.user_simulator import HiddenUserPreference
from tests.conftest import make_task


def test_drift_bonus_rewards_relevant_question():
    pref = HiddenUserPreference(
        budget_total=3000,
        delivery_days=3,
        usage_scenario="office",
        brand_preference=["Sony", "LG"],
        drift_type="budget_drift",
        drift_strength=0.5,
    )

    bonus = RLTrainer._compute_drift_clarification_bonus("budget_total", pref)
    no_bonus = RLTrainer._compute_drift_clarification_bonus("style", pref)

    assert bonus > 0.0
    assert no_bonus == 0.0


def test_drift_adaptation_score_increases_when_relevant_slot_is_resolved():
    pref = HiddenUserPreference(
        budget_total=3000,
        delivery_days=3,
        usage_scenario="office",
        brand_preference=["Sony", "LG"],
        drift_type="budget_drift",
        drift_strength=0.5,
    )
    task = make_task(categories=["headset"], budget=3000.0)
    task.uncertainty_slots = {"budget_total": None, "usage_scenario": None}

    cold_ep = Episode(session_id="s1", user_id="u1", task_id="t1")
    cold_score = RLTrainer._compute_drift_adaptation(task, pref, cold_ep)

    task.uncertainty_slots["budget_total"] = 3000.0
    warm_ep = Episode(session_id="s2", user_id="u1", task_id="t1", final_task_success=True)
    warm_ep.clarification_transitions = [
        type("T", (), {"action": type("A", (), {"slot": "budget_total"})()})()
    ]
    warm_score = RLTrainer._compute_drift_adaptation(task, pref, warm_ep)

    assert warm_score > cold_score
