from __future__ import annotations

from shopping_agent.common.types import UserProfile
from shopping_agent.interaction.clarification import ClarificationPolicy
from tests.conftest import make_task


def _make_profile(stability: float) -> UserProfile:
    return UserProfile(
        user_id="u_clar",
        identity_goal={"professional": 0.8},
        budget_sensitivity_profile={"strict": 0.4, "premium": 0.6},
        brand_orientation={"function_first": 0.4, "brand_signal": 0.8},
        aesthetic_preference={"clean": 0.8, "premium": 0.7},
        persona_stability=stability,
    )


def test_low_stability_profile_triggers_persona_questions():
    profile = _make_profile(0.35)

    task = make_task(
        categories=["headset"],
        budget=2000.0,
        uncertainty_score=0.4,
    )
    task.uncertainty_slots = {"usage_scenario": None}

    policy = ClarificationPolicy()
    questions = policy.generate(task, profile)
    slots = {q.slot for q in questions}

    assert "identity_goal" in slots
    assert "aesthetic_preference" in slots
    assert "budget_flexibility" in slots


def test_stable_profile_skips_persona_questions_when_profile_is_clear():
    profile = _make_profile(0.9)

    task = make_task(
        categories=["headset"],
        budget=2000.0,
        uncertainty_score=0.4,
    )
    task.uncertainty_slots = {"usage_scenario": None}

    policy = ClarificationPolicy()
    questions = policy.generate(task, profile)
    slots = {q.slot for q in questions}

    assert "identity_goal" not in slots
    assert "aesthetic_preference" not in slots
