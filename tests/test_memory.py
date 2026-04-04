"""
tests/test_memory.py — PreferenceMemory / PreferenceUpdater 持久化测试。
"""

from __future__ import annotations

import pytest

from shopping_agent.common.types import FeedbackRecord, FeedbackSignal
from shopping_agent.learning.preference_updater import PreferenceUpdater
from shopping_agent.memory.preference_memory import PreferenceMemory
from shopping_agent.storage.db import SQLiteDB
from shopping_agent.storage.profile_store import ProfileStore
from tests.conftest import make_task


@pytest.fixture
def persistent_memory() -> PreferenceMemory:
    db = SQLiteDB(db_path=":memory:")
    store = ProfileStore(db=db)
    return PreferenceMemory(profile_store=store)


class TestPreferenceMemoryPersistence:
    def test_update_explicit_persists_across_instances(self):
        db = SQLiteDB(db_path=":memory:")
        store = ProfileStore(db=db)

        memory_a = PreferenceMemory(profile_store=store)
        memory_a.update_explicit(
            "user001",
            "purchase",
            {"brand": "Sony", "category": "headset"},
            delta=0.15,
        )

        memory_b = PreferenceMemory(profile_store=store)
        profile = memory_b.load("user001")

        assert profile.brand_weights["Sony"] == pytest.approx(0.65)
        assert profile.category_affinity["headset"] == pytest.approx(0.15)

    def test_contextualized_load_does_not_mutate_persisted_profile(self, persistent_memory):
        profile = persistent_memory.load("user_ctx")
        profile.price_sensitivity = 0.4
        persistent_memory.save(profile)

        task = make_task(
            budget=500.0,
            raw_query="出差用耳机，预算500",
        )
        task.implicit_needs = ["出差商务"]

        contextualized = persistent_memory.load("user_ctx", task)
        reloaded = persistent_memory.load("user_ctx")

        assert contextualized.prefer_fast_delivery is True
        assert contextualized.price_sensitivity == pytest.approx(0.6)
        assert reloaded.prefer_fast_delivery is False
        assert reloaded.price_sensitivity == pytest.approx(0.4)


class TestPreferenceUpdaterPersistence:
    def test_revision_updates_are_saved(self):
        db = SQLiteDB(db_path=":memory:")
        store = ProfileStore(db=db)
        memory = PreferenceMemory(profile_store=store)
        updater = PreferenceUpdater(memory=memory)

        profile = memory.load("user_rev")
        profile.price_sensitivity = 0.5
        memory.save(profile)

        updater.update(
            "user_rev",
            FeedbackRecord(
                session_id="sess001",
                task_id="task001",
                plan_id=None,
                signal=FeedbackSignal.REVISION,
                context={"old_budget": 1000, "new_budget": 1500},
            ),
        )

        reloaded = PreferenceMemory(profile_store=store).load("user_rev")
        assert reloaded.price_sensitivity == pytest.approx(0.45)
        assert reloaded.budget_sensitivity_profile["premium"] > 0.0
        assert reloaded.recent_persona_drift["type"] == "budget_drift"
        assert reloaded.persona_transition_log

    def test_style_revision_updates_persona_drift(self):
        db = SQLiteDB(db_path=":memory:")
        store = ProfileStore(db=db)
        memory = PreferenceMemory(profile_store=store)
        updater = PreferenceUpdater(memory=memory)

        updater.update(
            "user_style",
            FeedbackRecord(
                session_id="sess002",
                task_id="task002",
                plan_id=None,
                signal=FeedbackSignal.REVISION,
                context={"old_style": "clean", "new_style": "premium"},
            ),
        )

        reloaded = PreferenceMemory(profile_store=store).load("user_style")
        assert reloaded.aesthetic_preference["premium"] > 0.0
        assert reloaded.recent_persona_drift["type"] == "style_drift"
        assert reloaded.persona_stability < 0.5

    def test_implicit_signals_are_persisted_and_reused(self):
        db = SQLiteDB(db_path=":memory:")
        store = ProfileStore(db=db)
        memory = PreferenceMemory(profile_store=store)

        memory.update_implicit(
            "user_sig",
            "dwell",
            {"brand": "Sony", "category": "headset", "price": 999},
            dwell_seconds=40,
        )

        signals = PreferenceMemory(profile_store=store).get_recent_signals("user_sig")
        assert len(signals) >= 1
        assert signals[0]["signal_type"] in {"dwell", "add_to_cart"}

        task = make_task(categories=["headset"], budget=None)
        contextualized = PreferenceMemory(profile_store=store).load("user_sig", task)
        assert contextualized.brand_weights.get("Sony", 0.0) > 0.5

    def test_explicit_persona_tags_update_profile(self):
        db = SQLiteDB(db_path=":memory:")
        store = ProfileStore(db=db)
        memory = PreferenceMemory(profile_store=store)

        memory.update_explicit(
            "user_persona",
            "purchase",
            {
                "brand": "Sony",
                "category": "headset",
                "price": 1999,
                "persona_tags": {
                    "identity_fit": ["professional"],
                    "style_signal": ["clean", "premium"],
                    "symbolic_value": "taste_signaling",
                },
            },
            delta=0.1,
        )

        profile = PreferenceMemory(profile_store=store).load("user_persona")
        assert profile.identity_goal["professional"] > 0.0
        assert profile.aesthetic_preference["clean"] > 0.0
        assert profile.brand_orientation["brand_signal"] > 0.0

    def test_purchase_updates_long_term_growth_state(self):
        db = SQLiteDB(db_path=":memory:")
        store = ProfileStore(db=db)
        memory = PreferenceMemory(profile_store=store)

        memory.update_explicit(
            "user_growth",
            "purchase",
            {
                "product_id": "monitor_001",
                "title": "LG 27寸显示器",
                "brand": "LG",
                "category": "monitor",
                "price": 2499,
                "persona_tags": {
                    "style_signal": ["clean", "premium"],
                    "identity_fit": ["professional"],
                    "symbolic_value": "taste_signaling",
                },
            },
            delta=0.1,
        )

        profile = PreferenceMemory(profile_store=store).load("user_growth")
        assert any(item["product_id"] == "monitor_001" for item in profile.owned_items)
        assert "monitor_setup" in profile.active_setups
        assert profile.active_setups["monitor_setup"]["next_best_upgrade"] == "monitor_arm"
        assert profile.upgrade_stage["monitor_setup"] == "starter"
        assert profile.purchase_rhythm["purchase_count"] == 1
