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
