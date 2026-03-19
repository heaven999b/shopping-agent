"""
tests/test_storage.py — SessionStore / ProfileStore 单元测试。

使用内存 SQLite（db_path=":memory:"）隔离测试，不污染磁盘。
"""

from __future__ import annotations

import pytest

from shopping_agent.common.types import UserProfile
from shopping_agent.storage.db import SQLiteDB
from shopping_agent.storage.profile_store import ProfileStore
from shopping_agent.storage.session_store import SessionStore
from tests.conftest import make_task


@pytest.fixture
def mem_db() -> SQLiteDB:
    """每个测试用独立的内存 DB。"""
    return SQLiteDB(db_path=":memory:")


@pytest.fixture
def session_store(mem_db) -> SessionStore:
    return SessionStore(db=mem_db)


@pytest.fixture
def profile_store(mem_db) -> ProfileStore:
    return ProfileStore(db=mem_db)


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class TestSessionStore:
    def test_create_and_load(self, session_store):
        session_store.create("sess001", "user001")
        record = session_store.load("sess001")
        assert record is not None
        assert record["session_id"] == "sess001"
        assert record["user_id"] == "user001"
        assert record["status"] == "active"

    def test_load_nonexistent_returns_none(self, session_store):
        assert session_store.load("nonexistent") is None

    def test_update_state(self, session_store):
        session_store.create("sess002", "user002")
        session_store.update_state(
            session_id="sess002",
            status="active",
            current_step="clarify",
            clarification_rounds_used=1,
        )
        record = session_store.load("sess002")
        assert record["current_step"] == "clarify"
        assert record["clarification_rounds_used"] == 1

    def test_mark_done(self, session_store):
        session_store.create("sess003", "user003")
        session_store.mark_done("sess003")
        record = session_store.load("sess003")
        assert record["status"] == "done"

    def test_mark_error(self, session_store):
        session_store.create("sess004", "user004")
        session_store.mark_error("sess004")
        record = session_store.load("sess004")
        assert record["status"] == "error"

    def test_list_by_user(self, session_store):
        session_store.create("sess_a", "user_x")
        session_store.create("sess_b", "user_x")
        session_store.create("sess_c", "user_y")
        sessions = session_store.list_by_user("user_x")
        assert len(sessions) == 2
        assert all(s["session_id"] in ("sess_a", "sess_b") for s in sessions)

    def test_count(self, session_store):
        assert session_store.count() == 0
        session_store.create("s1", "u1")
        session_store.create("s2", "u1")
        assert session_store.count() == 2
        assert session_store.count(status="active") == 2
        session_store.mark_done("s1")
        assert session_store.count(status="done") == 1
        assert session_store.count(status="active") == 1

    def test_create_idempotent(self, session_store):
        """INSERT OR IGNORE：重复创建不报错。"""
        session_store.create("s_dup", "u_dup")
        session_store.create("s_dup", "u_dup")  # should not raise
        assert session_store.count() == 1


# ---------------------------------------------------------------------------
# ProfileStore
# ---------------------------------------------------------------------------

class TestProfileStore:
    def test_load_nonexistent_returns_default(self, profile_store):
        profile = profile_store.load("new_user")
        assert profile.user_id == "new_user"
        assert profile.brand_weights == {}
        assert profile.price_sensitivity == 0.5

    def test_save_and_load(self, profile_store):
        profile = UserProfile(
            user_id="u001",
            brand_weights={"Sony": 0.9},
            price_sensitivity=0.7,
            style_tags=["商务", "简约"],
        )
        profile_store.save(profile)
        loaded = profile_store.load("u001")
        assert loaded.user_id == "u001"
        assert loaded.brand_weights == {"Sony": 0.9}
        assert loaded.price_sensitivity == pytest.approx(0.7)
        assert "商务" in loaded.style_tags

    def test_save_overwrites(self, profile_store):
        profile = UserProfile(user_id="u002", price_sensitivity=0.3)
        profile_store.save(profile)
        profile.price_sensitivity = 0.8
        profile_store.save(profile)  # overwrite
        loaded = profile_store.load("u002")
        assert loaded.price_sensitivity == pytest.approx(0.8)

    def test_delete(self, profile_store):
        profile = UserProfile(user_id="u003")
        profile_store.save(profile)
        profile_store.delete("u003")
        loaded = profile_store.load("u003")
        assert loaded.brand_weights == {}  # returns default blank profile

    def test_list_users(self, profile_store):
        for uid in ["ua", "ub", "uc"]:
            profile_store.save(UserProfile(user_id=uid))
        users = profile_store.list_users()
        assert set(users) == {"ua", "ub", "uc"}

    def test_count(self, profile_store):
        assert profile_store.count() == 0
        profile_store.save(UserProfile(user_id="u_cnt"))
        assert profile_store.count() == 1

    def test_log_and_list_interaction_signals(self, profile_store):
        profile_store.log_interaction_signal(
            user_id="u_sig",
            signal_type="dwell",
            product_attrs={"brand": "Sony", "category": "headset"},
            dwell_seconds=45,
        )
        signals = profile_store.list_interaction_signals("u_sig")
        assert len(signals) == 1
        assert signals[0]["signal_type"] == "dwell"
        assert signals[0]["product_attrs"]["brand"] == "Sony"
        assert signals[0]["dwell_seconds"] == pytest.approx(45)
