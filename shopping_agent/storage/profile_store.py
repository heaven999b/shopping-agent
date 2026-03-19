"""
ProfileStore — 用户画像持久化存储（SQLite 后端）。

表结构：
  user_profiles(user_id TEXT PK, profile_json TEXT, updated_at TEXT)

替换 PreferenceMemory 的内存存储，实现跨重启数据持久化。
生产环境可将 SQLiteDB 替换为 PostgreSQL，接口不变。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from shopping_agent.common.types import UserProfile
from shopping_agent.storage.db import SQLiteDB, get_db


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id     TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""

_CREATE_SIGNAL_TABLE = """
CREATE TABLE IF NOT EXISTS interaction_signals (
    signal_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL,
    signal_type      TEXT NOT NULL,
    product_attrs_json TEXT NOT NULL DEFAULT '{}',
    dwell_seconds    REAL NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
)
"""

_CREATE_SIGNAL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_interaction_signals_user_id_created_at
ON interaction_signals(user_id, created_at DESC)
"""


def _profile_to_dict(p: UserProfile) -> dict:
    return {
        "user_id": p.user_id,
        "brand_weights": p.brand_weights,
        "price_sensitivity": p.price_sensitivity,
        "platform_preferences": p.platform_preferences,
        "size_profile": p.size_profile,
        "category_affinity": p.category_affinity,
        "style_tags": p.style_tags,
        "interaction_signal_counts": p.interaction_signal_counts,
        "budget_anchor_history": p.budget_anchor_history,
        "prefer_fast_delivery": p.prefer_fast_delivery,
        "prefer_official_store": p.prefer_official_store,
        "last_updated": p.last_updated.isoformat(),
    }


def _dict_to_profile(d: dict) -> UserProfile:
    return UserProfile(
        user_id=d["user_id"],
        brand_weights=d.get("brand_weights", {}),
        price_sensitivity=d.get("price_sensitivity", 0.5),
        platform_preferences=d.get("platform_preferences", {}),
        size_profile=d.get("size_profile", {}),
        category_affinity=d.get("category_affinity", {}),
        style_tags=d.get("style_tags", []),
        interaction_signal_counts=d.get("interaction_signal_counts", {}),
        budget_anchor_history=d.get("budget_anchor_history", []),
        prefer_fast_delivery=d.get("prefer_fast_delivery", False),
        prefer_official_store=d.get("prefer_official_store", True),
        last_updated=datetime.fromisoformat(d.get("last_updated", datetime.now().isoformat())),
    )


class ProfileStore:
    """
    用户画像的 SQLite 持久化存储。

    用法：
        store = ProfileStore()
        profile = store.load("user_001")  # 不存在则返回新建的空白画像
        store.save(profile)
    """

    def __init__(self, db: Optional[SQLiteDB] = None):
        self._db = db or get_db()
        self._db.execute(_CREATE_TABLE)
        self._db.execute(_CREATE_SIGNAL_TABLE)
        self._db.execute(_CREATE_SIGNAL_INDEX)

    def load(self, user_id: str) -> UserProfile:
        """加载用户画像，若不存在则返回空白画像（冷启动）。"""
        row = self._db.fetchone(
            "SELECT profile_json FROM user_profiles WHERE user_id=?",
            (user_id,),
        )
        if row is None:
            return UserProfile(user_id=user_id)
        try:
            d = json.loads(row["profile_json"])
            return _dict_to_profile(d)
        except (json.JSONDecodeError, KeyError):
            return UserProfile(user_id=user_id)

    def save(self, profile: UserProfile) -> None:
        """持久化用户画像（INSERT OR REPLACE）。"""
        profile.last_updated = datetime.now()
        data = json.dumps(_profile_to_dict(profile), ensure_ascii=False)
        self._db.execute(
            "INSERT OR REPLACE INTO user_profiles(user_id, profile_json, updated_at) VALUES(?,?,?)",
            (profile.user_id, data, datetime.now().isoformat()),
        )

    def delete(self, user_id: str) -> None:
        self._db.execute("DELETE FROM user_profiles WHERE user_id=?", (user_id,))

    def list_users(self) -> list[str]:
        rows = self._db.fetchall("SELECT user_id FROM user_profiles ORDER BY updated_at DESC")
        return [r["user_id"] for r in rows]

    def count(self) -> int:
        row = self._db.fetchone("SELECT COUNT(*) as n FROM user_profiles")
        return row["n"] if row else 0

    def log_interaction_signal(
        self,
        user_id: str,
        signal_type: str,
        product_attrs: Optional[dict] = None,
        dwell_seconds: float = 0.0,
    ) -> None:
        self._db.execute(
            """INSERT INTO interaction_signals
               (user_id, signal_type, product_attrs_json, dwell_seconds, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                user_id,
                signal_type,
                json.dumps(product_attrs or {}, ensure_ascii=False),
                dwell_seconds,
                datetime.now().isoformat(),
            ),
        )

    def list_interaction_signals(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict]:
        rows = self._db.fetchall(
            """SELECT signal_type, product_attrs_json, dwell_seconds, created_at
               FROM interaction_signals
               WHERE user_id=?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit),
        )
        result = []
        for row in rows:
            result.append({
                "signal_type": row["signal_type"],
                "product_attrs": json.loads(row["product_attrs_json"] or "{}"),
                "dwell_seconds": row["dwell_seconds"],
                "created_at": row["created_at"],
            })
        return result
