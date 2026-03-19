"""
SessionStore — 会话持久化存储（SQLite 后端）。

表结构：
  sessions(
    session_id      TEXT PK,
    user_id         TEXT,
    task_json       TEXT,          -- ShoppingTask (序列化)
    conversation_json TEXT,        -- 对话历史 (序列化)
    status          TEXT,          -- "active"|"done"|"error"
    current_step    TEXT,
    clarification_rounds_used INT,
    created_at      TEXT,
    updated_at      TEXT
  )

用途：
  - 支持跨进程/重启的多轮对话续接
  - 提供会话审计日志
  - 为 RL 轨迹回溯提供会话上下文
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from shopping_agent.agent.state import ConversationTurn
from shopping_agent.common.types import (
    ConflictPair,
    ConflictResolution,
    Constraint,
    ConstraintSeverity,
    ShoppingTask,
    TaskRevision,
    TaskType,
)
from shopping_agent.storage.db import SQLiteDB, get_db


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT PRIMARY KEY,
    user_id                 TEXT NOT NULL,
    task_json               TEXT,
    conversation_json       TEXT NOT NULL DEFAULT '[]',
    status                  TEXT NOT NULL DEFAULT 'active',
    current_step            TEXT NOT NULL DEFAULT 'init',
    clarification_rounds_used INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)
"""


def _task_to_dict(task: Any) -> Optional[dict]:
    """将 ShoppingTask 序列化为 JSON-safe dict。"""
    if task is None:
        return None
    return {
        "task_id": task.task_id,
        "task_type": task.task_type.value,
        "categories": task.categories,
        "raw_query": task.raw_query,
        "constraints": [
            {
                "key": c.key,
                "value": c.value,
                "severity": c.severity.value,
                "source": c.source,
            }
            for c in task.constraints
        ],
        "implicit_needs": task.implicit_needs,
        "conflict_pairs": [
            {
                "constraint_a": cp.constraint_a,
                "constraint_b": cp.constraint_b,
                "description": cp.description,
                "resolution": cp.resolution.value,
            }
            for cp in task.conflict_pairs
        ],
        "uncertainty_slots": task.uncertainty_slots,
        "uncertainty_score": task.uncertainty_score,
        "revision_history": [
            {
                "round_index": rev.round_index,
                "changed_fields": rev.changed_fields,
                "user_utterance": rev.user_utterance,
                "timestamp": rev.timestamp.isoformat(),
            }
            for rev in task.revision_history
        ],
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }


def _conversation_to_list(history: list) -> list[dict]:
    """将 ConversationTurn 列表序列化为 JSON-safe list。"""
    return [
        {
            "round_index": t.round_index,
            "user_input": t.user_input,
            "agent_response": t.agent_response,
            "clarification_asked": t.clarification_asked,
            "timestamp": t.timestamp.isoformat(),
        }
        for t in history
    ]


def _dict_to_task(data: dict) -> ShoppingTask:
    constraints = [
        Constraint(
            key=item["key"],
            value=item.get("value"),
            severity=ConstraintSeverity(item.get("severity", ConstraintSeverity.HARD.value)),
            source=item.get("source", "user"),
        )
        for item in data.get("constraints", [])
    ]
    conflict_pairs = [
        ConflictPair(
            constraint_a=item.get("constraint_a", ""),
            constraint_b=item.get("constraint_b", ""),
            description=item.get("description", ""),
            resolution=ConflictResolution(
                item.get("resolution", ConflictResolution.ASK_USER.value)
            ),
        )
        for item in data.get("conflict_pairs", [])
    ]
    revision_history = [
        TaskRevision(
            round_index=item.get("round_index", 0),
            changed_fields=item.get("changed_fields", {}),
            user_utterance=item.get("user_utterance", ""),
            timestamp=datetime.fromisoformat(
                item.get("timestamp", datetime.now().isoformat())
            ),
        )
        for item in data.get("revision_history", [])
    ]
    return ShoppingTask(
        task_id=data["task_id"],
        task_type=TaskType(data.get("task_type", TaskType.SINGLE.value)),
        categories=data.get("categories", []),
        raw_query=data.get("raw_query", ""),
        constraints=constraints,
        implicit_needs=data.get("implicit_needs", []),
        conflict_pairs=conflict_pairs,
        uncertainty_slots=data.get("uncertainty_slots", {}),
        uncertainty_score=float(data.get("uncertainty_score", 1.0)),
        revision_history=revision_history,
        created_at=datetime.fromisoformat(
            data.get("created_at", datetime.now().isoformat())
        ),
        updated_at=datetime.fromisoformat(
            data.get("updated_at", datetime.now().isoformat())
        ),
    )


def _list_to_conversation(history: list[dict]) -> list[ConversationTurn]:
    return [
        ConversationTurn(
            round_index=item.get("round_index", 0),
            user_input=item.get("user_input", ""),
            agent_response=item.get("agent_response", ""),
            clarification_asked=item.get("clarification_asked"),
            timestamp=datetime.fromisoformat(
                item.get("timestamp", datetime.now().isoformat())
            ),
        )
        for item in history
    ]


class SessionStore:
    """
    会话持久化存储。

    用法：
        store = SessionStore()
        store.create(session_id="s001", user_id="u001")
        store.update_state(session_id="s001", task=state.task, ...)
        record = store.load("s001")       # 返回 dict
        history = store.list_by_user("u001")
    """

    def __init__(self, db: Optional[SQLiteDB] = None):
        self._db = db or get_db()
        self._db.execute(_CREATE_TABLE)
        self._db.execute(_CREATE_INDEX)

    def create(self, session_id: str, user_id: str) -> None:
        """创建新会话记录。"""
        now = datetime.now().isoformat()
        self._db.execute(
            """INSERT OR IGNORE INTO sessions
               (session_id, user_id, task_json, conversation_json,
                status, current_step, clarification_rounds_used, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (session_id, user_id, None, "[]", "active", "init", 0, now, now),
        )

    def update_state(
        self,
        session_id: str,
        task: Any = None,
        conversation_history: Optional[list] = None,
        status: str = "active",
        current_step: str = "init",
        clarification_rounds_used: int = 0,
    ) -> None:
        """将会话状态的关键字段同步到 DB。"""
        task_json = json.dumps(_task_to_dict(task), ensure_ascii=False) if task else None
        conv_json = json.dumps(
            _conversation_to_list(conversation_history or []),
            ensure_ascii=False,
        )
        self._db.execute(
            """UPDATE sessions SET
               task_json=?, conversation_json=?, status=?,
               current_step=?, clarification_rounds_used=?, updated_at=?
               WHERE session_id=?""",
            (
                task_json, conv_json, status,
                current_step, clarification_rounds_used,
                datetime.now().isoformat(), session_id,
            ),
        )

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        """加载会话记录，返回 dict（不重建 AgentState，避免复杂反序列化）。"""
        row = self._db.fetchone(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        )
        if row is None:
            return None
        result = dict(row)
        if result.get("task_json"):
            try:
                result["task"] = _dict_to_task(json.loads(result["task_json"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                result["task"] = None
        conversation = json.loads(result.get("conversation_json") or "[]")
        result["conversation"] = _list_to_conversation(conversation)
        return result

    def list_by_user(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """查询用户最近的会话列表。"""
        rows = self._db.fetchall(
            "SELECT session_id, status, current_step, created_at, updated_at "
            "FROM sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in rows]

    def mark_done(self, session_id: str) -> None:
        self._db.execute(
            "UPDATE sessions SET status='done', updated_at=? WHERE session_id=?",
            (datetime.now().isoformat(), session_id),
        )

    def mark_error(self, session_id: str) -> None:
        self._db.execute(
            "UPDATE sessions SET status='error', updated_at=? WHERE session_id=?",
            (datetime.now().isoformat(), session_id),
        )

    def count(self, status: Optional[str] = None) -> int:
        if status:
            row = self._db.fetchone(
                "SELECT COUNT(*) as n FROM sessions WHERE status=?", (status,)
            )
        else:
            row = self._db.fetchone("SELECT COUNT(*) as n FROM sessions")
        return row["n"] if row else 0
