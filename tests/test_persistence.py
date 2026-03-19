"""
tests/test_persistence.py — 会话持久化与结构化日志测试。
"""

from __future__ import annotations

import json

from shopping_agent.agent.orchestrator import ShoppingAgentOrchestrator
from shopping_agent.agent.state import AgentState, WorkflowStep
from shopping_agent.learning.logger import BehaviorLogger
from shopping_agent.storage.db import SQLiteDB
from shopping_agent.storage.session_store import SessionStore
from tests.conftest import make_task


class TestBehaviorLogger:
    def test_log_event_persists_jsonl(self, tmp_path):
        log_path = tmp_path / "behavior.jsonl"
        logger = BehaviorLogger(log_path=str(log_path))

        logger.log_event("workflow_step", session_id="sess001", step="retrieve")

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "workflow_step"
        assert entry["session_id"] == "sess001"
        assert entry["step"] == "retrieve"


class TestOrchestratorPersistence:
    def test_restore_session_from_store(self, tmp_path):
        db = SQLiteDB(db_path=":memory:")
        session_store = SessionStore(db=db)
        logger = BehaviorLogger(log_path=str(tmp_path / "behavior.jsonl"))

        session_store.create("sess_restore", "user_restore")
        state = AgentState(session_id="sess_restore", user_id="user_restore")
        state.task = make_task(task_id="task_restore")
        state.transition(WorkflowStep.CLARIFY)
        state.add_turn("帮我买耳机", "预算大概多少？", clarification_asked="预算大概多少？")
        session_store.update_state(
            session_id="sess_restore",
            task=state.task,
            conversation_history=state.conversation_history,
            status="active",
            current_step=state.current_step.value,
            clarification_rounds_used=1,
        )

        orchestrator = ShoppingAgentOrchestrator(
            session_store=session_store,
            behavior_logger=logger,
        )

        restored = orchestrator._restore_state("sess_restore")

        assert restored.session_id == "sess_restore"
        assert restored.user_id == "user_restore"
        assert restored.task is not None
        assert restored.task.task_id == "task_restore"
        assert restored.current_step == WorkflowStep.CLARIFY
        assert restored.clarification_rounds_used == 1
        assert len(restored.conversation_history) == 1
        assert restored.conversation_history[0].user_input == "帮我买耳机"
