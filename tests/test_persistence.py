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
from tests.conftest import make_plan, make_product, make_task


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

    def test_response_includes_workspace_artifacts(self, tmp_path):
        db = SQLiteDB(db_path=":memory:")
        session_store = SessionStore(db=db)
        logger = BehaviorLogger(log_path=str(tmp_path / "behavior.jsonl"))
        orchestrator = ShoppingAgentOrchestrator(
            session_store=session_store,
            behavior_logger=logger,
        )

        state = AgentState(session_id="sess_workspace", user_id="user_workspace")
        state.task = make_task(
            task_id="task_workspace",
            categories=["headset", "monitor"],
            budget=5000.0,
        )
        state.user_profile = orchestrator.preference_memory.load("user_workspace")
        state.selected_plan = make_plan(
            plan_id="plan_ws",
            task_id="task_workspace",
            products=[
                make_product(product_id="p1", category="headset", title="办公耳机", price=1200.0),
                make_product(product_id="p2", category="monitor", title="4K显示器", price=2200.0),
            ],
        )
        state.selected_plan.bundle_type = "bundle_plan"
        state.selected_plan.bundle_objective = "办公桌搭补齐方案"
        state.selected_plan.budget_allocation = {"headset": 0.24, "monitor": 0.44}
        state.selected_plan.phased_purchase_options = [
            {"route": "一步到位", "goal": "一次性完成", "budget": 3400},
            {"route": "分阶段升级", "goal": "先核心件后升级", "phase_1_slots": ["monitor"], "phase_1_budget": 2200, "phase_2_slots": ["headset"], "phase_2_budget": 1200},
        ]

        state.current_workspace = orchestrator._build_plan_workspace(state)
        response = orchestrator._build_response(state, "这是当前推荐")

        assert "workspace" in response
        assert response["workspace"]["title"] == "办公桌搭补齐方案"
        artifact_types = {artifact["artifact_type"] for artifact in response["workspace"]["artifacts"]}
        assert "growth_snapshot" in artifact_types
        assert "bundle_recommendation" in artifact_types
        assert "phase_plan" in artifact_types
