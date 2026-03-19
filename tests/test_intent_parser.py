"""
tests/test_intent_parser.py — IntentParser 增量修正测试。
"""

from __future__ import annotations

from shopping_agent.agent.state import ConversationTurn
from shopping_agent.interaction.intent_parser import IntentParser
from tests.conftest import make_task


class TestIntentParserRevisionFallback:
    def test_parse_revision_preserves_existing_task_context_without_llm(self):
        parser = IntentParser()
        task = make_task(
            task_id="t_rev",
            categories=["headset"],
            budget=None,
            raw_query="想买个耳机",
            uncertainty_score=0.9,
        )
        task.uncertainty_slots = {"budget_total": None, "usage_scenario": None}

        history = [
            ConversationTurn(
                round_index=0,
                user_input="想买个耳机",
                agent_response="您的总预算大概是多少？",
            )
        ]

        revised = parser.parse_revision("预算2000以内", task, history)

        assert revised.categories == ["headset"]
        assert revised.get_hard_constraints()["budget_total"] == 2000.0
        assert "budget_total" not in revised.uncertainty_slots
        assert len(revised.revision_history) == 1

    def test_parse_revision_can_fill_categories_from_clarification_answer(self):
        parser = IntentParser()
        task = make_task(
            task_id="t_rev_categories",
            categories=[],
            budget=None,
            raw_query="给我推荐一套设备",
            uncertainty_score=0.9,
        )
        task.uncertainty_slots = {"categories": None}

        history = [
            ConversationTurn(
                round_index=0,
                user_input="给我推荐一套设备",
                agent_response="您想买什么，或者需要哪些品类？",
            )
        ]

        revised = parser.parse_revision("显示器和键盘", task, history)

        assert set(revised.categories) == {"monitor", "keyboard"}
        assert "categories" not in revised.uncertainty_slots
