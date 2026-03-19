"""Stage 1 – Clarification: Decide if the query needs clarification."""
from __future__ import annotations

import anthropic

from ..models import BenchmarkTask, ClarificationOutput, Product
from .base import call_claude, mock_call

SYSTEM = """You are the Clarification Stage of a shopping agent.

Your job:
1. Analyze the user's shopping query.
2. Decide if clarification is needed (true if query is too vague, ambiguous, or lacks key constraints).
3. If clarification is needed, generate 1-3 targeted clarifying questions.
4. Extract any constraints already present in the query (budget, category, attributes, brand preferences, etc.).
5. Produce a refined query that incorporates any implicit context.

Clarification IS needed when:
- The query is so broad that many unrelated product categories could satisfy it
- Critical constraints (budget, specific use case) are completely absent AND the lack makes retrieval ambiguous
- The query is a single vague word/phrase like "something for commuting" or "a gift"

Clarification is NOT needed when:
- The query clearly identifies a product category with some constraints
- Simple budget/category queries are clear enough for retrieval
"""


def run_clarification(
    client: anthropic.Anthropic,
    task: BenchmarkTask,
    catalog: list[Product],
    mock: bool = False,
) -> ClarificationOutput:
    """Run the clarification stage for a given task."""
    if mock:
        return mock_call(
            ClarificationOutput,
            needs_clarification=task.needs_clarification,
            clarifying_questions=["What is your budget?"] if task.needs_clarification else [],
            extracted_constraints={"budget_max": 200} if task.constraints.budget_max else {},
            refined_query=task.user_query,
            reasoning="mock response",
        )

    # Provide a compact catalog overview to help determine if the query is ambiguous
    category_summary = sorted({p.category for p in catalog})
    user_msg = (
        f"User query: {task.user_query!r}\n\n"
        f"Available product categories: {', '.join(category_summary)}\n\n"
        "Analyze this query and decide if clarification is needed."
    )

    return call_claude(client, SYSTEM, user_msg, ClarificationOutput, use_thinking=False)
