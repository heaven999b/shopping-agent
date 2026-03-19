"""Stage 2 – Retrieval: Find relevant products from the catalog."""
from __future__ import annotations

import anthropic

from ..models import BenchmarkTask, ClarificationOutput, Product, RetrievalOutput
from .base import call_claude, mock_call

SYSTEM = """You are the Retrieval Stage of a shopping agent.

Your job:
1. Given a user query (possibly refined after clarification), retrieve the most relevant products.
2. Assign a relevance score (0.0–1.0) to each retrieved product.
3. Return the top-K most relevant product IDs (aim for 3–8 products, including all plausible matches).
4. Retrieve generously – it is better to include borderline candidates than to miss a valid option.
5. Explain your retrieval reasoning briefly.

Key rules:
- Only retrieve products that exist in the provided catalog.
- Do NOT invent product IDs.
- Include ALL products that could plausibly satisfy the query, even if they're slightly outside stated constraints – the verifier will filter later.
"""


def run_retrieval(
    client: anthropic.Anthropic,
    task: BenchmarkTask,
    clarification: ClarificationOutput,
    catalog: list[Product],
    mock: bool = False,
) -> RetrievalOutput:
    """Run the retrieval stage."""
    if mock:
        # For mock: return first 3 products in matching category or just first 3
        category = task.category.split("/")[0] if "/" in task.category else task.category
        matches = [p.id for p in catalog if category in p.category][:3]
        if not matches:
            matches = [p.id for p in catalog[:3]]
        return mock_call(
            RetrievalOutput,
            retrieved_product_ids=matches + task.acceptable_product_ids[:2],
            relevance_scores={pid: 0.8 for pid in matches},
            retrieval_reasoning="mock retrieval",
        )

    query = clarification.refined_query or task.user_query
    catalog_text = "\n".join(p.compact_repr() for p in catalog)

    # Surface extracted constraints from clarification
    constraint_notes = ""
    if clarification.extracted_constraints:
        constraint_notes = f"\nExtracted constraints: {clarification.extracted_constraints}"
    if task.constraints.budget_max:
        constraint_notes += f"\nBudget max: ${task.constraints.budget_max}"

    user_msg = (
        f"User query: {query!r}{constraint_notes}\n\n"
        f"Product catalog ({len(catalog)} items):\n{catalog_text}\n\n"
        "Retrieve the most relevant products. Return their IDs and relevance scores."
    )

    return call_claude(client, SYSTEM, user_msg, RetrievalOutput, use_thinking=False)
