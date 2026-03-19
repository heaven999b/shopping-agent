"""Stage 4 – Planner: Select the best product(s) satisfying all constraints."""
from __future__ import annotations

import anthropic

from ..models import (
    BenchmarkTask, ClarificationOutput, GraphOutput,
    PlanningOutput, Product, RetrievalOutput,
)
from .base import call_claude, mock_call

SYSTEM = """You are the Planning Stage of a shopping agent.

Your job:
1. Select the best product(s) from the ranked graph to recommend to the user.
2. Ensure all hard constraints (budget, required attributes) are satisfied.
3. Report which constraints are satisfied vs. violated in the final plan.
4. Compute the total price for the recommendation.
5. Assign a confidence score (0.0–1.0) to the plan.
6. Provide clear reasoning for each recommendation.

Rules:
- If the budget constraint is set, the total price of all recommended products MUST be ≤ budget.
- If no product satisfies all constraints, say so clearly and recommend the closest match.
- For multi-item queries (quantity > 1), select the best combination.
- Prefer higher-ranked (more compatible) products from the graph.
"""


def run_planner(
    client: anthropic.Anthropic,
    task: BenchmarkTask,
    clarification: ClarificationOutput,
    retrieval: RetrievalOutput,
    graph: GraphOutput,
    catalog: list[Product],
    mock: bool = False,
) -> PlanningOutput:
    """Run the planning stage."""
    if mock:
        top_ids = [
            (n.product_id if hasattr(n, "product_id") else n["product_id"])
            for n in graph.ranked_nodes[:2]
        ] if graph.ranked_nodes else []
        id_to_product = {p.id: p for p in catalog}
        total = sum(id_to_product[pid].price for pid in top_ids if pid in id_to_product)
        return mock_call(
            PlanningOutput,
            recommended_product_ids=top_ids,
            plan_reasoning="mock plan",
            constraint_satisfaction={"budget": True},
            confidence=0.8,
            total_price=total,
        )

    id_to_product = {p.id: p for p in catalog}

    # Format ranked graph
    graph_text = []
    for node in graph.ranked_nodes:
        pid = node["product_id"] if isinstance(node, dict) else node.product_id
        compat = node["constraint_compatibility"] if isinstance(node, dict) else node.constraint_compatibility
        rank = node["rank"] if isinstance(node, dict) else node.rank
        product = id_to_product.get(pid)
        if product:
            graph_text.append(
                f"Rank {rank}: {product.compact_repr()} | compatibility={compat:.2f}"
            )

    # Constraints summary
    constraints_text = []
    if task.constraints.budget_max:
        constraints_text.append(f"Budget max: ${task.constraints.budget_max}")
    for attr, val in task.constraints.required_attributes.items():
        constraints_text.append(f"Required {attr}: {val}")
    for attr, val in clarification.extracted_constraints.items():
        constraints_text.append(f"Extracted {attr}: {val}")
    if task.constraints.quantity > 1:
        constraints_text.append(f"Number of items needed: {task.constraints.quantity}")

    user_msg = (
        f"User query: {clarification.refined_query or task.user_query!r}\n\n"
        f"Constraints:\n" + "\n".join(f"  - {c}" for c in (constraints_text or ["none"])) + "\n\n"
        f"Ranked candidate products (from graph stage):\n" + "\n".join(graph_text) + "\n\n"
        "Select the best product(s) and produce the final shopping plan."
    )

    return call_claude(client, SYSTEM, user_msg, PlanningOutput, use_thinking=True)
