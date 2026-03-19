"""Stage 3 – Graph Builder: Build a constraint-compatibility graph over retrieved products."""
from __future__ import annotations

import anthropic

from ..models import (
    BenchmarkTask, ClarificationOutput, GraphNode, GraphOutput,
    Product, RetrievalOutput,
)
from .base import call_claude, mock_call

SYSTEM = """You are the Graph Builder Stage of a shopping agent.

Your job:
1. Take the retrieved candidate products and the user's constraints.
2. Score each product's compatibility with every stated constraint (0.0 = violates, 1.0 = fully satisfies).
3. Produce an overall constraint_compatibility score per product (weighted average across constraints).
4. Rank products from most to least compatible.
5. Report which constraints are covered (i.e., at least one product satisfies them) vs. uncoverable.
6. Provide brief reasoning.

This graph structure helps the planner select the best combination.
"""


def run_graph_builder(
    client: anthropic.Anthropic,
    task: BenchmarkTask,
    clarification: ClarificationOutput,
    retrieval: RetrievalOutput,
    catalog: list[Product],
    mock: bool = False,
) -> GraphOutput:
    """Run the graph building stage."""
    if mock:
        nodes = [
            GraphNode(product_id=pid, constraint_compatibility=0.8, rank=i + 1)
            for i, pid in enumerate(retrieval.retrieved_product_ids)
        ]
        return mock_call(
            GraphOutput,
            ranked_nodes=[n.model_dump() for n in nodes],
            constraint_coverage={"budget": True},
            graph_reasoning="mock graph",
        )

    # Build product details for retrieved items only
    id_to_product = {p.id: p for p in catalog}
    retrieved_products = [
        id_to_product[pid] for pid in retrieval.retrieved_product_ids
        if pid in id_to_product
    ]
    product_details = "\n".join(p.compact_repr() for p in retrieved_products)

    # Compile constraints
    constraints_text = []
    if task.constraints.budget_max:
        constraints_text.append(f"Budget max: ${task.constraints.budget_max}")
    if task.constraints.budget_min:
        constraints_text.append(f"Budget min: ${task.constraints.budget_min}")
    for attr, val in task.constraints.required_attributes.items():
        constraints_text.append(f"Required {attr}: {val}")
    for attr, val in clarification.extracted_constraints.items():
        constraints_text.append(f"Extracted {attr}: {val}")
    if not constraints_text:
        constraints_text.append("No hard constraints specified")

    user_msg = (
        f"User query: {clarification.refined_query or task.user_query!r}\n\n"
        f"Constraints:\n" + "\n".join(f"  - {c}" for c in constraints_text) + "\n\n"
        f"Retrieved candidate products:\n{product_details}\n\n"
        "Score each product's compatibility with each constraint and rank them."
    )

    return call_claude(client, SYSTEM, user_msg, GraphOutput, use_thinking=True)
