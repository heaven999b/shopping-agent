"""Stage 5 – Verifier: Verify the plan against all constraints."""
from __future__ import annotations

import anthropic

from ..models import (
    BenchmarkTask, ClarificationOutput, PlanningOutput,
    Product, VerifierOutput,
)
from .base import call_claude, mock_call

SYSTEM = """You are the Verifier Stage of a shopping agent.

Your job:
1. Independently verify the shopping plan against ALL stated constraints.
2. Report any constraint violations (hard failures) and warnings (soft concerns).
3. Assign a confidence score (0.0–1.0) to your verification.
4. Be strict: if any hard constraint is violated, is_valid must be false.

Hard constraints (violations make is_valid = false):
- Budget exceeded (total price > budget_max)
- Required attribute missing (e.g., needs wireless=true but product is wired)
- Wrong product category when explicitly specified

Warnings (soft issues, is_valid can still be true):
- Product is close to but under the budget
- Recommended product may not be the best value
- Constraint partially satisfied
"""


def run_verifier(
    client: anthropic.Anthropic,
    task: BenchmarkTask,
    clarification: ClarificationOutput,
    planning: PlanningOutput,
    catalog: list[Product],
    mock: bool = False,
) -> VerifierOutput:
    """Run the verification stage."""
    if mock:
        id_to_product = {p.id: p for p in catalog}
        violations = []
        # Budget check
        if task.constraints.budget_max and planning.total_price > task.constraints.budget_max:
            violations.append(
                f"Budget exceeded: ${planning.total_price:.2f} > ${task.constraints.budget_max}"
            )
        is_valid = len(violations) == 0
        return mock_call(
            VerifierOutput,
            is_valid=is_valid,
            violations=violations,
            warnings=[],
            confidence=0.9,
            verification_reasoning="mock verification",
        )

    id_to_product = {p.id: p for p in catalog}
    recommended_products = [
        id_to_product[pid] for pid in planning.recommended_product_ids
        if pid in id_to_product
    ]
    product_details = "\n".join(p.compact_repr() for p in recommended_products)

    # Compile all constraints to check
    constraints_text = []
    if task.constraints.budget_max:
        constraints_text.append(f"Budget max: ${task.constraints.budget_max}")
    for attr, val in task.constraints.required_attributes.items():
        constraints_text.append(f"Required {attr}: {val}")
    for attr, val in clarification.extracted_constraints.items():
        constraints_text.append(f"Extracted {attr}: {val}")
    if task.constraints.quantity > 1:
        constraints_text.append(f"Number of items: {task.constraints.quantity}")

    user_msg = (
        f"User query: {clarification.refined_query or task.user_query!r}\n\n"
        f"Constraints to verify:\n" + "\n".join(f"  - {c}" for c in (constraints_text or ["none"])) + "\n\n"
        f"Recommended products:\n{product_details}\n\n"
        f"Plan total price: ${planning.total_price:.2f}\n"
        f"Planner's stated constraint satisfaction: {planning.constraint_satisfaction}\n\n"
        "Independently verify this plan. Report any violations or warnings."
    )

    return call_claude(client, SYSTEM, user_msg, VerifierOutput, use_thinking=True)
