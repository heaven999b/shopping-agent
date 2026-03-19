"""Shopping Agent Pipeline – orchestrates all 5 stages."""
from __future__ import annotations

import anthropic

from ..models import BenchmarkTask, PipelineResult, Product
from .clarification import run_clarification
from .retrieval import run_retrieval
from .graph_builder import run_graph_builder
from .planner import run_planner
from .verifier import run_verifier


class ShoppingAgentPipeline:
    """
    Five-stage shopping agent pipeline:

        clarification → retrieval → graph → planning → verifier

    Each stage uses claude-opus-4-6 and feeds its output into the next stage.
    Results are structured Pydantic models, enabling per-stage evaluation.
    """

    def __init__(self, api_key: str | None = None, mock: bool = False):
        self.mock = mock
        if not mock:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None  # type: ignore

    def run(
        self,
        task: BenchmarkTask,
        catalog: list[Product],
        run_index: int = 0,
    ) -> PipelineResult:
        """Execute the full pipeline for a single task."""
        client = self.client  # type: ignore

        # Stage 1: Clarification
        clarification = run_clarification(client, task, catalog, mock=self.mock)

        # Stage 2: Retrieval
        retrieval = run_retrieval(client, task, clarification, catalog, mock=self.mock)

        # Stage 3: Graph Building
        graph = run_graph_builder(
            client, task, clarification, retrieval, catalog, mock=self.mock
        )

        # Stage 4: Planning
        planning = run_planner(
            client, task, clarification, retrieval, graph, catalog, mock=self.mock
        )

        # Stage 5: Verification
        verifier = run_verifier(
            client, task, clarification, planning, catalog, mock=self.mock
        )

        # If verifier invalidates the plan, final recommendations may be empty
        final_recommendations = (
            planning.recommended_product_ids if verifier.is_valid else []
        )

        return PipelineResult(
            task_id=task.id,
            clarification=clarification,
            retrieval=retrieval,
            graph=graph,
            planning=planning,
            verifier=verifier,
            final_recommendations=final_recommendations,
            run_index=run_index,
        )
