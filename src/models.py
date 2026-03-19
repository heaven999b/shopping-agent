"""Shared Pydantic data models for the Shopping Agent evaluation framework."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Catalog & Task ──────────────────────────────────────────────────────────

class Product(BaseModel):
    id: str
    name: str
    category: str
    price: float
    attributes: dict[str, Any]
    description: str
    rating: float
    in_stock: bool

    def compact_repr(self) -> str:
        """One-line representation for LLM prompts."""
        attrs = ", ".join(f"{k}={v}" for k, v in self.attributes.items()
                         if v is not None and v is not False)
        return f"[{self.id}] {self.name} | ${self.price:.2f} | {attrs}"


class Constraint(BaseModel):
    budget_max: Optional[float] = None
    budget_min: Optional[float] = None
    categories: list[str] = Field(default_factory=list)
    required_attributes: dict[str, Any] = Field(default_factory=dict)
    excluded_brands: list[str] = Field(default_factory=list)
    quantity: int = 1


class BenchmarkTask(BaseModel):
    id: str
    user_query: str
    constraints: Constraint
    needs_clarification: bool
    acceptable_product_ids: list[str]
    difficulty: str  # "easy" | "medium" | "hard"
    category: str
    description: str = ""

    @property
    def is_impossible(self) -> bool:
        return len(self.acceptable_product_ids) == 0


# ── Pipeline Stage Outputs ──────────────────────────────────────────────────

class ClarificationOutput(BaseModel):
    needs_clarification: bool
    clarifying_questions: list[str] = Field(default_factory=list)
    extracted_constraints: dict[str, Any] = Field(default_factory=dict)
    refined_query: str
    reasoning: str = ""


class RetrievalOutput(BaseModel):
    retrieved_product_ids: list[str]
    relevance_scores: dict[str, float] = Field(default_factory=dict)
    retrieval_reasoning: str = ""


class GraphNode(BaseModel):
    product_id: str
    constraint_compatibility: float  # 0.0–1.0
    rank: int


class GraphOutput(BaseModel):
    ranked_nodes: list[GraphNode]
    constraint_coverage: dict[str, bool] = Field(default_factory=dict)
    graph_reasoning: str = ""


class PlanningOutput(BaseModel):
    recommended_product_ids: list[str]
    plan_reasoning: str = ""
    constraint_satisfaction: dict[str, bool] = Field(default_factory=dict)
    confidence: float = 0.0
    total_price: float = 0.0


class VerifierOutput(BaseModel):
    is_valid: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    verification_reasoning: str = ""


class PipelineResult(BaseModel):
    task_id: str
    clarification: ClarificationOutput
    retrieval: RetrievalOutput
    graph: GraphOutput
    planning: PlanningOutput
    verifier: VerifierOutput
    final_recommendations: list[str]
    run_index: int = 0  # for pass^k


# ── Evaluation Outputs ──────────────────────────────────────────────────────

class StageMetrics(BaseModel):
    clarification_accuracy: float = 0.0   # 0 or 1: correct decision
    constraint_extraction_f1: float = 0.0  # F1 vs ground-truth constraints
    retrieval_recall: float = 0.0           # |retrieved ∩ acceptable| / |acceptable|
    retrieval_precision: float = 0.0        # |retrieved ∩ acceptable| / |retrieved|
    graph_coverage: float = 0.0             # fraction of constraints represented in graph
    planning_precision: float = 0.0         # |recommended ∩ acceptable| / |recommended|
    planning_constraint_sat: float = 0.0    # fraction of constraints satisfied in plan
    verifier_detection_rate: float = 0.0    # when plan invalid, fraction of violations caught


class UScore(BaseModel):
    """Understanding score: how well the agent understood the user's need."""
    clarification_decision: float = 0.0   # correct clarification call (0 or 1)
    constraint_extraction: float = 0.0    # constraint extraction quality
    total: float = 0.0

    @classmethod
    def compute(cls, clarification_decision: float, constraint_extraction: float) -> "UScore":
        total = 0.5 * clarification_decision + 0.5 * constraint_extraction
        return cls(
            clarification_decision=clarification_decision,
            constraint_extraction=constraint_extraction,
            total=round(total, 4),
        )


class EScore(BaseModel):
    """Execution score: how well the agent found and planned the right products."""
    retrieval_quality: float = 0.0    # weighted recall+precision
    planning_quality: float = 0.0     # precision × constraint sat
    verifier_quality: float = 0.0     # detection rate
    total: float = 0.0

    @classmethod
    def compute(
        cls,
        retrieval_recall: float,
        retrieval_precision: float,
        planning_precision: float,
        planning_constraint_sat: float,
        verifier_detection_rate: float,
    ) -> "EScore":
        retrieval_quality = 0.5 * retrieval_recall + 0.5 * retrieval_precision
        planning_quality = 0.5 * planning_precision + 0.5 * planning_constraint_sat
        verifier_quality = verifier_detection_rate
        total = 0.35 * retrieval_quality + 0.45 * planning_quality + 0.20 * verifier_quality
        return cls(
            retrieval_quality=round(retrieval_quality, 4),
            planning_quality=round(planning_quality, 4),
            verifier_quality=round(verifier_quality, 4),
            total=round(total, 4),
        )


class TaskEvaluation(BaseModel):
    task_id: str
    difficulty: str
    category: str
    u_score: UScore
    e_score: EScore
    stage_metrics: StageMetrics
    error_types: list[str] = Field(default_factory=list)
    success: bool = False
    quadrant: str = "LL"  # "HH" | "HL" | "LH" | "LL"
    run_index: int = 0

    def determine_quadrant(self) -> str:
        threshold = 0.5
        u_high = self.u_score.total >= threshold
        e_high = self.e_score.total >= threshold
        if u_high and e_high:
            return "HH"
        elif u_high and not e_high:
            return "HL"
        elif not u_high and e_high:
            return "LH"
        else:
            return "LL"


class StabilityResult(BaseModel):
    """pass^k stability analysis for a single task."""
    task_id: str
    k: int
    success_rate: float
    u_score_mean: float
    u_score_variance: float
    e_score_mean: float
    e_score_variance: float
    violation_counts: list[int] = Field(default_factory=list)
    violation_rate_variance: float = 0.0
    pass_at_k: float = 0.0  # P(at least one success in k runs)


class EvaluationSummary(BaseModel):
    """Aggregate evaluation results across all tasks."""
    n_tasks: int
    n_runs_per_task: int
    avg_u_score: float
    avg_e_score: float
    quadrant_distribution: dict[str, int]
    error_type_counts: dict[str, int]
    difficulty_breakdown: dict[str, dict[str, float]]
    avg_stage_metrics: StageMetrics
    stability_results: list[StabilityResult] = Field(default_factory=list)
