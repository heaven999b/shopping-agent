"""
Error Taxonomy – classify pipeline failures into 5 canonical error types.

Error types (from tSCOPE-inspired framework):
    1. CLARIFICATION_ERROR  – wrong clarification decision
    2. RETRIEVAL_ERROR       – acceptable products not retrieved
    3. PLANNING_ERROR        – retrieval succeeded but plan failed
    4. CONSTRAINT_VIOLATION  – plan violates hard constraints
    5. VERIFIER_ERROR        – verifier missed violations or false-alarmed
"""
from __future__ import annotations

from enum import Enum

from ..models import BenchmarkTask, PipelineResult, TaskEvaluation


class ErrorType(str, Enum):
    CLARIFICATION_ERROR = "clarification_error"
    RETRIEVAL_ERROR = "retrieval_error"
    PLANNING_ERROR = "planning_error"
    CONSTRAINT_VIOLATION = "constraint_violation"
    VERIFIER_ERROR = "verifier_error"


# Thresholds
_THRESHOLD_RECALL = 0.5       # retrieval is "bad" if recall < this
_THRESHOLD_PRECISION = 0.4    # planning is "bad" if precision < this


def classify_errors(
    task: BenchmarkTask,
    result: PipelineResult,
    evaluation: TaskEvaluation,
) -> list[str]:
    """
    Classify all errors present in a pipeline result.

    Returns a list of ErrorType string values (may be empty if no errors).
    """
    errors: list[str] = []
    sm = evaluation.stage_metrics

    # 1. Clarification error
    if sm.clarification_accuracy < 1.0:
        errors.append(ErrorType.CLARIFICATION_ERROR.value)

    # 2. Retrieval error – low recall on a non-impossible task
    acceptable = set(task.acceptable_product_ids)
    if acceptable and sm.retrieval_recall < _THRESHOLD_RECALL:
        errors.append(ErrorType.RETRIEVAL_ERROR.value)

    # 3. Planning error – retrieval was ok but planning failed
    if (
        acceptable
        and sm.retrieval_recall >= _THRESHOLD_RECALL
        and sm.planning_precision < _THRESHOLD_PRECISION
    ):
        errors.append(ErrorType.PLANNING_ERROR.value)

    # 4. Constraint violation – verifier found hard violations
    if result.verifier.violations:
        errors.append(ErrorType.CONSTRAINT_VIOLATION.value)

    # 5. Verifier error – verifier's detection rate is poor
    if sm.verifier_detection_rate < 0.5:
        errors.append(ErrorType.VERIFIER_ERROR.value)

    return errors


def error_type_description(error_type: str) -> str:
    descriptions = {
        ErrorType.CLARIFICATION_ERROR.value: (
            "Agent made wrong clarification decision "
            "(asked when it shouldn't, or didn't ask when needed)"
        ),
        ErrorType.RETRIEVAL_ERROR.value: (
            "Retrieval stage failed to surface acceptable products "
            "(low recall – missing relevant items)"
        ),
        ErrorType.PLANNING_ERROR.value: (
            "Retrieval found good candidates but planner chose wrong products "
            "(low planning precision despite good retrieval)"
        ),
        ErrorType.CONSTRAINT_VIOLATION.value: (
            "Final plan violates at least one hard constraint "
            "(budget exceeded, required attribute missing, etc.)"
        ),
        ErrorType.VERIFIER_ERROR.value: (
            "Verifier failed to correctly detect plan validity "
            "(missed violations or false-alarmed)"
        ),
    }
    return descriptions.get(error_type, "Unknown error type")
