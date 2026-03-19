"""shopping_agent.common — 共享类型、异常与常量。"""

from shopping_agent.common.exceptions import (
    BudgetExceededError,
    ConstraintConflictError,
    HighRiskActionError,
    IncompatibilityError,
    InfeasibleConstraintError,
    IntentParseError,
    MemoryAccessError,
    NoProductFoundError,
    PlanningError,
    ProductNormalizationError,
    RetrievalError,
    SessionNotFoundError,
    ShopPlanError,
    StaleProductDataError,
    ToolExecutionError,
    ToolRateLimitError,
    ToolUnavailableError,
    VerificationError,
)
from shopping_agent.common.types import (
    ClarificationStyle,
    Constraint,
    ConstraintSeverity,
    ConflictPair,
    ConflictResolution,
    ExplainStyle,
    FeedbackSignal,
    ShoppingTask,
    TaskRevision,
    TaskType,
    VerificationStatus,
)

__all__ = [
    # exceptions
    "BudgetExceededError", "ConstraintConflictError", "HighRiskActionError",
    "IncompatibilityError", "InfeasibleConstraintError", "IntentParseError",
    "MemoryAccessError", "NoProductFoundError", "PlanningError",
    "ProductNormalizationError", "RetrievalError", "SessionNotFoundError",
    "ShopPlanError", "StaleProductDataError", "ToolExecutionError",
    "ToolRateLimitError", "ToolUnavailableError", "VerificationError",
    # types
    "ClarificationStyle", "Constraint", "ConstraintSeverity",
    "ConflictPair", "ConflictResolution", "ExplainStyle", "FeedbackSignal",
    "ShoppingTask", "TaskRevision", "TaskType", "VerificationStatus",
]
