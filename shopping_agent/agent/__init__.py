"""shopping_agent.agent — 主流程协调器与会话状态。"""

from shopping_agent.agent.state import AgentState, AttributionEntry, ConversationTurn, WorkflowStep

__all__ = [
    "ShoppingAgentOrchestrator",
    "AgentState", "AttributionEntry", "ConversationTurn", "WorkflowStep",
]


def __getattr__(name: str):
    if name == "ShoppingAgentOrchestrator":
        from shopping_agent.agent.orchestrator import ShoppingAgentOrchestrator

        return ShoppingAgentOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
