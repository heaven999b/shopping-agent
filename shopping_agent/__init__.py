"""
shopping_agent — 约束感知、POMDP 驱动的购物智能体。

快速上手：
    from shopping_agent import ShoppingAgentOrchestrator

    agent = ShoppingAgentOrchestrator()
    resp = agent.run(user_id="u1", user_input="帮我配一套桌搭，预算5000")
    print(resp["response"])
"""

from shopping_agent.common.types import ShoppingTask

__all__ = ["ShoppingAgentOrchestrator", "ShoppingTask", "RLTrainer"]


def __getattr__(name: str):
    if name == "ShoppingAgentOrchestrator":
        from shopping_agent.agent.orchestrator import ShoppingAgentOrchestrator

        return ShoppingAgentOrchestrator
    if name == "RLTrainer":
        from shopping_agent.rl.trainer import RLTrainer

        return RLTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
