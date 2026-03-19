"""
shopping_agent — 约束感知、POMDP 驱动的购物智能体。

快速上手：
    from shopping_agent import ShoppingAgentOrchestrator

    agent = ShoppingAgentOrchestrator()
    resp = agent.run(user_id="u1", user_input="帮我配一套桌搭，预算5000")
    print(resp["response"])
"""

from shopping_agent.agent.orchestrator import ShoppingAgentOrchestrator
from shopping_agent.common.types import ShoppingTask
from shopping_agent.rl.trainer import RLTrainer

__all__ = [
    "ShoppingAgentOrchestrator",
    "ShoppingTask",
    "RLTrainer",
]
