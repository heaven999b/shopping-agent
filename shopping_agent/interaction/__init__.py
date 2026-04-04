"""shopping_agent.interaction — 意图解析与澄清策略。"""

__all__ = ["ClarificationPolicy", "IntentParser"]


def __getattr__(name: str):
    if name == "ClarificationPolicy":
        from shopping_agent.interaction.clarification import ClarificationPolicy

        return ClarificationPolicy
    if name == "IntentParser":
        from shopping_agent.interaction.intent_parser import IntentParser

        return IntentParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
