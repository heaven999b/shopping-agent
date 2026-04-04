"""shopping_agent.rl — 强化学习模块。"""

__all__ = [
    "BeliefState", "SlotBelief",
    "EpisodeBuffer",
    "LinearSoftmaxPolicy", "LinearValueCritic",
    "RLClarificationPolicy", "RLPlanningPolicy",
    "ClarificationAction", "ClarificationActionType",
    "ClarificationReward", "ClarificationState", "ClarificationTransition",
    "Episode",
    "PlanningAction", "PlanningActionType",
    "PlanningReward", "PlanningState", "PlanningTransition",
    "BehaviorCloningTrainer",
    "RLTrainer",
    "HiddenUserPreference", "LLMUserSimulator", "RuleBasedUserSimulator",
]


def __getattr__(name: str):
    if name in {"BeliefState", "SlotBelief"}:
        from shopping_agent.rl.belief import BeliefState, SlotBelief

        return {"BeliefState": BeliefState, "SlotBelief": SlotBelief}[name]
    if name == "EpisodeBuffer":
        from shopping_agent.rl.episode_buffer import EpisodeBuffer

        return EpisodeBuffer
    if name in {
        "LinearSoftmaxPolicy",
        "LinearValueCritic",
        "RLClarificationPolicy",
        "RLPlanningPolicy",
    }:
        from shopping_agent.rl.policy import (
            LinearSoftmaxPolicy,
            LinearValueCritic,
            RLClarificationPolicy,
            RLPlanningPolicy,
        )

        return {
            "LinearSoftmaxPolicy": LinearSoftmaxPolicy,
            "LinearValueCritic": LinearValueCritic,
            "RLClarificationPolicy": RLClarificationPolicy,
            "RLPlanningPolicy": RLPlanningPolicy,
        }[name]
    if name in {
        "ClarificationAction",
        "ClarificationActionType",
        "ClarificationReward",
        "ClarificationState",
        "ClarificationTransition",
        "Episode",
        "PlanningAction",
        "PlanningActionType",
        "PlanningReward",
        "PlanningState",
        "PlanningTransition",
    }:
        from shopping_agent.rl.pomdp import (
            ClarificationAction,
            ClarificationActionType,
            ClarificationReward,
            ClarificationState,
            ClarificationTransition,
            Episode,
            PlanningAction,
            PlanningActionType,
            PlanningReward,
            PlanningState,
            PlanningTransition,
        )

        return {
            "ClarificationAction": ClarificationAction,
            "ClarificationActionType": ClarificationActionType,
            "ClarificationReward": ClarificationReward,
            "ClarificationState": ClarificationState,
            "ClarificationTransition": ClarificationTransition,
            "Episode": Episode,
            "PlanningAction": PlanningAction,
            "PlanningActionType": PlanningActionType,
            "PlanningReward": PlanningReward,
            "PlanningState": PlanningState,
            "PlanningTransition": PlanningTransition,
        }[name]
    if name == "BehaviorCloningTrainer":
        from shopping_agent.rl.pretrain import BehaviorCloningTrainer

        return BehaviorCloningTrainer
    if name == "RLTrainer":
        from shopping_agent.rl.trainer import RLTrainer

        return RLTrainer
    if name in {"HiddenUserPreference", "LLMUserSimulator", "RuleBasedUserSimulator"}:
        from shopping_agent.rl.user_simulator import (
            HiddenUserPreference,
            LLMUserSimulator,
            RuleBasedUserSimulator,
        )

        return {
            "HiddenUserPreference": HiddenUserPreference,
            "LLMUserSimulator": LLMUserSimulator,
            "RuleBasedUserSimulator": RuleBasedUserSimulator,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
