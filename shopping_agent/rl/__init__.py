"""shopping_agent.rl — 强化学习模块。"""

from shopping_agent.rl.belief import BeliefState, SlotBelief
from shopping_agent.rl.episode_buffer import EpisodeBuffer
from shopping_agent.rl.policy import (
    LinearSoftmaxPolicy,
    LinearValueCritic,
    RLClarificationPolicy,
    RLPlanningPolicy,
)
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
from shopping_agent.rl.pretrain import BehaviorCloningTrainer
from shopping_agent.rl.trainer import RLTrainer
from shopping_agent.rl.user_simulator import (
    HiddenUserPreference,
    LLMUserSimulator,
    RuleBasedUserSimulator,
)

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
