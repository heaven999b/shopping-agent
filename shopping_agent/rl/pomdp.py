"""
POMDP Formalization for Option A: RL-driven Clarification + Planning

论文级形式化定义。所有 RL 模块共享此文件的类型定义。

=== Clarification POMDP ===

  State  s  = (uncertainty_vector, slot_impact_vector, round, profile_features)
  Action a  ∈ {ASK_slot_i | i=0..K-1} ∪ {PROCEED}
  Obs    o  = user_answer (noisy observation of true preference)
  Reward R  = α·success - β·turns - γ·dropout_prob

  核心研究问题：
    在不确定需求下，何时问哪个澄清问题，使任务成功率最高、用户负担最低？

=== Planning POMDP ===

  State  s  = (budget_remaining, slot_fill_ratio, graph_coverage,
               constraint_sat_partial, preference_match_partial)
  Action a  ∈ {SELECT_product_p_for_slot_j} ∪ {REPLACE_slot_j} ∪ {FINALIZE}
  Reward R  = constraint_score + preference_score + value_score - cost_penalty

  核心研究问题：
    在候选商品图上，如何做逐步决策（每次选一个 slot），最终组合方案质量最优？
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Action 定义
# ---------------------------------------------------------------------------

class ClarificationActionType(str, Enum):
    PROCEED = "proceed"      # 不问，直接进入检索规划
    ASK_SLOT = "ask_slot"    # 询问指定槽位


@dataclass
class ClarificationAction:
    action_type: ClarificationActionType
    slot: Optional[str] = None     # 仅 ASK_SLOT 时有值
    action_id: int = 0             # 在动作空间中的索引

    def __str__(self):
        if self.action_type == ClarificationActionType.PROCEED:
            return "PROCEED"
        return f"ASK({self.slot})"


class PlanningActionType(str, Enum):
    SELECT = "select"      # 为某坑位选定一个商品
    REPLACE = "replace"    # 替换某坑位已选商品
    FINALIZE = "finalize"  # 确认方案，结束规划


@dataclass
class PlanningAction:
    action_type: PlanningActionType
    slot: Optional[str] = None
    product_id: Optional[str] = None
    action_id: int = 0


# ---------------------------------------------------------------------------
# State 表示（向量化，供 Policy 网络使用）
# ---------------------------------------------------------------------------

@dataclass
class ClarificationState:
    """
    澄清决策的状态表示。

    特征维度说明（对应 policy 的输入向量）：
      [0:K]      uncertainty_vector   各槽位不确定性 (0=确定, 1=完全未知)
      [K:2K]     impact_vector        各槽位对最终方案的影响分
      [2K]       conversation_round   当前轮次（归一化）
      [2K+1]     budget_known         预算是否已知 (0/1)
      [2K+2]     task_type_id         任务类型编码
      [2K+3:2K+3+M] profile_features  用户画像特征
    """
    slot_names: list[str]                   # 槽位名称列表（决定 K）
    uncertainty_vector: np.ndarray          # shape (K,)
    impact_vector: np.ndarray               # shape (K,)
    conversation_round: int = 0
    budget_known: bool = False
    task_type_id: int = 0
    profile_features: np.ndarray = field(
        default_factory=lambda: np.zeros(4)
    )
    # profile_features: [price_sensitivity, prefer_fast, brand_diversity, category_count]

    @property
    def num_slots(self) -> int:
        return len(self.slot_names)

    def to_vector(self) -> np.ndarray:
        """转换为 policy 网络的输入向量。"""
        return np.concatenate([
            self.uncertainty_vector,                        # K
            self.impact_vector,                             # K
            [self.conversation_round / 5.0],               # 1（归一化）
            [float(self.budget_known)],                     # 1
            [self.task_type_id / 5.0],                     # 1
            self.profile_features,                          # M
        ])

    @property
    def feature_dim(self) -> int:
        return self.num_slots * 2 + 3 + len(self.profile_features)

    def action_space(self) -> list[ClarificationAction]:
        """返回当前状态下所有合法动作。"""
        actions = [ClarificationAction(
            action_type=ClarificationActionType.PROCEED,
            action_id=0
        )]
        for i, slot in enumerate(self.slot_names):
            # 只对不确定的槽位生成 ASK 动作
            if self.uncertainty_vector[i] > 0.1:
                actions.append(ClarificationAction(
                    action_type=ClarificationActionType.ASK_SLOT,
                    slot=slot,
                    action_id=i + 1,
                ))
        return actions


@dataclass
class PlanningState:
    """
    规划决策的状态表示。

    特征维度说明：
      [0]    budget_utilization       已使用预算比例
      [1]    slot_fill_ratio          坑位填充比例
      [2]    avg_constraint_sat       当前部分方案的约束满足均值
      [3]    avg_preference_match     偏好匹配均值
      [4]    graph_incompatible_risk  当前选择组合的不兼容风险
      [5]    remaining_slots          剩余未填坑位数（归一化）
    """
    budget_total: float
    budget_used: float
    total_slots: int
    filled_slots: int
    constraint_sat_partial: float = 0.0
    preference_match_partial: float = 0.0
    incompatible_risk: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.budget_used / max(self.budget_total, 1.0),
            self.filled_slots / max(self.total_slots, 1),
            self.constraint_sat_partial,
            self.preference_match_partial,
            self.incompatible_risk,
            (self.total_slots - self.filled_slots) / max(self.total_slots, 1),
        ])

    @property
    def feature_dim(self) -> int:
        return 6

    def is_terminal(self) -> bool:
        return self.filled_slots >= self.total_slots


# ---------------------------------------------------------------------------
# Reward 定义
# ---------------------------------------------------------------------------

@dataclass
class ClarificationReward:
    """
    澄清策略的奖励组成。

    R = α·task_success - β·num_turns - γ·user_dropout
    """
    task_success: float = 0.0        # 任务成功 (0 or 1，episode 结束时赋值)
    turn_cost: float = 0.0           # 每轮对话消耗
    constraint_reduction: float = 0.0  # 本轮澄清带来的约束满足度提升（即时奖励）
    user_dropout_penalty: float = 0.0  # 用户疲劳惩罚

    # 权重（可调，作为超参数）
    alpha: float = 1.0   # 任务成功权重
    beta: float = 0.2    # 轮次成本权重
    gamma: float = 0.5   # 用户流失惩罚权重

    @property
    def total(self) -> float:
        return (self.alpha * self.task_success
                - self.beta * self.turn_cost
                + 0.1 * self.constraint_reduction
                - self.gamma * self.user_dropout_penalty)


@dataclass
class PlanningReward:
    """
    规划策略的奖励组成。

    R = constraint_score + preference_score + value_score - cost_penalty
    """
    constraint_score: float = 0.0    # 硬约束满足度
    preference_score: float = 0.0    # 软偏好匹配度
    value_score: float = 0.0         # 性价比得分
    cost_penalty: float = 0.0        # 超预算惩罚

    w_constraint: float = 0.5
    w_preference: float = 0.3
    w_value: float = 0.2

    @property
    def total(self) -> float:
        return (self.w_constraint * self.constraint_score
                + self.w_preference * self.preference_score
                + self.w_value * self.value_score
                - self.cost_penalty)


# ---------------------------------------------------------------------------
# Episode 记录（用于 REINFORCE 训练）
# ---------------------------------------------------------------------------

@dataclass
class ClarificationTransition:
    """单步澄清转移记录。"""
    state_vec: np.ndarray
    action: ClarificationAction
    log_prob: float          # log π(a|s)，由 policy 输出
    reward: float
    next_state_vec: Optional[np.ndarray]
    done: bool = False


@dataclass
class PlanningTransition:
    """单步规划转移记录。"""
    state_vec: np.ndarray
    action: PlanningAction
    log_prob: float
    reward: float
    next_state_vec: Optional[np.ndarray]
    done: bool = False


@dataclass
class Episode:
    """一次完整任务的轨迹记录。"""
    session_id: str
    user_id: str
    task_id: str

    clarification_transitions: list[ClarificationTransition] = field(default_factory=list)
    planning_transitions: list[PlanningTransition] = field(default_factory=list)

    final_task_success: bool = False
    final_plan_score: float = 0.0
    intent_resolution_score: float = 0.0
    execution_readiness_score: float = 0.0
    phase_completion_score: float = 0.0
    drift_adaptation_score: float = 0.0
    drift_detected: bool = False
    failure_bucket: str = "success"
    total_turns: int = 0

    def add_clarification_step(self, t: ClarificationTransition) -> None:
        self.clarification_transitions.append(t)
        self.total_turns += 1

    def add_planning_step(self, t: PlanningTransition) -> None:
        self.planning_transitions.append(t)

    def compute_returns(self, gamma: float = 0.99) -> tuple[list[float], list[float]]:
        """
        计算折扣累积回报（用于 REINFORCE baseline）。
        返回 (clarification_returns, planning_returns)
        """
        def _discounted_returns(rewards: list[float]) -> list[float]:
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            return returns

        clar_rewards = [t.reward for t in self.clarification_transitions]
        plan_rewards = [t.reward for t in self.planning_transitions]

        # 在最后一步追加任务成功奖励
        if clar_rewards:
            clar_rewards[-1] += (
                float(self.final_task_success)
                + 0.5 * self.intent_resolution_score
                + 0.35 * self.drift_adaptation_score
            )
        if plan_rewards:
            plan_rewards[-1] += (
                self.final_plan_score
                + 0.5 * self.execution_readiness_score
                + 0.25 * self.drift_adaptation_score
            )

        return (
            _discounted_returns(clar_rewards),
            _discounted_returns(plan_rewards),
        )

    def compute_gae_advantages(
        self,
        clar_critic,
        plan_critic,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        计算 GAE（Generalized Advantage Estimation）优势和 MC 回报。

        GAE 公式（Schulman et al. 2015）：
          δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
          A_t^GAE(γ,λ) = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}

        λ 控制偏差-方差权衡：
          λ=0 → 1-step TD advantage（低方差，高偏差）
          λ=1 → MC advantage G_t - V(s_t)（无偏，高方差）

        返回：
          (clar_advantages, plan_advantages, clar_returns, plan_returns)
          clar_returns / plan_returns 是 MC 回报，用于更新 Critic
        """
        clar_rewards = [t.reward for t in self.clarification_transitions]
        plan_rewards = [t.reward for t in self.planning_transitions]

        if clar_rewards:
            clar_rewards[-1] += (
                float(self.final_task_success)
                + 0.5 * self.intent_resolution_score
                + 0.35 * self.drift_adaptation_score
            )
        if plan_rewards:
            plan_rewards[-1] += (
                self.final_plan_score
                + 0.5 * self.execution_readiness_score
                + 0.25 * self.drift_adaptation_score
            )

        def _gae(transitions, rewards, critic):
            if not transitions:
                return [], []

            T = len(transitions)
            advantages = [0.0] * T
            returns = [0.0] * T

            # MC returns (backward pass)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + gamma * G
                returns[t] = G

            # GAE advantages (backward pass)
            gae = 0.0
            for t in reversed(range(T)):
                sv_t = transitions[t].state_vec
                V_t = critic.predict(sv_t)

                sv_next = transitions[t].next_state_vec
                done = transitions[t].done
                V_next = critic.predict(sv_next) if (sv_next is not None and not done) else 0.0

                delta = rewards[t] + gamma * V_next * (1.0 - float(done)) - V_t
                gae = delta + gamma * lam * gae * (1.0 - float(done))
                advantages[t] = gae

            # 标准化优势（降低数值不稳定性）
            adv_arr = np.array(advantages)
            if adv_arr.std() > 1e-8:
                adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
            advantages = adv_arr.tolist()

            return advantages, returns

        clar_adv, clar_ret = _gae(self.clarification_transitions,
                                   clar_rewards, clar_critic)
        plan_adv, plan_ret = _gae(self.planning_transitions,
                                   plan_rewards, plan_critic)
        return clar_adv, plan_adv, clar_ret, plan_ret
