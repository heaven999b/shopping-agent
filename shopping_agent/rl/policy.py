"""
Policy Networks for Clarification and Planning.

实现两个独立的 softmax 策略（线性参数化）：
  - ClarificationPolicy: π_θ(a | s_clar)
  - PlanningPolicy:      π_φ(a | s_plan)

算法：REINFORCE with baseline（Policy Gradient）

选择线性策略的理由：
  1. 参数量小，数据效率高（购物任务数据有限）
  2. 可解释性强（权重直接对应特征重要性）
  3. 容易在论文中分析（不需要 GPU）
  4. 可无缝替换为神经网络版本（接口不变）

论文中可以写：
  π_θ(a|s) = softmax(W_θ · φ(s))
  其中 φ(s) 为状态特征向量，W_θ 为可学习参数矩阵
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from shopping_agent.rl.pomdp import (
    ClarificationAction,
    ClarificationActionType,
    ClarificationState,
    PlanningAction,
    PlanningActionType,
    PlanningState,
)


class LinearSoftmaxPolicy:
    """
    线性 Softmax 策略基类。

    π(a|s) = softmax(W @ φ(s))

    参数：
      W: shape (num_actions, feature_dim)  — 策略参数矩阵
      b: shape (num_actions,)              — 偏置项
    """

    def __init__(self, feature_dim: int, num_actions: int,
                 learning_rate: float = 1e-3):
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.lr = learning_rate

        # Xavier 初始化
        scale = np.sqrt(2.0 / (feature_dim + num_actions))
        self.W = np.random.randn(num_actions, feature_dim) * scale
        self.b = np.zeros(num_actions)

        # 基线（baseline）：用于降低方差，用移动平均估计 V(s)
        self._baseline = 0.0
        self._baseline_alpha = 0.1  # 指数移动平均系数

    def logits(self, state_vec: np.ndarray) -> np.ndarray:
        """计算动作 logits。"""
        return self.W @ state_vec + self.b

    def probs(self, state_vec: np.ndarray) -> np.ndarray:
        """计算动作概率分布（数值稳定的 softmax）。"""
        l = self.logits(state_vec)
        l = l - np.max(l)  # 数值稳定
        exp_l = np.exp(l)
        return exp_l / exp_l.sum()

    def sample(self, state_vec: np.ndarray,
               valid_action_ids: Optional[list[int]] = None) -> tuple[int, float]:
        """
        按策略采样动作。

        返回 (action_id, log_prob)
        valid_action_ids: 合法动作的 id 列表（None 表示全部合法）
        """
        p = self.probs(state_vec)

        if valid_action_ids is not None:
            mask = np.zeros(self.num_actions)
            for aid in valid_action_ids:
                if aid < self.num_actions:
                    mask[aid] = 1.0
            p = p * mask
            if p.sum() < 1e-10:
                p = mask / mask.sum()  # 均匀分布兜底
            else:
                p = p / p.sum()

        action_id = int(np.random.choice(self.num_actions, p=p))
        log_prob = float(np.log(p[action_id] + 1e-10))
        return action_id, log_prob

    def greedy(self, state_vec: np.ndarray,
               valid_action_ids: Optional[list[int]] = None) -> int:
        """贪心选择（推理时使用）。"""
        p = self.probs(state_vec)
        if valid_action_ids is not None:
            mask = np.zeros(self.num_actions)
            for aid in valid_action_ids:
                if aid < self.num_actions:
                    mask[aid] = 1.0
            p = p * mask
        return int(np.argmax(p))

    def update(self, state_vec: np.ndarray, action_id: int,
               G: float) -> float:
        """
        REINFORCE 梯度更新。

        ∇_θ J = (G - baseline) · ∇_θ log π(a|s)
        W += lr · (G - baseline) · ∇_θ log π(a|s)

        返回本步的 loss（用于日志）
        """
        p = self.probs(state_vec)
        advantage = G - self._baseline

        # 更新基线（移动平均）
        self._baseline = (
            (1 - self._baseline_alpha) * self._baseline
            + self._baseline_alpha * G
        )

        # 计算梯度：∂log π(a|s) / ∂W
        # = (e_a - p) ⊗ φ(s)，其中 e_a 为 one-hot
        one_hot = np.zeros(self.num_actions)
        one_hot[action_id] = 1.0
        grad_logits = one_hot - p                    # shape (num_actions,)
        grad_W = np.outer(grad_logits, state_vec)    # shape (num_actions, feature_dim)
        grad_b = grad_logits                          # shape (num_actions,)

        self.W += self.lr * advantage * grad_W
        self.b += self.lr * advantage * grad_b

        loss = -np.log(p[action_id] + 1e-10) * advantage
        return float(loss)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"W": self.W, "b": self.b, "baseline": self._baseline}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.W = data["W"]
        self.b = data["b"]
        self._baseline = data.get("baseline", 0.0)


# ---------------------------------------------------------------------------
# Clarification Policy
# ---------------------------------------------------------------------------

# 所有可能的槽位（固定动作空间上限）
ALL_SLOTS = [
    "budget_total", "delivery_days", "usage_scenario",
    "brand_preference", "color", "size",
    "platform", "style", "compatibility",
]
MAX_SLOTS = len(ALL_SLOTS)
# 动作空间：0=PROCEED, 1..K=ASK_slot_i
CLAR_NUM_ACTIONS = MAX_SLOTS + 1


class RLClarificationPolicy:
    """
    强化学习澄清策略。

    替代原来的启发式 info_gain 方法，改为学习得到的策略。

    动作空间（大小固定为 MAX_SLOTS + 1）：
      0         → PROCEED（不问，直接规划）
      1..K      → ASK_SLOT_i（询问第 i 个槽位）
    """

    def __init__(self, profile_feature_dim: int = 4,
                 learning_rate: float = 1e-3):
        # 状态特征维度：K*2 + 3 + M
        feature_dim = MAX_SLOTS * 2 + 3 + profile_feature_dim
        self._policy = LinearSoftmaxPolicy(
            feature_dim=feature_dim,
            num_actions=CLAR_NUM_ACTIONS,
            learning_rate=learning_rate,
        )
        self._training = True

    def decide(
        self,
        state: ClarificationState,
        explore: bool = True,
    ) -> tuple[ClarificationAction, float]:
        """
        根据当前澄清状态决定动作。

        返回 (action, log_prob)
        explore=True  → 采样（训练时）
        explore=False → 贪心（推理时）
        """
        state_vec = self._pad_state_vector(state)
        valid_ids = self._valid_action_ids(state)

        if explore and self._training:
            action_id, log_prob = self._policy.sample(state_vec, valid_ids)
        else:
            action_id = self._policy.greedy(state_vec, valid_ids)
            p = self._policy.probs(state_vec)
            log_prob = float(np.log(p[action_id] + 1e-10))

        action = self._id_to_action(action_id, state)
        return action, log_prob

    def update(self, state: ClarificationState, action: ClarificationAction,
               G: float) -> float:
        """REINFORCE 更新，返回 loss。"""
        state_vec = self._pad_state_vector(state)
        return self._policy.update(state_vec, action.action_id, G)

    def set_training(self, training: bool) -> None:
        self._training = training

    def save(self, path: str) -> None:
        self._policy.save(path)

    def load(self, path: str) -> None:
        if Path(path).exists():
            self._policy.load(path)

    def _pad_state_vector(self, state: ClarificationState) -> np.ndarray:
        """
        将 ClarificationState 的变长向量 pad 到固定维度。
        （因为不同任务的槽位数不同，但策略网络需要固定输入维度）
        """
        K = state.num_slots
        unc = np.zeros(MAX_SLOTS)
        imp = np.zeros(MAX_SLOTS)
        unc[:K] = state.uncertainty_vector[:MAX_SLOTS]
        imp[:K] = state.impact_vector[:MAX_SLOTS]

        return np.concatenate([
            unc,
            imp,
            [state.conversation_round / 5.0],
            [float(state.budget_known)],
            [state.task_type_id / 5.0],
            state.profile_features,
        ])

    def _valid_action_ids(self, state: ClarificationState) -> list[int]:
        """返回当前状态下合法动作的 id 列表。"""
        valid = [0]  # PROCEED 始终合法
        for i, slot in enumerate(ALL_SLOTS):
            if i < state.num_slots and state.uncertainty_vector[i] > 0.1:
                valid.append(i + 1)
        return valid

    def _id_to_action(self, action_id: int,
                      state: ClarificationState) -> ClarificationAction:
        if action_id == 0:
            return ClarificationAction(
                action_type=ClarificationActionType.PROCEED,
                action_id=0,
            )
        slot_idx = action_id - 1
        slot_name = ALL_SLOTS[slot_idx] if slot_idx < len(ALL_SLOTS) else "unknown"
        return ClarificationAction(
            action_type=ClarificationActionType.ASK_SLOT,
            slot=slot_name,
            action_id=action_id,
        )


# ---------------------------------------------------------------------------
# Planning Policy
# ---------------------------------------------------------------------------

class RLPlanningPolicy:
    """
    强化学习规划策略。

    在候选商品图上逐步为每个坑位选择商品。

    动作空间（简化为 3 类高层动作）：
      0 → SELECT_BEST_MATCH    选评分+偏好最高的候选
      1 → SELECT_BUDGET_OPT   选性价比最高的候选
      2 → SELECT_SAFE          选最保守（评分高、库存充足）的候选

    论文可以扩展为：每个具体商品 ID 作为一个动作
    （但 action space 会很大，需要 pointer network 或 graph attention）
    """

    PLAN_NUM_ACTIONS = 3

    def __init__(self, learning_rate: float = 1e-3):
        self._policy = LinearSoftmaxPolicy(
            feature_dim=PlanningState.__dataclass_fields__.__len__()
            if hasattr(PlanningState, '__dataclass_fields__') else 6,
            num_actions=self.PLAN_NUM_ACTIONS,
            learning_rate=learning_rate,
        )
        self._training = True

    def decide(
        self,
        state: PlanningState,
        explore: bool = True,
    ) -> tuple[PlanningAction, float]:
        state_vec = state.to_vector()

        if explore and self._training:
            action_id, log_prob = self._policy.sample(state_vec)
        else:
            action_id = self._policy.greedy(state_vec)
            p = self._policy.probs(state_vec)
            log_prob = float(np.log(p[action_id] + 1e-10))

        action = PlanningAction(
            action_type=PlanningActionType.SELECT,
            action_id=action_id,
        )
        return action, log_prob

    def update(self, state: PlanningState, action: PlanningAction,
               G: float) -> float:
        state_vec = state.to_vector()
        return self._policy.update(state_vec, action.action_id, G)

    def set_training(self, training: bool) -> None:
        self._training = training

    def save(self, path: str) -> None:
        self._policy.save(path)

    def load(self, path: str) -> None:
        if Path(path).exists():
            self._policy.load(path)
