"""
Bayesian Belief State — POMDP 信念状态模块。

在 POMDP 框架中，agent 无法直接观测用户的真实偏好，
只能通过对话观测到带噪声的用户回答，并维护一个后验分布。

数学定义：
  b_t(p) = P(user_preference = p | o_1, a_1, ..., o_t)

  Bayesian 更新：
    b_{t+1}(p') ∝ P(o_{t+1} | p', a_t) · Σ_p P(p' | p, a_t) · b_t(p)

  在购物场景的简化假设下（用户偏好稳定，不随时间变化）：
    P(p' | p, a_t) = 1 if p'=p else 0  →  仅更新观测似然

    b_{t+1}(p) ∝ P(o | preference_value = p) · b_t(p)

Value of Information（VoI / 信息增益）：
  VoI(slot) = H(b_before) - E[H(b_after | ask slot)]
            ≈ H(b_before) - Σ_o P(o) · H(b_after(o))

  用于澄清策略：优先询问 VoI 最高的槽位。

论文定位：
  这是 POMDP 澄清策略的核心理论贡献。
  相比启发式方法（"哪个槽位不确定性高就问哪个"），
  VoI-based belief 能更精确地估计问题价值，
  在有限对话轮次内获得最多信息。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 槽位先验知识库（可扩展）
# ---------------------------------------------------------------------------

# 各槽位的候选取值域 + 先验概率（均匀先验时等权）
SLOT_PRIOR: dict[str, list[Any]] = {
    "budget_total": [500.0, 1000.0, 2000.0, 3000.0, 5000.0, 8000.0, 12000.0, 20000.0],
    "delivery_days": [1, 2, 3, 5, 7],
    "usage_scenario": ["home", "office", "travel", "outdoor", "gaming"],
    "brand_preference": ["Sony", "Samsung", "LG", "Apple", "Huawei",
                         "Lenovo", "Dell", "HP", "Asus", "Logitech"],
    "color": ["black", "white", "silver", "red", "blue"],
    "size": ["small", "medium", "large"],
    "platform": ["JD", "Tmall", "PDD"],
    "style": ["简约", "商务", "潮流", "极简"],
    "compatibility": ["Windows", "Mac", "both"],
}

# 似然噪声水平（用户回答的可信度）
_DEFAULT_NOISE = 0.1   # 10% 的概率用户回答不准确


# ---------------------------------------------------------------------------
# 单槽位信念
# ---------------------------------------------------------------------------

@dataclass
class SlotBelief:
    """
    单个槽位的 Bayesian 信念状态。

    维护一个离散概率分布 P(true_value = v) 对所有候选值 v。

    属性：
        slot        — 槽位名称
        values      — 候选取值列表（离散化）
        probs       — 对应概率分布（和为1）
        is_observed — 是否已直接观测到确定值
        observed_value — 若已观测，记录该值
    """
    slot: str
    values: list[Any]
    probs: np.ndarray       # shape: (len(values),)
    is_observed: bool = False
    observed_value: Any = None

    @classmethod
    def uniform(cls, slot: str, values: Optional[list[Any]] = None) -> "SlotBelief":
        """用均匀先验初始化槽位信念。"""
        if values is None:
            values = SLOT_PRIOR.get(slot, [None])
        n = len(values)
        return cls(
            slot=slot,
            values=values,
            probs=np.ones(n) / n,
        )

    def bayesian_update(self, observation: Any, noise: float = _DEFAULT_NOISE) -> None:
        """
        观测到 observation 后做 Bayesian 更新。

        似然模型（简单高斯/分类）：
          若 v == observation：P(obs | v) = 1 - noise
          否则：P(obs | v) = noise / (n-1)，均匀分配到其他值

        参数：
            observation — 用户回答（字符串/数字）
            noise       — 观测噪声水平（0=完全可信, 1=完全无用）
        """
        n = len(self.values)
        likelihood = np.full(n, noise / max(n - 1, 1))

        matched = False
        for i, v in enumerate(self.values):
            if self._matches(v, observation):
                likelihood[i] = 1.0 - noise
                matched = True
                break

        if not matched:
            # 无匹配候选：软更新，所有候选轻微降权
            likelihood = np.ones(n)

        # P(v | obs) ∝ P(obs | v) * P(v)
        posterior = likelihood * self.probs
        total = posterior.sum()
        if total > 1e-10:
            self.probs = posterior / total
        # else: 保持原分布不变（避免数值崩溃）

    def mark_observed(self, value: Any) -> None:
        """明确标记某槽位已知（delta 分布）。"""
        self.is_observed = True
        self.observed_value = value
        n = len(self.values)
        self.probs = np.zeros(n)
        # 找到最近的候选值
        for i, v in enumerate(self.values):
            if self._matches(v, value):
                self.probs[i] = 1.0
                return
        # 没有精确匹配：均匀
        self.probs = np.ones(n) / n

    def entropy(self) -> float:
        """
        香农熵 H(b) = -Σ p_i log p_i（以 nat 为单位）。

        熵越高 → 不确定性越大 → 越值得澄清。
        """
        p = self.probs[self.probs > 1e-12]
        return float(-np.sum(p * np.log(p)))

    def max_entropy(self) -> float:
        """均匀分布时的最大熵 log(n)。"""
        return math.log(len(self.values))

    def uncertainty(self) -> float:
        """归一化不确定性 ∈ [0,1]，= entropy / max_entropy。"""
        me = self.max_entropy()
        return self.entropy() / me if me > 0 else 0.0

    def mode(self) -> Any:
        """MAP 估计：最高概率的候选值。"""
        return self.values[int(np.argmax(self.probs))]

    def mode_prob(self) -> float:
        """MAP 估计的概率。"""
        return float(np.max(self.probs))

    def expected_value(self) -> Optional[float]:
        """若候选值为数值型，返回期望值。"""
        try:
            vals = np.array(self.values, dtype=float)
            return float(np.dot(vals, self.probs))
        except (TypeError, ValueError):
            return None

    def expected_info_gain(self, noise: float = _DEFAULT_NOISE) -> float:
        """
        期望信息增益（VoI）：询问此槽位期望能降低多少熵。

        VoI(slot) ≈ H(b_t) - Σ_o P(o) · H(b_{t+1}(o))

        计算方式：枚举所有可能的观测 o（= 候选值之一），
        加权平均更新后的熵。
        """
        current_entropy = self.entropy()
        if current_entropy < 1e-8:
            return 0.0  # 已完全确定，无信息增益

        expected_posterior_entropy = 0.0
        for obs_value in self.values:
            # P(o = obs_value) ≈ Σ_v P(obs=v | true=v) * P(true=v)
            p_obs = 0.0
            for i, true_v in enumerate(self.values):
                if self._matches(true_v, obs_value):
                    p_obs += (1.0 - noise) * self.probs[i]
                else:
                    p_obs += (noise / max(len(self.values) - 1, 1)) * self.probs[i]

            if p_obs < 1e-12:
                continue

            # 计算观测后的后验熵
            posterior_belief = SlotBelief(
                slot=self.slot,
                values=self.values,
                probs=self.probs.copy(),
            )
            posterior_belief.bayesian_update(obs_value, noise)
            expected_posterior_entropy += p_obs * posterior_belief.entropy()

        return max(0.0, current_entropy - expected_posterior_entropy)

    @staticmethod
    def _matches(candidate: Any, observation: Any) -> bool:
        """判断候选值是否与观测匹配（数值型做近似匹配）。"""
        if candidate == observation:
            return True
        # 字符串大小写不敏感
        if isinstance(candidate, str) and isinstance(observation, str):
            return candidate.lower() == observation.lower()
        # 数值型：在 20% 以内算匹配
        try:
            c, o = float(candidate), float(observation)
            return abs(c - o) / (abs(c) + 1e-8) < 0.2
        except (TypeError, ValueError):
            return False


# ---------------------------------------------------------------------------
# 全局信念状态
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """
    用户偏好的全局 Bayesian 信念状态。

    维护所有槽位的 SlotBelief，支持：
    - 单槽位或全局 Bayesian 更新
    - 最优澄清问题选择（VoI argmax）
    - 信念转向量（供策略网络使用）
    """
    slot_beliefs: dict[str, SlotBelief] = field(default_factory=dict)

    @classmethod
    def create(cls, slots: Optional[list[str]] = None) -> "BeliefState":
        """
        工厂方法：用均匀先验初始化所有槽位。

        参数：
            slots — 需要跟踪的槽位列表，默认使用 SLOT_PRIOR 的全部槽位
        """
        if slots is None:
            slots = list(SLOT_PRIOR.keys())
        beliefs = {slot: SlotBelief.uniform(slot) for slot in slots}
        return cls(slot_beliefs=beliefs)

    def update(self, slot: str, observation: Any,
               noise: float = _DEFAULT_NOISE) -> None:
        """观测到某槽位的用户回答后更新信念。"""
        if slot not in self.slot_beliefs:
            # 动态添加槽位（使用均匀先验）
            self.slot_beliefs[slot] = SlotBelief.uniform(slot)
        self.slot_beliefs[slot].bayesian_update(observation, noise)

    def mark_known(self, slot: str, value: Any) -> None:
        """明确标记槽位已知（如用户主动说明预算）。"""
        if slot not in self.slot_beliefs:
            self.slot_beliefs[slot] = SlotBelief.uniform(slot)
        self.slot_beliefs[slot].mark_observed(value)

    def best_slot_to_ask(
        self,
        unknown_slots: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        返回 VoI 最高的槽位名，即最值得澄清的问题。

        参数：
            unknown_slots — 仅在这些槽位中选择（None 表示所有槽位）
        """
        candidates = unknown_slots or list(self.slot_beliefs.keys())
        candidates = [s for s in candidates
                      if s in self.slot_beliefs
                      and not self.slot_beliefs[s].is_observed]

        if not candidates:
            return None

        voi_scores = {
            s: self.slot_beliefs[s].expected_info_gain()
            for s in candidates
        }
        return max(voi_scores, key=voi_scores.__getitem__)

    def total_entropy(self) -> float:
        """所有未知槽位的熵之和（全局不确定性度量）。"""
        return sum(
            b.entropy()
            for b in self.slot_beliefs.values()
            if not b.is_observed
        )

    def to_vector(self) -> np.ndarray:
        """
        将信念状态转为定长特征向量，供策略网络使用。

        每个槽位编码为 3 个特征：
          [uncertainty, mode_prob, expected_info_gain]
        """
        features = []
        for slot, belief in self.slot_beliefs.items():
            features.extend([
                belief.uncertainty(),          # 不确定性 ∈ [0,1]
                belief.mode_prob(),            # MAP 置信度
                belief.expected_info_gain(),   # VoI
            ])
        return np.array(features, dtype=np.float32)

    def summary(self) -> dict[str, dict]:
        """
        人类可读的信念状态摘要，用于日志和调试。
        """
        result = {}
        for slot, belief in self.slot_beliefs.items():
            result[slot] = {
                "mode": belief.mode(),
                "mode_prob": round(belief.mode_prob(), 3),
                "uncertainty": round(belief.uncertainty(), 3),
                "voi": round(belief.expected_info_gain(), 4),
                "is_known": belief.is_observed,
            }
        return result

    def get_map_estimates(self) -> dict[str, Any]:
        """返回所有槽位的 MAP 估计（最高概率值）。"""
        return {slot: b.mode() for slot, b in self.slot_beliefs.items()}

    def get_expected_values(self) -> dict[str, Optional[float]]:
        """返回数值型槽位的期望值（用于约束推断）。"""
        return {slot: b.expected_value() for slot, b in self.slot_beliefs.items()}
