"""
RLTrainer — REINFORCE 训练器。

训练两个策略：
  - ClarificationPolicy: π_θ
  - PlanningPolicy:      π_φ

训练流程（每次迭代）：
  1. 用 UserSimulator 采集 N 个 episode
  2. 计算每个 episode 的折扣回报 G_t
  3. 对所有 (s_t, a_t, G_t) 做 REINFORCE 梯度更新
  4. 记录指标，保存检查点

论文实验设计（建议的 Ablation）：
  - Full model:          RL Clarification + RL Planning
  - Ablation A:          Heuristic Clarification + RL Planning
  - Ablation B:          RL Clarification + Heuristic Planning
  - Baseline:            Heuristic Clarification + Heuristic Planning
  - Upper bound:         Oracle（已知用户偏好，不需要澄清）
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from shopping_agent.common.types import (
    ClarificationQuestion,
    ClarificationStyle,
    ShoppingTask,
    TaskType,
    UserProfile,
)
from shopping_agent.rl.episode_buffer import EpisodeBuffer
from shopping_agent.rl.pomdp import (
    ClarificationReward,
    ClarificationState,
    ClarificationTransition,
    Episode,
    PlanningReward,
    PlanningState,
    PlanningTransition,
)
from shopping_agent.rl.policy import (
    ALL_SLOTS,
    RLClarificationPolicy,
    RLPlanningPolicy,
)
from shopping_agent.rl.user_simulator import (
    HiddenUserPreference,
    RuleBasedUserSimulator,
)

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    REINFORCE 训练器。

    示例用法：
        trainer = RLTrainer(checkpoint_dir="./checkpoints")
        trainer.train(num_iterations=100, episodes_per_iter=32)
        stats = trainer.evaluate(num_episodes=50)
    """

    def __init__(
        self,
        clar_lr: float = 5e-4,
        plan_lr: float = 5e-4,
        gamma: float = 0.99,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.clar_policy = RLClarificationPolicy(learning_rate=clar_lr)
        self.plan_policy = RLPlanningPolicy(learning_rate=plan_lr)
        self.simulator = RuleBasedUserSimulator(noise_level=0.1)
        self.buffer = EpisodeBuffer(capacity=512)
        self.gamma = gamma
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._iteration = 0
        self._training_log: list[dict] = []

    # ---------------------------------------------------------------------------
    # 主训练入口
    # ---------------------------------------------------------------------------

    def train(
        self,
        num_iterations: int = 200,
        episodes_per_iter: int = 32,
        eval_interval: int = 20,
        save_interval: int = 50,
    ) -> list[dict]:
        """
        运行 REINFORCE 训练循环。

        返回每次迭代的训练指标列表。
        """
        logger.info(f"开始 RL 训练：{num_iterations} 次迭代，每次 {episodes_per_iter} 个 episode")

        self.clar_policy.set_training(True)
        self.plan_policy.set_training(True)

        for iteration in range(num_iterations):
            self._iteration = iteration
            t0 = time.time()

            # 1. 采集 episodes
            episodes = self._collect_episodes(episodes_per_iter)
            for ep in episodes:
                self.buffer.add(ep)

            # 2. 批量更新
            metrics = self._update_policies(episodes)

            # 3. 清空 buffer（on-policy）
            self.buffer.clear()

            metrics["iteration"] = iteration
            metrics["time_s"] = round(time.time() - t0, 2)
            self._training_log.append(metrics)

            if (iteration + 1) % eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=20)
                logger.info(
                    f"[Iter {iteration+1}] "
                    f"success={eval_stats['success_rate']:.2%} "
                    f"turns={eval_stats['avg_turns']:.1f} "
                    f"plan_score={eval_stats['avg_plan_score']:.3f}"
                )

            if (iteration + 1) % save_interval == 0:
                self.save_checkpoint(f"iter_{iteration+1}")

        self.clar_policy.set_training(False)
        self.plan_policy.set_training(False)
        return self._training_log

    # ---------------------------------------------------------------------------
    # Episode 采集
    # ---------------------------------------------------------------------------

    def _collect_episodes(self, n: int) -> list[Episode]:
        episodes = []
        for _ in range(n):
            user_pref = HiddenUserPreference.sample_random()
            task = self._sample_task(user_pref)
            ep = self._run_episode(task, user_pref, explore=True)
            episodes.append(ep)
        return episodes

    def _run_episode(
        self,
        task: ShoppingTask,
        user_pref: HiddenUserPreference,
        explore: bool = True,
    ) -> Episode:
        """
        运行一个完整的购物 episode。

        Phase 1: Clarification（澄清阶段）
        Phase 2: Planning（规划阶段）
        """
        ep = Episode(
            session_id=str(uuid.uuid4()),
            user_id="sim_user",
            task_id=task.task_id,
        )

        # ── Phase 1: Clarification ──
        from shopping_agent.common.constants import CLARIFICATION_MAX_ROUNDS
        user_dropped_out = False

        for _round in range(CLARIFICATION_MAX_ROUNDS + 1):
            clar_state = self._build_clar_state(task, _round)

            # 策略决策
            action, log_prob = self.clar_policy.decide(clar_state, explore=explore)

            if action.action_type.value == "proceed":
                # 不再澄清，进入规划
                reward = ClarificationReward(turn_cost=0.0).total
                ep.add_clarification_step(ClarificationTransition(
                    state_vec=clar_state.to_vector() if hasattr(clar_state, 'to_vector')
                    else self.clar_policy._pad_state_vector(clar_state),
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    next_state_vec=None,
                    done=True,
                ))
                break

            # 构造澄清问题
            question = ClarificationQuestion(
                slot=action.slot or "",
                question=f"请问{action.slot}是什么？",
                style=ClarificationStyle.OPEN_ENDED,
            )

            # 用户模拟回答
            answer, dropout_prob = self.simulator.answer_clarification(
                question, user_pref
            )

            # 更新任务状态（填充槽位）
            if action.slot in task.uncertainty_slots:
                task.uncertainty_slots[action.slot] = answer

            # 计算即时奖励
            unc_reduction = self._compute_uncertainty_reduction(task, action.slot)
            reward = ClarificationReward(
                turn_cost=1.0,
                constraint_reduction=unc_reduction,
                user_dropout_penalty=dropout_prob,
            ).total

            # 用户流失检查
            if np.random.random() < dropout_prob * 0.3:
                user_dropped_out = True
                ep.add_clarification_step(ClarificationTransition(
                    state_vec=self.clar_policy._pad_state_vector(clar_state),
                    action=action,
                    log_prob=log_prob,
                    reward=reward - 1.0,  # 额外惩罚
                    next_state_vec=None,
                    done=True,
                ))
                break

            next_state = self._build_clar_state(task, _round + 1)
            ep.add_clarification_step(ClarificationTransition(
                state_vec=self.clar_policy._pad_state_vector(clar_state),
                action=action,
                log_prob=log_prob,
                reward=reward,
                next_state_vec=self.clar_policy._pad_state_vector(next_state),
                done=False,
            ))

        # ── Phase 2: Planning（简化模拟）──
        if not user_dropped_out:
            plan_summary = self._simulate_planning(task, user_pref, ep)
            satisfaction = self.simulator.evaluate_plan(plan_summary, user_pref)
            ep.final_task_success = satisfaction >= 0.7
            ep.final_plan_score = satisfaction

        ep.total_turns = len(ep.clarification_transitions)
        return ep

    # ---------------------------------------------------------------------------
    # 策略更新（REINFORCE）
    # ---------------------------------------------------------------------------

    def _update_policies(self, episodes: list[Episode]) -> dict:
        clar_losses = []
        plan_losses = []

        for ep in episodes:
            clar_returns, plan_returns = ep.compute_returns(self.gamma)

            # 更新澄清策略
            for t, (transition, G) in enumerate(
                zip(ep.clarification_transitions, clar_returns)
            ):
                state = self._vec_to_clar_state(transition.state_vec)
                loss = self.clar_policy.update(state, transition.action, G)
                clar_losses.append(loss)

            # 更新规划策略
            for transition, G in zip(ep.planning_transitions, plan_returns):
                state = self._vec_to_plan_state(transition.state_vec)
                loss = self.plan_policy.update(state, transition.action, G)
                plan_losses.append(loss)

        success_rate = np.mean([float(ep.final_task_success) for ep in episodes])
        avg_turns = np.mean([ep.total_turns for ep in episodes])
        avg_score = np.mean([ep.final_plan_score for ep in episodes])

        return {
            "clar_loss": float(np.mean(clar_losses)) if clar_losses else 0.0,
            "plan_loss": float(np.mean(plan_losses)) if plan_losses else 0.0,
            "success_rate": float(success_rate),
            "avg_turns": float(avg_turns),
            "avg_plan_score": float(avg_score),
            "num_episodes": len(episodes),
        }

    # ---------------------------------------------------------------------------
    # 评估
    # ---------------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 50) -> dict:
        """在探索关闭的情况下评估策略性能。"""
        self.clar_policy.set_training(False)
        self.plan_policy.set_training(False)

        episodes = []
        for _ in range(num_episodes):
            user_pref = HiddenUserPreference.sample_random()
            task = self._sample_task(user_pref)
            ep = self._run_episode(task, user_pref, explore=False)
            episodes.append(ep)

        self.clar_policy.set_training(True)
        self.plan_policy.set_training(True)

        return {
            "success_rate": float(np.mean([ep.final_task_success for ep in episodes])),
            "avg_turns": float(np.mean([ep.total_turns for ep in episodes])),
            "avg_plan_score": float(np.mean([ep.final_plan_score for ep in episodes])),
            "num_episodes": num_episodes,
        }

    # ---------------------------------------------------------------------------
    # 检查点
    # ---------------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> None:
        self.clar_policy.save(str(self.checkpoint_dir / f"clar_policy_{tag}.pkl"))
        self.plan_policy.save(str(self.checkpoint_dir / f"plan_policy_{tag}.pkl"))
        log_path = self.checkpoint_dir / f"training_log_{tag}.json"
        with open(log_path, "w") as f:
            json.dump(self._training_log, f, indent=2)
        logger.info(f"检查点已保存: {tag}")

    def load_checkpoint(self, tag: str = "latest") -> None:
        self.clar_policy.load(str(self.checkpoint_dir / f"clar_policy_{tag}.pkl"))
        self.plan_policy.load(str(self.checkpoint_dir / f"plan_policy_{tag}.pkl"))

    # ---------------------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------------------

    def _sample_task(self, user_pref: HiddenUserPreference) -> ShoppingTask:
        """从用户偏好采样一个购物任务（初始不确定性较高）。"""
        from shopping_agent.common.types import Constraint, ConstraintSeverity
        import random as _random

        # 随机选择场景对应的品类
        scenario_categories = {
            "home": ["monitor", "keyboard", "mouse"],
            "travel": ["laptop", "bag"],
            "outdoor": ["tent", "sleeping_bag", "backpack"],
            "office": ["chair", "desk", "monitor"],
        }
        categories = scenario_categories.get(
            user_pref.usage_scenario,
            ["monitor", "keyboard"]
        )

        # 初始任务：部分约束已知，部分不确定
        known_budget = _random.random() > (1 - user_pref.clarity)
        constraints = []
        if known_budget:
            constraints.append(Constraint(
                key="budget_total",
                value=user_pref.budget_total,
                severity=ConstraintSeverity.HARD,
                source="user",
            ))

        uncertainty_slots = {slot: None for slot in ALL_SLOTS[:6]}
        if known_budget:
            uncertainty_slots["budget_total"] = user_pref.budget_total

        return ShoppingTask(
            task_id=str(uuid.uuid4()),
            task_type=TaskType.BUNDLE,
            categories=categories,
            raw_query=f"{user_pref.usage_scenario}场景购物",
            constraints=constraints,
            uncertainty_slots=uncertainty_slots,
            uncertainty_score=1.0 - user_pref.clarity,
        )

    def _build_clar_state(self, task: ShoppingTask, round_idx: int) -> ClarificationState:
        """从任务状态构建澄清 POMDP 状态。"""
        slot_names = list(task.uncertainty_slots.keys())[:len(ALL_SLOTS)]

        unc = np.array([
            0.0 if task.uncertainty_slots.get(s) is not None else 1.0
            for s in slot_names
        ])
        # 影响分：预算和品类场景影响最大
        impact_map = {
            "budget_total": 1.0, "usage_scenario": 0.9, "delivery_days": 0.7,
            "brand_preference": 0.6, "size": 0.8, "color": 0.2,
        }
        imp = np.array([impact_map.get(s, 0.3) for s in slot_names])

        return ClarificationState(
            slot_names=slot_names,
            uncertainty_vector=unc,
            impact_vector=imp,
            conversation_round=round_idx,
            budget_known=task.uncertainty_slots.get("budget_total") is not None,
            task_type_id=list(TaskType).index(task.task_type),
            profile_features=np.array([0.5, 0.5, 0.5, len(task.categories) / 5.0]),
        )

    def _compute_uncertainty_reduction(self, task: ShoppingTask, slot: Optional[str]) -> float:
        """计算填充某个槽位带来的不确定性降低量。"""
        if slot is None:
            return 0.0
        total = len([v for v in task.uncertainty_slots.values() if v is None])
        return 1.0 / max(total + 1, 1)

    def _simulate_planning(
        self, task: ShoppingTask, user_pref: HiddenUserPreference, ep: Episode
    ) -> dict:
        """简化的规划模拟（用于计算 episode reward）。"""
        budget = user_pref.budget_total
        n_slots = len(task.categories)
        per_slot = budget * 0.8 / max(n_slots, 1)

        plan_state = PlanningState(
            budget_total=budget,
            budget_used=0.0,
            total_slots=n_slots,
            filled_slots=0,
        )

        brands_chosen = []
        delivery_days = []

        for i, slot in enumerate(task.categories):
            action, log_prob = self.plan_policy.decide(plan_state, explore=True)

            # 根据动作选择商品（简化：不同 action 对应不同选品策略）
            price = per_slot * [0.9, 0.75, 0.95][action.action_id % 3]
            brand = user_pref.brand_preference[0] if action.action_id == 0 else "Samsung"
            days = 2 if action.action_id != 1 else 3

            brands_chosen.append(brand)
            delivery_days.append(days)
            plan_state.budget_used += price
            plan_state.filled_slots += 1

            # 规划即时奖励
            plan_reward = PlanningReward(
                constraint_score=1.0 if plan_state.budget_used <= budget else 0.0,
                preference_score=1.0 if brand in user_pref.brand_preference else 0.3,
                value_score=0.8,
            )
            next_state = PlanningState(
                budget_total=budget,
                budget_used=plan_state.budget_used,
                total_slots=n_slots,
                filled_slots=plan_state.filled_slots,
            )
            ep.add_planning_step(PlanningTransition(
                state_vec=plan_state.to_vector(),
                action=action,
                log_prob=log_prob,
                reward=plan_reward.total,
                next_state_vec=next_state.to_vector(),
                done=(i == n_slots - 1),
            ))
            plan_state = next_state

        return {
            "total_price": plan_state.budget_used,
            "brands": brands_chosen,
            "max_delivery_days": max(delivery_days) if delivery_days else 3,
            "avg_rating": 4.5,
        }

    def _vec_to_clar_state(self, vec: np.ndarray) -> ClarificationState:
        """从向量重建澄清状态（仅用于 update 时传入 policy）。"""
        K = len(ALL_SLOTS)
        return ClarificationState(
            slot_names=ALL_SLOTS,
            uncertainty_vector=vec[:K],
            impact_vector=vec[K:2*K],
            conversation_round=int(vec[2*K] * 5),
            budget_known=vec[2*K+1] > 0.5,
            task_type_id=int(vec[2*K+2] * 5),
            profile_features=vec[2*K+3:],
        )

    def _vec_to_plan_state(self, vec: np.ndarray) -> PlanningState:
        """从向量重建规划状态。"""
        return PlanningState(
            budget_total=1.0,
            budget_used=float(vec[0]),
            total_slots=max(1, int(1.0 / max(1 - float(vec[1]), 0.01))),
            filled_slots=int(float(vec[1]) * 5),
            constraint_sat_partial=float(vec[2]),
            preference_match_partial=float(vec[3]),
            incompatible_risk=float(vec[4]),
        )
