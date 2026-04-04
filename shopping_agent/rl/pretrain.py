"""
行为克隆（Behavior Cloning）预训练器。

从 logs/trajectories.jsonl 中读取专家轨迹，
用监督学习方法初始化 ClarificationPolicy 和 PlanningPolicy，
使策略在进入 REINFORCE 在线训练前已有合理的初始参数。

算法：
  最小化负对数似然（Cross-Entropy Loss）：
    L_BC = -Σ log π_θ(a_expert | s)

  等价于让策略模仿专家动作的分布。

优势（相比随机初始化直接 RL）：
  1. 避免冷启动探索阶段的大量无效尝试
  2. 在数据量有限时（<1000 条轨迹）效果尤其显著
  3. 为后续 REINFORCE 提供良好的初始点（Policy Gradient 对初始值敏感）

使用方式：
  python -m shopping_agent.rl.pretrain
  python -m shopping_agent.rl.pretrain --epochs 20 --lr 0.01 --output models/

论文参考：
  - Pomerleau 1991, ALVINN（行为克隆鼻祖）
  - Ho & Ermon 2016, GAIL（行为克隆的局限及改进）
  - 本项目：BC 作预训练，REINFORCE 作在线微调（两阶段训练）
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from shopping_agent.rl.policy import (
    ALL_SLOTS,
    CLAR_NUM_ACTIONS,
    MAX_SLOTS,
    LinearSoftmaxPolicy,
    RLClarificationPolicy,
    RLPlanningPolicy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 轨迹数据加载
# ---------------------------------------------------------------------------

def load_trajectories(path: str) -> list[dict]:
    """
    从 JSONL 文件加载轨迹步骤。

    只加载 step != "__episode_end__" 的行（过滤掉 summary 行）。
    返回所有步骤的 dict 列表。
    """
    traj_path = Path(path)
    if not traj_path.exists():
        logger.warning("轨迹文件不存在: %s", path)
        return []

    steps: list[dict] = []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("step") != "__episode_end__":
                    steps.append(rec)
            except json.JSONDecodeError:
                continue

    logger.info("加载轨迹步骤 %d 条 from %s", len(steps), path)
    return steps


def split_by_phase(steps: list[dict]) -> tuple[list[dict], list[dict]]:
    """将步骤按 phase 拆分为澄清步骤和规划步骤。"""
    clar_steps = [s for s in steps if s.get("phase") == "clarification"]
    plan_steps = [s for s in steps if s.get("phase") == "planning"]
    return clar_steps, plan_steps


# ---------------------------------------------------------------------------
# 特征向量提取
# ---------------------------------------------------------------------------

def _extract_clar_state_vec(step: dict) -> Optional[np.ndarray]:
    """
    从澄清步骤中提取定长状态向量。

    兼容 TrajectoryLogger.log_clarification_step 记录的格式：
      state = {"vector": [...], "round": int}
    """
    state = step.get("state", {})
    vec = state.get("vector")
    if vec is None:
        return None
    arr = np.array(vec, dtype=np.float32)
    # pad / truncate 到 ClarificationPolicy 的期望维度
    # feature_dim = MAX_SLOTS * 2 + 3 + 4（profile_feature_dim=4）
    expected_dim = MAX_SLOTS * 2 + 3 + 4
    if len(arr) < expected_dim:
        arr = np.concatenate([arr, np.zeros(expected_dim - len(arr))])
    else:
        arr = arr[:expected_dim]
    return arr


def _extract_clar_action_id(step: dict) -> Optional[int]:
    """
    从澄清步骤中提取动作 ID。

    动作格式：{"type": "PROCEED" | "ASK_SLOT", "slot": str}
    """
    action = step.get("action", {})
    action_type = action.get("type", "")
    if action_type == "PROCEED":
        return 0
    elif action_type == "ASK_SLOT":
        slot = action.get("slot")
        if slot and slot in ALL_SLOTS:
            return ALL_SLOTS.index(slot) + 1
        # 未知 slot → PROCEED（安全兜底）
        return 0
    return None


def _extract_plan_state_vec(step: dict) -> Optional[np.ndarray]:
    """
    从规划步骤中提取状态向量。

    兼容 TrajectoryLogger.log_planning_step 记录的格式：
      state = {"vector": [...], "budget_used": float, "filled_slots": int}
    """
    state = step.get("state", {})
    vec = state.get("vector")
    if vec is None:
        return None
    arr = np.array(vec, dtype=np.float32)
    # RLPlanningPolicy.feature_dim = len(PlanningState.__dataclass_fields__) = 7
    expected_dim = 7
    if len(arr) < expected_dim:
        arr = np.concatenate([arr, np.zeros(expected_dim - len(arr))])
    else:
        arr = arr[:expected_dim]
    return arr


def _extract_plan_action_id(step: dict) -> Optional[int]:
    """
    从规划步骤中提取动作 ID（0/1/2）。
    """
    action = step.get("action", {})
    aid = action.get("action_id")
    if aid is not None and 0 <= int(aid) <= 2:
        return int(aid)
    return None


# ---------------------------------------------------------------------------
# 行为克隆训练器
# ---------------------------------------------------------------------------

class BehaviorCloningTrainer:
    """
    行为克隆预训练器。

    支持对 ClarificationPolicy 和 PlanningPolicy 分别做监督学习预训练，
    使策略参数在进入 REINFORCE 前收敛到专家分布附近。

    损失函数：
      L_BC = -1/N · Σ_i log π_θ(a_i | s_i)

    梯度更新（手动 SGD，与 REINFORCE 接口统一）：
      W += lr · (1 - π(a|s)) · φ(s)   （对正确动作的 one-hot log-softmax 梯度）
    """

    def __init__(self, learning_rate: float = 1e-3, batch_size: int = 32):
        self.lr = learning_rate
        self.batch_size = batch_size

    def pretrain_clarification(
        self,
        policy: RLClarificationPolicy,
        clar_steps: list[dict],
        epochs: int = 10,
    ) -> dict[str, Any]:
        """
        对澄清策略做行为克隆。

        参数：
            policy      — 待初始化的 RLClarificationPolicy 实例
            clar_steps  — 澄清轨迹步骤列表
            epochs      — 训练轮数

        返回：训练统计 dict（loss / accuracy）
        """
        # 提取有效样本
        samples: list[tuple[np.ndarray, int]] = []
        for step in clar_steps:
            sv = _extract_clar_state_vec(step)
            aid = _extract_clar_action_id(step)
            if sv is not None and aid is not None:
                samples.append((sv, aid))

        if not samples:
            logger.warning("澄清轨迹：无有效样本，跳过预训练")
            return {"trained": False, "reason": "no_valid_samples"}

        logger.info("澄清策略 BC 预训练：%d 样本，%d epochs", len(samples), epochs)

        inner_policy = policy._policy  # LinearSoftmaxPolicy
        orig_lr = inner_policy.lr
        inner_policy.lr = self.lr

        history = []
        for epoch in range(epochs):
            np.random.shuffle(samples)
            epoch_losses = []

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                batch_loss = 0.0
                for sv, aid in batch:
                    loss = self._bc_update(inner_policy, sv, aid)
                    batch_loss += loss
                epoch_losses.append(batch_loss / len(batch))

            avg_loss = float(np.mean(epoch_losses))
            accuracy = self._compute_accuracy(inner_policy, samples)
            history.append({"epoch": epoch + 1, "loss": round(avg_loss, 4),
                            "accuracy": round(accuracy, 4)})
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("[Clar BC] epoch=%d loss=%.4f acc=%.3f",
                            epoch + 1, avg_loss, accuracy)

        inner_policy.lr = orig_lr
        return {
            "trained": True,
            "samples": len(samples),
            "epochs": epochs,
            "final_loss": history[-1]["loss"] if history else None,
            "final_accuracy": history[-1]["accuracy"] if history else None,
            "history": history,
        }

    def pretrain_planning(
        self,
        policy: RLPlanningPolicy,
        plan_steps: list[dict],
        epochs: int = 10,
    ) -> dict[str, Any]:
        """
        对规划策略做行为克隆。
        """
        samples: list[tuple[np.ndarray, int]] = []
        for step in plan_steps:
            sv = _extract_plan_state_vec(step)
            aid = _extract_plan_action_id(step)
            if sv is not None and aid is not None:
                samples.append((sv, aid))

        if not samples:
            logger.warning("规划轨迹：无有效样本，跳过预训练")
            return {"trained": False, "reason": "no_valid_samples"}

        logger.info("规划策略 BC 预训练：%d 样本，%d epochs", len(samples), epochs)

        inner_policy = policy._policy
        orig_lr = inner_policy.lr
        inner_policy.lr = self.lr

        history = []
        for epoch in range(epochs):
            np.random.shuffle(samples)
            epoch_losses = []

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                batch_loss = 0.0
                for sv, aid in batch:
                    loss = self._bc_update(inner_policy, sv, aid)
                    batch_loss += loss
                epoch_losses.append(batch_loss / len(batch))

            avg_loss = float(np.mean(epoch_losses))
            accuracy = self._compute_accuracy(inner_policy, samples)
            history.append({"epoch": epoch + 1, "loss": round(avg_loss, 4),
                            "accuracy": round(accuracy, 4)})
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("[Plan BC] epoch=%d loss=%.4f acc=%.3f",
                            epoch + 1, avg_loss, accuracy)

        inner_policy.lr = orig_lr
        return {
            "trained": True,
            "samples": len(samples),
            "epochs": epochs,
            "final_loss": history[-1]["loss"] if history else None,
            "final_accuracy": history[-1]["accuracy"] if history else None,
            "history": history,
        }

    @staticmethod
    def _bc_update(policy: LinearSoftmaxPolicy, state_vec: np.ndarray,
                   expert_action_id: int) -> float:
        """
        行为克隆单步更新：模仿学习的监督梯度。

        等价于 REINFORCE with G=1（专家动作奖励固定为1，非专家动作奖励为0）。
        实现方式：直接调用 policy.update(state_vec, expert_action_id, G=1.0)
        并禁用基线（避免基线消除正确动作的梯度信号）。

        损失：L = -log π(a_expert | s)
        梯度：∂L/∂W = -(e_{a*} - π(·|s)) ⊗ φ(s)
        """
        p = policy.probs(state_vec)

        # 手动计算梯度（不用基线，BC 不需要 variance reduction）
        one_hot = np.zeros(policy.num_actions)
        if expert_action_id < policy.num_actions:
            one_hot[expert_action_id] = 1.0

        grad_logits = one_hot - p
        grad_W = np.outer(grad_logits, state_vec)

        policy.W += policy.lr * grad_W
        policy.b += policy.lr * grad_logits

        loss = -float(np.log(p[expert_action_id] + 1e-10))
        return loss

    @staticmethod
    def _compute_accuracy(policy: LinearSoftmaxPolicy,
                          samples: list[tuple[np.ndarray, int]]) -> float:
        """计算贪心准确率（greedy action == expert action 的比例）。"""
        if not samples:
            return 0.0
        correct = sum(
            1 for sv, aid in samples
            if policy.greedy(sv) == aid
        )
        return correct / len(samples)


# ---------------------------------------------------------------------------
# 合成专家轨迹（冷启动时没有真实轨迹时使用）
# ---------------------------------------------------------------------------

def generate_synthetic_trajectories(
    n_episodes: int = 50,
    output_path: str = "logs/trajectories.jsonl",
) -> int:
    """
    生成合成专家轨迹，用于冷启动 BC 预训练。

    策略规则（确定性专家）：
      澄清：uncertainty_score > 0.6 时 ASK budget_total，否则 PROCEED
      规划：第一轮选 SELECT_BEST_MATCH（action_id=0）

    这不是真实的专家，而是一个可以生成合理轨迹的启发式策略，
    用来帮助策略网络从一个合理的初始值开始，而不是完全随机。
    """
    from shopping_agent.learning.trajectory_logger import TrajectoryLogger

    log_path = Path(output_path)
    traj_logger = TrajectoryLogger(
        log_dir=str(log_path.parent),
        filename=log_path.name,
    )

    rng = np.random.default_rng(42)
    total_steps = 0

    for ep_idx in range(n_episodes):
        episode_id = traj_logger.begin_episode(
            task_id=f"synthetic_{ep_idx:04d}",
            user_id="synthetic_expert",
            mode="synthetic",
        )

        # 1. 澄清阶段（1-2 步）
        uncertainty = rng.uniform(0.2, 0.9)
        clar_done = uncertainty <= 0.5

        # 状态向量：MAX_SLOTS*2 + 3 + 4 维
        state_vec = np.zeros(MAX_SLOTS * 2 + 7)
        state_vec[0] = uncertainty       # 第0槽位不确定性
        state_vec[MAX_SLOTS] = 0.8      # 第0槽位影响力
        state_vec[MAX_SLOTS * 2] = 0.0  # round=0

        if clar_done:
            action_type, slot = "PROCEED", None
        else:
            action_type, slot = "ASK_SLOT", "budget_total"

        traj_logger.log_clarification_step(
            episode_id=episode_id,
            state_vector=state_vec.tolist(),
            action_type=action_type,
            slot=slot,
            log_prob=-0.5,
            reward=0.0 if not clar_done else 0.1,
            done=clar_done,
            round_idx=0,
        )
        total_steps += 1

        # 2. 规划阶段（1-3 步，每步选一个 slot）
        n_slots = rng.integers(1, 4)
        budget_total = rng.uniform(1000.0, 8000.0)
        budget_used = 0.0

        for slot_idx in range(n_slots):
            budget_remaining_ratio = (budget_total - budget_used) / budget_total
            slot_fill_ratio = slot_idx / n_slots
            is_last = slot_idx == n_slots - 1

            # 专家策略：预算充裕 → BEST_MATCH，预算紧 → BUDGET_OPT
            if budget_remaining_ratio > 0.5:
                action_id = 0  # SELECT_BEST_MATCH
            elif budget_remaining_ratio > 0.25:
                action_id = 1  # SELECT_BUDGET_OPT
            else:
                action_id = 2  # SELECT_SAFE

            plan_state_vec = np.array([
                budget_total,
                budget_used,
                n_slots,
                slot_idx,
                rng.uniform(0.7, 1.0),   # constraint_sat
                rng.uniform(0.5, 1.0),   # preference_match
                0.0,                     # incompatible_risk (7th field)
            ], dtype=np.float32)

            reward = 0.8 + rng.uniform(0.0, 0.2) if is_last else 0.0
            budget_used += rng.uniform(100.0, budget_total / n_slots)

            traj_logger.log_planning_step(
                episode_id=episode_id,
                state_vector=plan_state_vec.tolist(),
                action_id=action_id,
                log_prob=-0.3,
                reward=reward,
                done=is_last,
                budget_used=budget_used,
                filled_slots=slot_idx + 1,
            )
            total_steps += 1

        # Episode 结束
        traj_logger.end_episode(
            episode_id=episode_id,
            total_reward=rng.uniform(0.5, 1.0),
            success=True,
        )

    logger.info("生成合成轨迹 %d 条 episode，%d 步骤 → %s",
                n_episodes, total_steps, output_path)
    return total_steps


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_pretrain(
    traj_path: str = "logs/trajectories.jsonl",
    output_dir: str = "models",
    epochs: int = 10,
    lr: float = 5e-3,
    generate_synthetic: bool = False,
    n_synthetic: int = 50,
) -> dict[str, Any]:
    """
    执行行为克隆预训练流程。

    返回包含训练统计的 dict。
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # 冷启动：若无真实轨迹则生成合成数据
    if not Path(traj_path).exists() or generate_synthetic:
        logger.info("生成合成专家轨迹用于冷启动...")
        generate_synthetic_trajectories(n_synthetic, traj_path)

    # 加载轨迹
    steps = load_trajectories(traj_path)
    if not steps:
        logger.error("无轨迹数据，退出")
        return {"success": False, "reason": "no_data"}

    clar_steps, plan_steps = split_by_phase(steps)
    logger.info("澄清步骤: %d，规划步骤: %d", len(clar_steps), len(plan_steps))

    # 初始化策略
    clar_policy = RLClarificationPolicy(learning_rate=lr)
    plan_policy = RLPlanningPolicy(learning_rate=lr)

    # 行为克隆训练
    trainer = BehaviorCloningTrainer(learning_rate=lr)
    clar_stats = trainer.pretrain_clarification(clar_policy, clar_steps, epochs=epochs)
    plan_stats = trainer.pretrain_planning(plan_policy, plan_steps, epochs=epochs)

    # 保存模型
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clar_model_path = str(output_path / "clar_policy_pretrained.pkl")
    plan_model_path = str(output_path / "plan_policy_pretrained.pkl")

    clar_policy.save(clar_model_path)
    plan_policy.save(plan_model_path)

    logger.info("模型已保存: %s, %s", clar_model_path, plan_model_path)

    return {
        "success": True,
        "clarification": clar_stats,
        "planning": plan_stats,
        "model_paths": {
            "clarification": clar_model_path,
            "planning": plan_model_path,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="行为克隆预训练")
    parser.add_argument("--traj", default="logs/trajectories.jsonl", help="轨迹文件路径")
    parser.add_argument("--output", default="models", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-3, help="学习率")
    parser.add_argument("--synthetic", action="store_true", help="生成合成训练数据")
    parser.add_argument("--n-synthetic", type=int, default=50, help="合成 episode 数")
    args = parser.parse_args()

    result = run_pretrain(
        traj_path=args.traj,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        generate_synthetic=args.synthetic,
        n_synthetic=args.n_synthetic,
    )

    print("\n=== 预训练完成 ===")
    if result.get("clarification", {}).get("trained"):
        c = result["clarification"]
        print(f"澄清策略: {c['samples']} 样本，最终 loss={c['final_loss']:.4f}，acc={c['final_accuracy']:.3f}")
    if result.get("planning", {}).get("trained"):
        p = result["planning"]
        print(f"规划策略: {p['samples']} 样本，最终 loss={p['final_loss']:.4f}，acc={p['final_accuracy']:.3f}")
    print(f"模型路径: {result.get('model_paths', {})}")
