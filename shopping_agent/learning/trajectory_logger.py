"""
TrajectoryLogger — RL 轨迹记录器。

将 agent 在每次交互中的决策轨迹记录为 JSONL 格式，用于：
  1. 离线分析决策质量
  2. 行为克隆（模仿学习）预训练
  3. RL 训练数据复盘

记录格式（每行一个 JSON）：
  {
    "episode_id": "...",
    "step": 0,
    "phase": "clarification" | "planning",
    "state": {...},          # 状态向量及语义描述
    "action": {...},         # 动作类型及参数
    "reward": 0.0,           # 即时奖励
    "done": false,           # 是否终止
    "info": {...}            # 附加信息（log_prob / task_id / ...）
  }
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class TrajectoryLogger:
    """
    线程安全的 RL 轨迹记录器。

    一个 episode 对应一次完整的购物任务（从意图到规划结束）。
    每个步骤记录 (state, action, reward, done, info)。

    用法：
        logger = TrajectoryLogger()
        episode_id = logger.begin_episode(task_id="t001", user_id="u1")
        logger.log_step(episode_id, phase="clarification", state={...}, action={...}, reward=0.0)
        logger.end_episode(episode_id, total_reward=1.0, success=True)
    """

    def __init__(self, log_dir: str = "logs", filename: str = "trajectories.jsonl"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._log_dir / filename
        self._lock = threading.Lock()
        self._episodes: dict[str, dict] = {}  # episode_id → metadata
        self._step_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Episode 生命周期
    # ------------------------------------------------------------------

    def begin_episode(
        self,
        task_id: str,
        user_id: str = "unknown",
        mode: str = "heuristic",
        metadata: Optional[dict] = None,
    ) -> str:
        """开始一个新 episode，返回 episode_id。"""
        episode_id = str(uuid.uuid4())[:12]
        self._episodes[episode_id] = {
            "episode_id": episode_id,
            "task_id": task_id,
            "user_id": user_id,
            "mode": mode,
            "started_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._step_counters[episode_id] = 0
        return episode_id

    def log_step(
        self,
        episode_id: str,
        phase: str,
        state: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        next_state: Optional[dict[str, Any]] = None,
        done: bool = False,
        info: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        记录一个决策步骤。

        参数：
          episode_id  — begin_episode 返回的 ID
          phase       — "clarification" | "planning"
          state       — 状态描述 dict（建议包含 vector 和 semantic 两部分）
          action      — 动作描述 dict（type / slot / action_id 等）
          reward      — 即时奖励
          next_state  — 下一状态（可选）
          done        — 是否终止
          info        — 附加信息（log_prob / constraint_score / ...）
        """
        step = self._step_counters.get(episode_id, 0)
        self._step_counters[episode_id] = step + 1

        record = {
            "episode_id": episode_id,
            "task_id": self._episodes.get(episode_id, {}).get("task_id"),
            "step": step,
            "phase": phase,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "info": info or {},
            "timestamp": datetime.now().isoformat(),
        }
        self._write(record)

    def end_episode(
        self,
        episode_id: str,
        total_reward: float,
        success: bool,
        summary: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        结束 episode，写入汇总记录。
        """
        meta = self._episodes.pop(episode_id, {})
        record = {
            "episode_id": episode_id,
            "task_id": meta.get("task_id"),
            "step": "__episode_end__",
            "phase": "summary",
            "total_reward": total_reward,
            "success": success,
            "num_steps": self._step_counters.pop(episode_id, 0),
            "mode": meta.get("mode"),
            "started_at": meta.get("started_at"),
            "ended_at": datetime.now().isoformat(),
            "summary": summary or {},
        }
        self._write(record)

    # ------------------------------------------------------------------
    # 便捷方法：从 POMDP 对象直接记录
    # ------------------------------------------------------------------

    def log_clarification_step(
        self,
        episode_id: str,
        state_vector: list[float],
        action_type: str,
        slot: Optional[str],
        log_prob: float,
        reward: float,
        done: bool,
        round_idx: int,
    ) -> None:
        """从澄清 POMDP 步骤直接记录。"""
        self.log_step(
            episode_id=episode_id,
            phase="clarification",
            state={"vector": state_vector, "round": round_idx},
            action={"type": action_type, "slot": slot},
            reward=reward,
            done=done,
            info={"log_prob": log_prob},
        )

    def log_planning_step(
        self,
        episode_id: str,
        state_vector: list[float],
        action_id: int,
        log_prob: float,
        reward: float,
        done: bool,
        budget_used: float,
        filled_slots: int,
    ) -> None:
        """从规划 POMDP 步骤直接记录。"""
        action_names = {0: "SELECT_BEST", 1: "SELECT_BUDGET", 2: "SELECT_SAFE"}
        self.log_step(
            episode_id=episode_id,
            phase="planning",
            state={"vector": state_vector, "budget_used": budget_used, "filled_slots": filled_slots},
            action={"action_id": action_id, "name": action_names.get(action_id, "?")},
            reward=reward,
            done=done,
            info={"log_prob": log_prob},
        )

    # ------------------------------------------------------------------
    # 离线读取（供分析脚本使用）
    # ------------------------------------------------------------------

    def load_episodes(self, max_episodes: Optional[int] = None) -> list[list[dict]]:
        """
        读取 JSONL 文件，按 episode_id 分组，返回 episode 列表。
        每个 episode 是该 episode 所有步骤的 list（包含 summary）。
        """
        if not self._path.exists():
            return []

        records: dict[str, list[dict]] = {}
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    eid = rec.get("episode_id", "unknown")
                    records.setdefault(eid, []).append(rec)
                except json.JSONDecodeError:
                    continue

        episodes = list(records.values())
        if max_episodes:
            episodes = episodes[-max_episodes:]
        return episodes

    def stats(self) -> dict[str, Any]:
        """快速统计 JSONL 文件中的 episode 数和步骤数。"""
        if not self._path.exists():
            return {"episodes": 0, "steps": 0, "file_size_kb": 0}

        episode_ids: set[str] = set()
        step_count = 0
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    episode_ids.add(rec.get("episode_id", ""))
                    if rec.get("step") != "__episode_end__":
                        step_count += 1
                except json.JSONDecodeError:
                    continue

        file_size_kb = os.path.getsize(self._path) / 1024
        return {
            "episodes": len(episode_ids),
            "steps": step_count,
            "file_size_kb": round(file_size_kb, 1),
            "path": str(self._path),
        }

    # ------------------------------------------------------------------
    # 内部写入（线程安全）
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_logger_instance: Optional[TrajectoryLogger] = None


def get_trajectory_logger(log_dir: str = "logs") -> TrajectoryLogger:
    """获取全局单例轨迹记录器。"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TrajectoryLogger(log_dir=log_dir)
    return _logger_instance
