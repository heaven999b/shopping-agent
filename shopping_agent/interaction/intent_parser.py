"""
IntentParser — 将用户自然语言转换为结构化 ShoppingTask。

支持：
  - 首轮全新解析
  - 续轮增量修正（用户调整约束后的 diff 更新）
  - 约束冲突检测
  - 不确定性评分
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

import anthropic

from shopping_agent.common.constants import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from shopping_agent.common.exceptions import IntentParseError
from shopping_agent.common.types import (
    Constraint,
    ConstraintSeverity,
    ConflictPair,
    ConflictResolution,
    ShoppingTask,
    TaskRevision,
    TaskType,
    UserProfile,
)
from shopping_agent.agent.state import ConversationTurn


_PARSE_SYSTEM_PROMPT = """你是一个购物意图解析专家。
从用户的自然语言输入中提取结构化购物任务信息，以 JSON 格式返回。

JSON 格式：
{
  "task_type": "single|bundle|comparison|gift|replenish",
  "categories": ["品类1", "品类2"],
  "hard_constraints": {
    "budget_total": 数字或null,
    "delivery_days": 数字或null,
    "brands_required": ["品牌"] 或 null
  },
  "soft_preferences": {
    "brands_preferred": ["品牌"],
    "color": "颜色",
    "style": "风格",
    "platform": "平台"
  },
  "implicit_needs": ["推断出的隐式需求"],
  "uncertainty_slots": {"槽位名": null},
  "uncertainty_score": 0.0到1.0的数字,
  "conflict_pairs": [
    {"a": "约束A", "b": "约束B", "description": "冲突说明"}
  ]
}

规则：
- budget_total 单位为人民币元
- delivery_days 为最大可接受天数
- uncertainty_score: 0=完全确定，1=完全不确定
- 若预算极低但要求高端品牌，标记为冲突
- implicit_needs: 从场景词推断，如"出差"→轻便防摔，"家庭"→耐用实惠"""

_REVISION_SYSTEM_PROMPT = """你是一个购物意图修正专家。
用户在已有购物任务基础上提出了修正，提取本次修正的变更内容，以 JSON 格式返回。

JSON 格式：
{
  "changed_fields": {
    "字段名": {"from": 旧值, "to": 新值}
  },
  "new_hard_constraints": {},
  "new_soft_preferences": {},
  "removed_constraints": ["约束key列表"],
  "uncertainty_score": 修正后的不确定性分数
}"""


class IntentParser:
    def __init__(self):
        self._client = anthropic.Anthropic()

    def parse(self, user_input: str,
              user_profile: Optional[UserProfile] = None) -> ShoppingTask:
        """首轮解析：从用户输入生成全新 ShoppingTask。"""
        prompt = self._build_parse_prompt(user_input, user_profile)
        try:
            raw = self._call_llm(_PARSE_SYSTEM_PROMPT, prompt)
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            raise IntentParseError(f"意图解析失败: {e}") from e

        return self._build_task(data, user_input)

    def parse_revision(
        self,
        user_input: str,
        existing_task: ShoppingTask,
        conversation_history: list[ConversationTurn],
    ) -> ShoppingTask:
        """续轮解析：在已有任务上做增量修正。"""
        history_text = "\n".join(
            f"用户: {t.user_input}\nAgent: {t.agent_response}"
            for t in conversation_history[-3:]  # 最近 3 轮
        )
        existing_summary = json.dumps({
            "task_type": existing_task.task_type.value,
            "categories": existing_task.categories,
            "hard_constraints": existing_task.get_hard_constraints(),
            "soft_preferences": existing_task.get_soft_preferences(),
        }, ensure_ascii=False)

        prompt = (
            f"已有任务：\n{existing_summary}\n\n"
            f"近期对话：\n{history_text}\n\n"
            f"用户最新输入：{user_input}"
        )

        try:
            raw = self._call_llm(_REVISION_SYSTEM_PROMPT, prompt)
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            # 修正解析失败时，退回到全新解析
            return self.parse(user_input)

        return self._apply_revision(existing_task, data, user_input)

    # ---------------------------------------------------------------------------
    # 内部方法
    # ---------------------------------------------------------------------------

    def _build_parse_prompt(self, user_input: str,
                            user_profile: Optional[UserProfile]) -> str:
        profile_hint = ""
        if user_profile:
            if user_profile.brand_weights:
                top_brands = sorted(user_profile.brand_weights.items(),
                                    key=lambda x: x[1], reverse=True)[:3]
                profile_hint = f"\n用户历史偏好品牌：{[b for b, _ in top_brands]}"
            if user_profile.size_profile:
                profile_hint += f"\n用户尺码：{user_profile.size_profile}"

        return f"用户输入：{user_input}{profile_hint}"

    def _call_llm(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=DEFAULT_MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        for block in response.content:
            if block.type == "text":
                text = block.text.strip()
                # 提取 JSON（可能被 markdown 包裹）
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                return text
        raise IntentParseError("LLM 未返回有效文本")

    def _build_task(self, data: dict, raw_query: str) -> ShoppingTask:
        # 构建约束列表
        constraints = []
        hard = data.get("hard_constraints", {})
        for key, value in hard.items():
            if value is not None:
                constraints.append(Constraint(
                    key=key, value=value,
                    severity=ConstraintSeverity.HARD, source="user"
                ))

        soft = data.get("soft_preferences", {})
        for key, value in soft.items():
            if value is not None:
                constraints.append(Constraint(
                    key=key, value=value,
                    severity=ConstraintSeverity.SOFT, source="user"
                ))

        # 构建冲突列表
        conflict_pairs = []
        for cp in data.get("conflict_pairs", []):
            conflict_pairs.append(ConflictPair(
                constraint_a=cp.get("a", ""),
                constraint_b=cp.get("b", ""),
                description=cp.get("description", ""),
                resolution=ConflictResolution.ASK_USER,
            ))

        try:
            task_type = TaskType(data.get("task_type", "single"))
        except ValueError:
            task_type = TaskType.SINGLE

        return ShoppingTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            categories=data.get("categories", []),
            raw_query=raw_query,
            constraints=constraints,
            implicit_needs=data.get("implicit_needs", []),
            conflict_pairs=conflict_pairs,
            uncertainty_slots=data.get("uncertainty_slots", {}),
            uncertainty_score=float(data.get("uncertainty_score", 0.5)),
        )

    def _apply_revision(self, task: ShoppingTask, data: dict,
                        user_input: str) -> ShoppingTask:
        """将 revision diff 应用到已有任务。"""
        changed_fields = data.get("changed_fields", {})

        # 更新硬约束
        for key, value in data.get("new_hard_constraints", {}).items():
            # 找到已有约束并更新，或新增
            existing = next((c for c in task.constraints if c.key == key), None)
            if existing:
                existing.value = value
            else:
                task.constraints.append(Constraint(
                    key=key, value=value,
                    severity=ConstraintSeverity.HARD, source="user"
                ))

        # 更新软偏好
        for key, value in data.get("new_soft_preferences", {}).items():
            existing = next((c for c in task.constraints if c.key == key), None)
            if existing:
                existing.value = value
            else:
                task.constraints.append(Constraint(
                    key=key, value=value,
                    severity=ConstraintSeverity.SOFT, source="user"
                ))

        # 移除约束
        removed = set(data.get("removed_constraints", []))
        task.constraints = [c for c in task.constraints if c.key not in removed]

        # 更新不确定性分数
        if "uncertainty_score" in data:
            task.uncertainty_score = float(data["uncertainty_score"])

        # 记录修正历史
        revision = TaskRevision(
            round_index=len(task.revision_history),
            changed_fields=changed_fields,
            user_utterance=user_input,
            timestamp=datetime.now(),
        )
        task.apply_revision(revision)

        return task
