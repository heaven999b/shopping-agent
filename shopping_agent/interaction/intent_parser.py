"""
IntentParser — 将用户自然语言转换为结构化 ShoppingTask。

双层解析架构：
  Layer 1 (LLM)：调用 Claude API 做深度语义理解（首选）
  Layer 2 (规则)：正则 + 关键词匹配 fallback（LLM 不可用/超时/解析失败时触发）

新增能力：
  - 置信度分数（confidence: 0~1）
  - 输出结构校验（_validate_parsed_data）
  - 品牌实体识别（从已知品牌列表匹配）
  - 约束槽位类型强校验（budget→float, delivery_days→int）
  - parse() 永远不会抛 IntentParseError，仅在两层都失败时返回最简任务
"""

from __future__ import annotations

import json
import re
import uuid
import copy
from datetime import datetime
from typing import Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only in minimal environments
    anthropic = None

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


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 已知品牌列表（用于实体识别，可扩展）
_KNOWN_BRANDS = {
    "sony", "bose", "apple", "samsung", "lg", "dell", "hp", "lenovo", "asus",
    "acer", "microsoft", "google", "xiaomi", "huawei", "oppo", "vivo", "oneplus",
    "logitech", "razer", "corsair", "steelseries", "hyperx", "jabra", "sennheiser",
    "anker", "baseus", "philips", "ikea", "herman miller", "secretlab", "keychron",
    "hhkb", "realforce", "aoc", "benq", "viewsonic", "asus rog", "msi",
}

# 品类关键词映射（中英文）
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "headset": ["耳机", "headset", "headphone", "降噪", "蓝牙耳机"],
    "monitor": ["显示器", "屏幕", "monitor", "display", "液晶"],
    "keyboard": ["键盘", "keyboard", "机械键盘"],
    "mouse": ["鼠标", "mouse"],
    "laptop": ["笔记本", "laptop", "notebook", "电脑", "本子"],
    "chair": ["椅子", "椅", "chair", "人体工学椅", "电竞椅"],
    "desk": ["桌子", "桌", "desk", "升降桌", "办公桌"],
    "desk_lamp": ["台灯", "lamp", "屏幕挂灯"],
    "cpu": ["cpu", "处理器", "processor"],
    "gpu": ["显卡", "gpu", "graphics"],
    "ram": ["内存", "ram", "memory"],
    "storage": ["硬盘", "ssd", "storage", "固态"],
}

# 任务类型关键词
_TASK_TYPE_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.BUNDLE: ["套装", "套", "一套", "组合", "配齐", "配一套", "bundle"],
    TaskType.COMPARISON: ["对比", "比较", "哪个好", "vs", "选哪个"],
    TaskType.GIFT: ["送礼", "礼物", "gift", "送人", "生日礼"],
    TaskType.REPLENISH: ["续购", "补货", "再买一个", "replenish"],
}

_PARSE_SYSTEM_PROMPT = """你是一个购物意图解析专家。
从用户的自然语言输入中提取结构化购物任务信息，以 JSON 格式返回。

JSON 格式：
{
  "task_type": "single|bundle|comparison|gift|replenish",
  "categories": ["品类1", "品类2"],
  "hard_constraints": {
    "budget_total": 数字或null,
    "delivery_days": 数字或null,
    "brand": "品牌名"或null
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
  "confidence": 0.0到1.0（你对解析结果的置信度）,
  "conflict_pairs": [
    {"a": "约束A", "b": "约束B", "description": "冲突说明"}
  ]
}

规则：
- budget_total 单位为人民币元（整数或小数）
- delivery_days 为最大可接受天数（整数）
- uncertainty_score: 0=完全确定，1=完全不确定
- confidence: 1=你非常确信解析正确，0=你不确定
- 若预算极低但要求高端品牌，标记为冲突
- implicit_needs: 从场景词推断，如"出差"→["轻便","防摔"]，"居家"→["耐用","实惠"]
- 只返回 JSON，不要附加说明文字"""

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
  "uncertainty_score": 修正后的不确定性分数,
  "confidence": 0.0到1.0
}

只返回 JSON，不要附加说明文字。"""


# ---------------------------------------------------------------------------
# Rule-based Fallback Parser
# ---------------------------------------------------------------------------

class RuleFallbackParser:
    """
    基于正则 + 关键词的轻量级意图解析器。

    当 LLM 不可用/超时/解析失败时启用，保证系统基本可用性。
    精度低于 LLM，但覆盖大多数简单购物场景。
    """

    # 预算提取：支持 "2000以内"、"预算3000"、"不超过500元"、"5k"
    _BUDGET_PATTERNS = [
        r"预算[约大概]*\s*([0-9]+(?:\.[0-9]+)?)\s*[元块万k]?",
        r"([0-9]+(?:\.[0-9]+)?)\s*[元块]?\s*以内",
        r"不超过\s*([0-9]+(?:\.[0-9]+)?)\s*[元块]?",
        r"([0-9]+(?:\.[0-9]+)?)\s*[元块万k]\s*左右",
        r"budget[:\s]*([0-9]+)",
    ]

    # 配送天数
    _DELIVERY_PATTERNS = [
        r"([0-9]+)\s*天内",
        r"([0-9]+)\s*天到",
        r"明天到",
        r"次日达",
        r"当天到",
    ]

    def parse(self, user_input: str) -> dict:
        """返回与 LLM 格式兼容的 dict，附带 confidence 分数。"""
        text = user_input.lower()

        budget = self._extract_budget(text)
        delivery_days = self._extract_delivery(text)
        categories = self._extract_categories(text)
        brands = self._extract_brands(text)
        task_type = self._detect_task_type(text, categories)
        implicit_needs = self._extract_implicit_needs(text)
        uncertainty_slots = self._estimate_uncertainty_slots(budget, delivery_days, brands)
        uncertainty_score = len(uncertainty_slots) * 0.25

        # 置信度：有品类 + 有预算 = 高置信度
        confidence = 0.5
        if categories:
            confidence += 0.2
        if budget:
            confidence += 0.2
        if brands:
            confidence += 0.1

        hard_constraints: dict = {}
        if budget:
            hard_constraints["budget_total"] = budget
        if delivery_days:
            hard_constraints["delivery_days"] = delivery_days
        if brands and len(brands) == 1:
            hard_constraints["brand"] = brands[0]

        return {
            "task_type": task_type.value,
            "categories": categories,
            "hard_constraints": hard_constraints,
            "soft_preferences": {
                "brands_preferred": brands if len(brands) > 1 else [],
            },
            "implicit_needs": implicit_needs,
            "uncertainty_slots": uncertainty_slots,
            "uncertainty_score": min(1.0, uncertainty_score),
            "confidence": round(confidence, 2),
            "conflict_pairs": [],
            "_source": "rule_fallback",
        }

    def _extract_budget(self, text: str) -> Optional[float]:
        for pattern in self._BUDGET_PATTERNS:
            m = re.search(pattern, text)
            if m:
                if "万" in text[m.start():m.end()]:
                    return float(m.group(1)) * 10000
                if "k" in text[m.start():m.end()].lower():
                    return float(m.group(1)) * 1000
                return float(m.group(1))
        return None

    def _extract_delivery(self, text: str) -> Optional[int]:
        if "当天到" in text:
            return 0
        if "明天到" in text or "次日达" in text:
            return 1
        for pattern in self._DELIVERY_PATTERNS[:2]:
            m = re.search(pattern, text)
            if m:
                return int(m.group(1))
        return None

    def _extract_categories(self, text: str) -> list[str]:
        found = []
        for category, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                found.append(category)
        return found

    def _extract_brands(self, text: str) -> list[str]:
        found = []
        for brand in _KNOWN_BRANDS:
            if brand in text:
                found.append(brand.title())
        return found

    def _detect_task_type(self, text: str, categories: list[str]) -> TaskType:
        for task_type, keywords in _TASK_TYPE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return task_type
        if len(categories) > 1:
            return TaskType.BUNDLE
        return TaskType.SINGLE

    def _extract_implicit_needs(self, text: str) -> list[str]:
        needs = []
        hints = {
            "出差": "轻便防摔",
            "商务": "商务风格",
            "游戏": "高性能",
            "居家": "耐用实惠",
            "学生": "性价比",
            "办公": "生产力",
            "户外": "便携耐用",
        }
        for keyword, need in hints.items():
            if keyword in text:
                needs.append(need)
        return needs

    def _estimate_uncertainty_slots(
        self, budget: Optional[float], delivery_days: Optional[int], brands: list[str]
    ) -> dict:
        slots = {}
        if budget is None:
            slots["budget_total"] = None
        if delivery_days is None:
            slots["delivery_days"] = None
        if not brands:
            slots["brand_preference"] = "uncertain"
        return slots


# ---------------------------------------------------------------------------
# Output Validator
# ---------------------------------------------------------------------------

class ParsedDataValidator:
    """
    对 LLM 或规则解析的 dict 做结构校验 + 类型修正。
    返回 (cleaned_data, warnings)
    """

    def validate(self, data: dict) -> tuple[dict, list[str]]:
        warnings = []

        # 必填字段
        if not data.get("categories"):
            data["categories"] = []
            warnings.append("categories 为空，可能解析失败")

        # 类型修正
        hard = data.setdefault("hard_constraints", {})

        budget = hard.get("budget_total")
        if budget is not None:
            try:
                hard["budget_total"] = float(budget)
            except (ValueError, TypeError):
                warnings.append(f"budget_total 类型错误: {budget!r}，已移除")
                hard.pop("budget_total", None)

        days = hard.get("delivery_days")
        if days is not None:
            try:
                hard["delivery_days"] = int(days)
            except (ValueError, TypeError):
                warnings.append(f"delivery_days 类型错误: {days!r}，已移除")
                hard.pop("delivery_days", None)

        # 范围校验
        score = data.get("uncertainty_score", 0.5)
        try:
            data["uncertainty_score"] = max(0.0, min(1.0, float(score)))
        except (ValueError, TypeError):
            data["uncertainty_score"] = 0.5

        confidence = data.get("confidence", 0.7)
        try:
            data["confidence"] = max(0.0, min(1.0, float(confidence)))
        except (ValueError, TypeError):
            data["confidence"] = 0.5

        # 品类去重
        data["categories"] = list(dict.fromkeys(data.get("categories", [])))

        return data, warnings


# ---------------------------------------------------------------------------
# IntentParser（主类）
# ---------------------------------------------------------------------------

class IntentParser:
    """
    双层意图解析器：LLM（首选）→ 规则 fallback（降级）。

    parse() 保证永远返回有效的 ShoppingTask，不会抛异常。
    """

    def __init__(self):
        self._client = anthropic.Anthropic() if anthropic is not None else None
        self._fallback = RuleFallbackParser()
        self._validator = ParsedDataValidator()

    def parse(
        self,
        user_input: str,
        user_profile: Optional[UserProfile] = None,
    ) -> ShoppingTask:
        """
        首轮解析。
        LLM 失败时自动降级到规则解析，并标记 confidence 较低。
        """
        data, source = self._try_llm_parse(user_input, user_profile)
        data, warnings = self._validator.validate(data)

        # 若品类依然为空（两层都没识别出），尝试规则补充
        if not data.get("categories"):
            fallback_data = self._fallback.parse(user_input)
            if fallback_data.get("categories"):
                data["categories"] = fallback_data["categories"]
                warnings.append("品类由规则 fallback 补充")

        task = self._build_task(data, user_input)
        task.implicit_needs = data.get("implicit_needs", [])

        # 将 parse 来源记录到 raw_query（便于 attribution 追踪）
        if source == "fallback":
            task.raw_query = f"[rule-fallback] {user_input}"

        return task

    def parse_revision(
        self,
        user_input: str,
        existing_task: ShoppingTask,
        conversation_history: list[ConversationTurn],
    ) -> ShoppingTask:
        """续轮解析：在已有任务上做增量修正。LLM 失败时退回全新解析。"""
        history_text = "\n".join(
            f"用户: {t.user_input}\nAgent: {t.agent_response}"
            for t in conversation_history[-3:]
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
            data, _ = self._validator.validate(data)
            return self._apply_revision(existing_task, data, user_input)
        except Exception:
            # 修正解析失败：退回规则式增量修正，而不是丢失原任务上下文
            return self._fallback_revision(
                user_input, existing_task, conversation_history
            )

    # ---------------------------------------------------------------------------
    # 内部方法
    # ---------------------------------------------------------------------------

    def _try_llm_parse(
        self, user_input: str, user_profile: Optional[UserProfile]
    ) -> tuple[dict, str]:
        """
        尝试 LLM 解析，失败则降级到规则 fallback。
        返回 (data_dict, source_label)
        """
        prompt = self._build_parse_prompt(user_input, user_profile)
        try:
            raw = self._call_llm(_PARSE_SYSTEM_PROMPT, prompt)
            data = json.loads(raw)
            data.setdefault("confidence", 0.85)
            return data, "llm"
        except Exception:
            data = self._fallback.parse(user_input)
            return data, "fallback"

    def _build_parse_prompt(
        self, user_input: str, user_profile: Optional[UserProfile]
    ) -> str:
        profile_hint = ""
        if user_profile:
            if user_profile.brand_weights:
                top_brands = sorted(
                    user_profile.brand_weights.items(), key=lambda x: x[1], reverse=True
                )[:3]
                profile_hint = f"\n用户历史偏好品牌：{[b for b, _ in top_brands]}"
            if user_profile.size_profile:
                profile_hint += f"\n用户尺码：{user_profile.size_profile}"
        return f"用户输入：{user_input}{profile_hint}"

    def _call_llm(self, system: str, user: str) -> str:
        if self._client is None:
            raise IntentParseError("anthropic package is not installed")

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
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                return text
        raise IntentParseError("LLM 未返回有效文本")

    def _build_task(self, data: dict, raw_query: str) -> ShoppingTask:
        constraints = []
        hard = data.get("hard_constraints", {})
        for key, value in hard.items():
            if value is not None:
                constraints.append(Constraint(
                    key=key, value=value,
                    severity=ConstraintSeverity.HARD, source="user",
                ))

        soft = data.get("soft_preferences", {})
        for key, value in soft.items():
            if value is not None and value != [] and value != "":
                constraints.append(Constraint(
                    key=key, value=value,
                    severity=ConstraintSeverity.SOFT, source="user",
                ))

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

    def _apply_revision(
        self, task: ShoppingTask, data: dict, user_input: str
    ) -> ShoppingTask:
        changed_fields = data.get("changed_fields", {})

        for key, value in data.get("new_hard_constraints", {}).items():
            existing = next((c for c in task.constraints if c.key == key), None)
            if existing:
                existing.value = value
            else:
                task.constraints.append(Constraint(
                    key=key, value=value,
                    severity=ConstraintSeverity.HARD, source="user",
                ))

        for key, value in data.get("new_soft_preferences", {}).items():
            existing = next((c for c in task.constraints if c.key == key), None)
            if existing:
                existing.value = value
            else:
                task.constraints.append(Constraint(
                    key=key, value=value,
                    severity=ConstraintSeverity.SOFT, source="user",
                ))

        removed = set(data.get("removed_constraints", []))
        task.constraints = [c for c in task.constraints if c.key not in removed]

        if "uncertainty_score" in data:
            task.uncertainty_score = float(data["uncertainty_score"])

        revision = TaskRevision(
            round_index=len(task.revision_history),
            changed_fields=changed_fields,
            user_utterance=user_input,
            timestamp=datetime.now(),
        )
        task.apply_revision(revision)
        return task

    def _fallback_revision(
        self,
        user_input: str,
        existing_task: ShoppingTask,
        conversation_history: list[ConversationTurn],
    ) -> ShoppingTask:
        task = copy.deepcopy(existing_task)
        parsed = self._fallback.parse(user_input)
        asked_slot = self._infer_last_clarification_slot(conversation_history)
        changed_fields: dict[str, dict] = {}

        hard_updates = parsed.get("hard_constraints", {})
        soft_updates = parsed.get("soft_preferences", {})

        if parsed.get("categories"):
            before = task.categories.copy()
            task.categories = parsed["categories"]
            changed_fields["categories"] = {"from": before, "to": task.categories}

        for key, value in hard_updates.items():
            if value is None:
                continue
            before = task.get_hard_constraints().get(key)
            self._upsert_constraint(task, key, value, ConstraintSeverity.HARD)
            changed_fields[key] = {"from": before, "to": value}

        for key, value in soft_updates.items():
            if value in (None, [], ""):
                continue
            before = task.get_soft_preferences().get(key)
            self._upsert_constraint(task, key, value, ConstraintSeverity.SOFT)
            changed_fields[key] = {"from": before, "to": value}

        implicit_needs = parsed.get("implicit_needs", [])
        if implicit_needs:
            merged = list(dict.fromkeys(task.implicit_needs + implicit_needs))
            if merged != task.implicit_needs:
                changed_fields["implicit_needs"] = {
                    "from": task.implicit_needs.copy(),
                    "to": merged,
                }
                task.implicit_needs = merged

        if asked_slot:
            if asked_slot == "categories" and task.categories:
                task.uncertainty_slots.pop("categories", None)
            elif asked_slot in {"budget_total", "delivery_days"} and asked_slot in hard_updates:
                task.uncertainty_slots.pop(asked_slot, None)
            elif asked_slot in {"brand_preference", "platform", "style", "color"} and asked_slot in soft_updates:
                task.uncertainty_slots.pop(asked_slot, None)
            elif user_input.strip():
                task.uncertainty_slots.pop(asked_slot, None)

        task.uncertainty_score = min(
            1.0,
            len(task.uncertainty_slots) * 0.3,
        )

        revision = TaskRevision(
            round_index=len(task.revision_history),
            changed_fields=changed_fields,
            user_utterance=user_input,
            timestamp=datetime.now(),
        )
        task.apply_revision(revision)
        return task

    @staticmethod
    def _upsert_constraint(
        task: ShoppingTask,
        key: str,
        value,
        severity: ConstraintSeverity,
    ) -> None:
        existing = next((c for c in task.constraints if c.key == key), None)
        if existing:
            existing.value = value
            existing.severity = severity
        else:
            task.constraints.append(
                Constraint(key=key, value=value, severity=severity, source="user")
            )

    @staticmethod
    def _infer_last_clarification_slot(
        conversation_history: list[ConversationTurn],
    ) -> Optional[str]:
        if not conversation_history:
            return None

        last_response = conversation_history[-1].agent_response
        if "预算" in last_response:
            return "budget_total"
        if "几天内" in last_response or "次日达" in last_response:
            return "delivery_days"
        if "什么场景" in last_response:
            return "usage_scenario"
        if "品牌偏好" in last_response:
            return "brand_preference"
        if "平台" in last_response:
            return "platform"
        if "风格" in last_response:
            return "style"
        if "颜色" in last_response:
            return "color"
        if "尺寸" in last_response or "尺码" in last_response:
            return "size"
        if "兼容" in last_response:
            return "compatibility"
        if "买什么" in last_response or "哪些品类" in last_response:
            return "categories"
        return None
