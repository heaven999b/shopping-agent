"""
Core data structures for SHOP-PLAN Shopping Agent.

All inter-module communication uses these types as the shared contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    SINGLE = "single"          # 购买单品
    BUNDLE = "bundle"          # 组合购买（如装机、穿搭）
    COMPARISON = "comparison"  # 对比选型
    GIFT = "gift"              # 礼品选购
    REPLENISH = "replenish"    # 补货/续购


class ConstraintSeverity(str, Enum):
    HARD = "hard"   # 违反 → 直接剔除候选
    SOFT = "soft"   # 违反 → 降分，但仍可出现


class ConflictResolution(str, Enum):
    ASK_USER = "ask_user"          # 询问用户选择
    RELAX_SOFT = "relax_soft"      # 自动放宽软约束
    INFORM_TRADEOFF = "inform_tradeoff"  # 告知 trade-off，让用户决定


class VerificationStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"    # 可继续，但需告知用户
    BLOCK = "block"  # 必须重规划
    STALE = "stale"  # 数据过期，需重新拉取


class ClarificationStyle(str, Enum):
    BINARY = "binary"          # 是/否
    CHOICE = "choice"          # 多选一（最多 3 个选项）
    OPEN_ENDED = "open_ended"  # 开放式


class ExplainStyle(str, Enum):
    BRIEF = "brief"          # 一句话总结
    DETAILED = "detailed"    # 逐项说明
    TRADEOFF = "tradeoff"    # 为什么不选其他方案
    ALERT = "alert"          # 风险/注意事项


class FeedbackSignal(str, Enum):
    EXPLICIT_POSITIVE = "explicit_positive"   # 加购/下单
    EXPLICIT_NEGATIVE = "explicit_negative"   # 明确拒绝
    IMPLICIT_POSITIVE = "implicit_positive"   # 长时间停留
    IMPLICIT_NEGATIVE = "implicit_negative"   # 快速跳过
    REVISION = "revision"                     # 主动修改约束
    TASK_COMPLETE = "task_complete"           # 成功下单
    TASK_ABANDON = "task_abandon"             # 放弃任务


# ---------------------------------------------------------------------------
# ShoppingTask — 意图解析结果，所有模块的输入契约
# ---------------------------------------------------------------------------

@dataclass
class Constraint:
    """单条约束，含严重级别与来源。"""
    key: str                          # e.g. "budget_total", "delivery_days"
    value: Any                        # e.g. 8000, 2, "Sony"
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    source: str = "user"              # "user" | "inferred" | "history"


@dataclass
class ConflictPair:
    """两条约束之间的冲突描述。"""
    constraint_a: str
    constraint_b: str
    description: str                  # 人类可读的冲突说明
    resolution: ConflictResolution = ConflictResolution.ASK_USER


@dataclass
class TaskRevision:
    """用户每次修正任务后记录的 diff。"""
    round_index: int
    changed_fields: dict[str, Any]   # {"budget_total": {"from": 3000, "to": 5000}}
    user_utterance: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ShoppingTask:
    """
    结构化购物任务。由 IntentParser 生成，贯穿全流程。
    """
    task_id: str
    task_type: TaskType

    # 品类与核心需求
    categories: list[str]            # e.g. ["laptop", "monitor", "keyboard"]
    raw_query: str                   # 原始用户输入

    # 约束体系
    constraints: list[Constraint] = field(default_factory=list)
    implicit_needs: list[str] = field(default_factory=list)
    # e.g. ["出差用" → 轻便、防摔]

    # 冲突管理
    conflict_pairs: list[ConflictPair] = field(default_factory=list)

    # 不确定性状态
    uncertainty_slots: dict[str, Optional[Any]] = field(default_factory=dict)
    # e.g. {"size": None, "color": "uncertain"}
    uncertainty_score: float = 1.0   # 0.0~1.0，越高越需要澄清

    # 需求演化日志
    revision_history: list[TaskRevision] = field(default_factory=list)

    # 元信息
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_hard_constraints(self) -> dict[str, Any]:
        return {c.key: c.value for c in self.constraints
                if c.severity == ConstraintSeverity.HARD}

    def get_soft_preferences(self) -> dict[str, Any]:
        return {c.key: c.value for c in self.constraints
                if c.severity == ConstraintSeverity.SOFT}

    def has_conflict(self) -> bool:
        return len(self.conflict_pairs) > 0

    def apply_revision(self, revision: TaskRevision) -> None:
        self.revision_history.append(revision)
        self.updated_at = datetime.now()


# ---------------------------------------------------------------------------
# UserProfile — 用户长期偏好画像
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    user_id: str

    # 品牌偏好权重，e.g. {"Sony": 0.9, "Huawei": 0.7}
    brand_weights: dict[str, float] = field(default_factory=dict)

    # 价格敏感度 0.0（不敏感）~ 1.0（极度敏感）
    price_sensitivity: float = 0.5

    # 平台偏好，e.g. {"JD": 0.8, "Tmall": 0.6}
    platform_preferences: dict[str, float] = field(default_factory=dict)

    # 尺码画像，e.g. {"clothes_top": "M", "shoes": "42"}
    size_profile: dict[str, str] = field(default_factory=dict)

    # 历史购买品类偏好分布
    category_affinity: dict[str, float] = field(default_factory=dict)

    # 风格标签，e.g. ["简约", "商务", "性价比"]
    style_tags: list[str] = field(default_factory=list)

    # 交互记忆摘要
    interaction_signal_counts: dict[str, int] = field(default_factory=dict)
    budget_anchor_history: list[float] = field(default_factory=list)

    # Persona state（最小可用版本）
    identity_goal: dict[str, float] = field(default_factory=dict)
    budget_sensitivity_profile: dict[str, float] = field(default_factory=dict)
    brand_orientation: dict[str, float] = field(default_factory=dict)
    aesthetic_preference: dict[str, float] = field(default_factory=dict)
    persona_stability: float = 0.5
    recent_persona_drift: dict[str, Any] = field(default_factory=dict)
    persona_transition_log: list[dict[str, Any]] = field(default_factory=list)

    # Long-horizon profile
    owned_items: list[dict[str, Any]] = field(default_factory=list)
    active_setups: dict[str, dict[str, Any]] = field(default_factory=dict)
    upgrade_stage: dict[str, str] = field(default_factory=dict)
    purchase_rhythm: dict[str, Any] = field(default_factory=dict)
    aspiration_signals: list[dict[str, Any]] = field(default_factory=list)

    # 履约偏好
    prefer_fast_delivery: bool = False
    prefer_official_store: bool = True

    last_updated: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Product — 标准化商品对象
# ---------------------------------------------------------------------------

@dataclass
class ProductAttribute:
    name: str
    value: Any
    unit: Optional[str] = None       # e.g. "g", "inch", "W"


@dataclass
class LogisticsInfo:
    platform: str
    delivery_days: Optional[int] = None
    delivery_fee: float = 0.0
    supports_return: bool = True
    return_days: int = 7
    is_official_store: bool = False


@dataclass
class Product:
    """
    标准化商品对象，跨平台对齐后的统一格式。
    """
    product_id: str
    title: str
    platform: str                    # "JD" | "Tmall" | "PDD" | ...

    # 价格
    price: float
    original_price: Optional[float] = None
    coupon_discount: float = 0.0

    @property
    def final_price(self) -> float:
        return self.price - self.coupon_discount

    # 品类与属性
    category: str = ""
    brand: str = ""
    attributes: list[ProductAttribute] = field(default_factory=list)

    # 库存与物流
    in_stock: bool = True
    stock_count: Optional[int] = None
    logistics: Optional[LogisticsInfo] = None

    # 质量信号
    rating: Optional[float] = None   # 0~5
    review_count: int = 0
    sales_volume: int = 0

    # 数据时效
    fetched_at: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 300           # 默认 5 分钟 TTL

    # 来源
    source_url: Optional[str] = None
    image_url: Optional[str] = None

    # Persona-facing tags
    persona_tags: dict[str, Any] = field(default_factory=dict)

    def is_fresh(self) -> bool:
        elapsed = (datetime.now() - self.fetched_at).total_seconds()
        return elapsed < self.ttl_seconds

    def get_attribute(self, name: str) -> Optional[Any]:
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None


# ---------------------------------------------------------------------------
# CandidatePlan / BundlePlan — 规划器输出的候选方案
# ---------------------------------------------------------------------------

@dataclass
class PlanItem:
    """方案中的单个商品选择。"""
    bundle_slot: str                 # 对应 ShoppingTask 中的哪个品类坑位
    product: Product
    reason: str                      # 为什么选这个
    persona_reason: str = ""
    alternatives: list[Product] = field(default_factory=list)


@dataclass
class TradeoffNote:
    """方案的 trade-off 说明。"""
    dimension: str                   # e.g. "price", "performance", "delivery"
    note: str
    severity: str = "info"           # "info" | "warning"


@dataclass
class CandidatePlan:
    """
    一套候选购物方案，包含所有选中商品、总价、得分与说明。
    """
    plan_id: str
    task_id: str
    items: list[PlanItem]

    # 财务汇总
    total_price: float = 0.0
    total_discount: float = 0.0

    @property
    def net_price(self) -> float:
        return self.total_price - self.total_discount

    # 评分体系
    constraint_score: float = 0.0    # 约束满足度 0~1
    preference_score: float = 0.0    # 偏好匹配度 0~1
    value_score: float = 0.0         # 性价比 0~1
    persona_alignment_score: float = 0.0  # 用户画像匹配度 0~1
    style_coherence_score: float = 0.0
    scenario_fit_score: float = 0.0
    bundle_completeness_score: float = 0.0
    long_term_fit_score: float = 0.0
    phased_purchase_score: float = 0.0

    @property
    def overall_score(self) -> float:
        return (
            self.constraint_score * 0.27
            + self.preference_score * 0.15
            + self.value_score * 0.1
            + self.persona_alignment_score * 0.14
            + self.style_coherence_score * 0.09
            + self.scenario_fit_score * 0.08
            + self.bundle_completeness_score * 0.06
            + self.long_term_fit_score * 0.06
            + self.phased_purchase_score * 0.05
        )

    # 方案说明
    tradeoff_notes: list[TradeoffNote] = field(default_factory=list)
    explanation: str = ""
    persona_summary: str = ""
    bundle_type: str = "single_item"
    bundle_objective: str = ""
    budget_allocation: dict[str, float] = field(default_factory=dict)
    phased_purchase_options: list[dict[str, Any]] = field(default_factory=list)

    # 校验状态（由 Verifier 填写）
    verification_status: VerificationStatus = VerificationStatus.PASS
    verification_issues: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BundlePlan(CandidatePlan):
    """
    组合级规划对象。

    为兼容历史接口，BundlePlan 继承 CandidatePlan；
    旧模块仍可按 CandidatePlan 消费，新链路可显式依赖 BundlePlan。
    """
    required_slots: list[str] = field(default_factory=list)
    filled_slots: list[str] = field(default_factory=list)
    slot_coverage: dict[str, bool] = field(default_factory=dict)
    compatibility_score: float = 0.0
    relation_coverage_score: float = 0.0
    bundle_decision_score: float = 0.0
    phased_upgrade_plan: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_bundle_native(self) -> bool:
        return True

    @property
    def overall_score(self) -> float:
        if self.bundle_decision_score > 0:
            return self.bundle_decision_score
        return super().overall_score


@dataclass
class PlanArtifact:
    artifact_id: str
    artifact_type: str              # snapshot | bundle_recommendation | phase_plan | tradeoff
    title: str
    summary: str = ""
    content: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanWorkspace:
    workspace_id: str
    title: str
    status: str = "draft"           # draft | active | accepted | completed | archived
    objective: str = ""
    lifecycle_stage: str = "current_recommendation"
    updated_at: datetime = field(default_factory=datetime.now)
    artifacts: list[PlanArtifact] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ClarificationQuestion — 澄清问题
# ---------------------------------------------------------------------------

@dataclass
class ClarificationQuestion:
    slot: str                        # 对应 uncertainty_slots 中的哪个槽位
    question: str                    # 向用户展示的问题文本
    style: ClarificationStyle
    options: list[str] = field(default_factory=list)  # 仅 CHOICE 时有值
    info_gain: float = 0.0           # 预期信息增益
    impact_score: float = 0.0        # 对方案的影响程度


# ---------------------------------------------------------------------------
# VerificationReport — 校验报告
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    checker_name: str
    status: VerificationStatus
    message: str
    severity: str = "info"           # "info" | "warning" | "error"
    fix_suggestion: Optional[str] = None


@dataclass
class VerificationReport:
    plan_id: str
    passed: bool
    results: list[CheckResult]

    @property
    def blocking_issues(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == VerificationStatus.BLOCK]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == VerificationStatus.WARN]

    @property
    def stale_items(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == VerificationStatus.STALE]


# ---------------------------------------------------------------------------
# FeedbackRecord — 用户行为反馈
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    session_id: str
    task_id: str
    plan_id: Optional[str]
    signal: FeedbackSignal
    context: dict[str, Any] = field(default_factory=dict)
    # 归因追踪：记录是哪个模块/决策导致了这个结果
    attribution_trace: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
