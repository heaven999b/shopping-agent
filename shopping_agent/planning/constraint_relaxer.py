"""
ConstraintRelaxer — 约束松弛与用户协商模块。

背景：
  在 InfeasibleConstraintError 或 VerificationStatus.BLOCK 时，
  直接告知用户"找不到商品"体验极差。
  工业级系统会自动分析不可行原因，并提出有理有据的松弛方案。

功能：
  1. 诊断不可行约束（哪个约束导致了零候选？）
  2. 生成排序好的松弛选项（按用户代价从小到大）
  3. 生成自然语言解释（用于向用户呈现）
  4. 与 Orchestrator 集成：在规划失败后自动触发

松弛策略（按优先级）：
  RELAX_SOFT    — 放弃软约束（成本最低）
  RELAX_BUDGET  — 适度提升预算上限（+10%, +20%, +30%）
  DROP_ATTR     — 放弃非关键硬属性（如 noise_cancelling → wireless 仍保留）
  WIDEN_CAT     — 扩展品类范围（如 "旗舰耳机" → 包含中端耳机）
  CHANGE_PLAT   — 换平台（JD → Tmall，可能有不同货源）

理论依据：
  Faltings et al. (2003) "Hierarchical Constraint Satisfaction"
  Yang et al. (2020) "Query Reformulation via Constraint Relaxation"
  本项目：基于 Verifier 的 blocking_issues 自动触发松弛流程
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from shopping_agent.common.types import (
    Constraint,
    ConstraintSeverity,
    ShoppingTask,
    VerificationReport,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# 松弛操作类型
# ---------------------------------------------------------------------------

class RelaxationType(str, Enum):
    RELAX_SOFT   = "relax_soft"    # 放弃软约束
    RELAX_BUDGET = "relax_budget"  # 提升预算上限
    DROP_ATTR    = "drop_attr"     # 放弃某个属性约束
    WIDEN_CAT    = "widen_cat"     # 扩宽品类
    CHANGE_PLAT  = "change_plat"   # 换平台


@dataclass
class RelaxationOption:
    """一条候选松弛方案。"""
    relaxation_type: RelaxationType
    description: str              # 人类可读的方案描述
    user_cost: float              # 对用户的代价估计 (0=无代价, 1=高代价)
    modified_constraints: dict[str, Any]  # 松弛后的约束变更
    explanation: str              # 向用户呈现的解释
    confidence: float = 1.0      # 该松弛能解决问题的置信度

    def __lt__(self, other: "RelaxationOption") -> bool:
        return self.user_cost < other.user_cost


@dataclass
class RelaxationResult:
    """
    约束松弛诊断结果。

    包含：
    - 不可行原因列表
    - 排序好的松弛方案
    - 推荐方案（user_cost 最低且 confidence 足够高）
    """
    infeasible_reasons: list[str] = field(default_factory=list)
    options: list[RelaxationOption] = field(default_factory=list)
    recommended: Optional[RelaxationOption] = None
    can_auto_relax: bool = False  # 是否可以不询问用户直接松弛（仅软约束）

    def user_message(self) -> str:
        """生成向用户呈现的诊断信息。"""
        if not self.infeasible_reasons:
            return "当前条件下暂无合适商品。"

        lines = ["抱歉，当前条件下暂无完全匹配的商品："]
        for r in self.infeasible_reasons:
            lines.append(f"  • {r}")

        if self.options:
            lines.append("\n为您推荐以下调整方案：")
            for i, opt in enumerate(self.options[:3], 1):
                lines.append(f"  {i}. {opt.description}")
                lines.append(f"     {opt.explanation}")

        return "\n".join(lines)

    def apply_recommended(self, task: ShoppingTask) -> Optional[ShoppingTask]:
        """
        将推荐的松弛方案应用到任务，返回修改后的新任务。

        不修改原任务（不可变性），返回副本。
        若无推荐方案则返回 None。
        """
        if self.recommended is None:
            return None
        return _apply_relaxation(task, self.recommended)


# ---------------------------------------------------------------------------
# 主松弛器
# ---------------------------------------------------------------------------

class ConstraintRelaxer:
    """
    约束松弛诊断器。

    调用方式：
        relaxer = ConstraintRelaxer()
        result = relaxer.diagnose(task, blocking_report, catalog)
        if result.can_auto_relax:
            relaxed_task = result.apply_recommended(task)
        else:
            print(result.user_message())
    """

    # 预算松弛级别：(倍数, 用户代价, 描述)
    _BUDGET_RELAXATIONS = [
        (1.10, 0.2, "预算上浮 10%"),
        (1.20, 0.4, "预算上浮 20%"),
        (1.30, 0.6, "预算上浮 30%"),
        (1.50, 0.8, "预算上浮 50%"),
    ]

    # 属性重要性（越低越可以优先放弃）
    _ATTR_IMPORTANCE: dict[str, float] = {
        "noise_cancelling": 0.7,
        "wireless": 0.8,
        "mechanical": 0.5,
        "rgb": 0.2,           # 低优先级，优先放弃
        "adjustable_lumbar": 0.6,
        "motorized": 0.7,
        "refresh_rate_hz": 0.6,
        "weight_kg": 0.5,
        "resolution": 0.7,
    }

    def diagnose(
        self,
        task: ShoppingTask,
        report: Optional[VerificationReport] = None,
        product_count_by_constraint: Optional[dict[str, int]] = None,
    ) -> RelaxationResult:
        """
        分析任务约束的不可行原因，生成松弛方案。

        参数：
            task                         — 原始购物任务
            report                       — Verifier 的 VerificationReport（可选）
            product_count_by_constraint  — {constraint_key: 满足该约束的商品数}
                                           用于定位最严格的约束
        返回：
            RelaxationResult
        """
        result = RelaxationResult()
        hard = task.get_hard_constraints()
        soft = task.get_soft_preferences()

        # ── Step 1: 诊断不可行原因 ──
        result.infeasible_reasons = self._diagnose_reasons(
            task, report, product_count_by_constraint
        )

        options: list[RelaxationOption] = []

        # ── Step 2: 软约束松弛（代价最低）──
        if soft:
            for key, value in soft.items():
                opt = RelaxationOption(
                    relaxation_type=RelaxationType.RELAX_SOFT,
                    description=f"放弃软性要求「{_constraint_label(key, value)}」",
                    user_cost=0.1,
                    modified_constraints={"drop_soft": key},
                    explanation=f"该条件不是强制要求，去掉后将有更多选择。",
                    confidence=0.9,
                )
                options.append(opt)

        # ── Step 3: 预算松弛 ──
        budget = hard.get("budget_total")
        if budget and budget > 0:
            for multiplier, cost, label in self._BUDGET_RELAXATIONS:
                new_budget = round(budget * multiplier, -1)  # 取整到十位
                opt = RelaxationOption(
                    relaxation_type=RelaxationType.RELAX_BUDGET,
                    description=f"{label}（预算调整为 ¥{new_budget:.0f}）",
                    user_cost=cost,
                    modified_constraints={"budget_total": new_budget},
                    explanation=(
                        f"当前预算 ¥{budget:.0f} 内的选择较少，"
                        f"调整至 ¥{new_budget:.0f} 可显著增加候选商品。"
                    ),
                    confidence=0.85,
                )
                options.append(opt)

        # ── Step 4: 属性约束放弃（按重要性从低到高）──
        attr_constraints = {
            k: v for k, v in hard.items()
            if k not in {"budget_total", "delivery_days", "brand", "platform"}
            and isinstance(v, bool) and v is True  # 只松弛布尔型 True 约束
        }
        # 按重要性升序排列（先尝试放弃不重要的）
        sorted_attrs = sorted(
            attr_constraints.items(),
            key=lambda kv: self._ATTR_IMPORTANCE.get(kv[0], 0.5),
        )
        for key, value in sorted_attrs[:2]:  # 最多建议 2 个属性放弃
            importance = self._ATTR_IMPORTANCE.get(key, 0.5)
            opt = RelaxationOption(
                relaxation_type=RelaxationType.DROP_ATTR,
                description=f"不强制要求「{_constraint_label(key, value)}」功能",
                user_cost=importance,
                modified_constraints={"drop_hard": key},
                explanation=(
                    f"去掉「{_constraint_label(key, value)}」限制后，"
                    f"可以在同价位找到性价比更高的选择。"
                ),
                confidence=0.8,
            )
            options.append(opt)

        # ── Step 5: 排序 ──
        options.sort(key=lambda o: (o.user_cost, -o.confidence))
        result.options = options

        # ── Step 6: 选推荐方案 ──
        viable = [o for o in options if o.confidence >= 0.7]
        if viable:
            result.recommended = viable[0]

        # ── Step 7: 判断是否可自动松弛（仅软约束）──
        auto_opts = [o for o in options
                     if o.relaxation_type == RelaxationType.RELAX_SOFT
                     and o.user_cost < 0.2]
        result.can_auto_relax = len(auto_opts) > 0

        return result

    def diagnose_from_empty_retrieval(
        self,
        task: ShoppingTask,
        attempted_constraints: dict[str, Any],
    ) -> RelaxationResult:
        """
        检索结果为空时的特化诊断（无需 VerificationReport）。

        attempted_constraints — 检索时使用的完整约束 dict
        """
        result = RelaxationResult()
        result.infeasible_reasons = [
            f"在约束条件（预算={attempted_constraints.get('budget_total')}，"
            f"品类={task.categories}）下未找到任何商品"
        ]
        return self.diagnose(task)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _diagnose_reasons(
        self,
        task: ShoppingTask,
        report: Optional[VerificationReport],
        product_count: Optional[dict[str, int]],
    ) -> list[str]:
        reasons = []
        hard = task.get_hard_constraints()

        # 从 VerificationReport 提取 blocking issues
        if report:
            for issue in report.blocking_issues:
                reasons.append(issue.message)
            return reasons  # verifier 已给出具体原因

        # 无 report 时启发式诊断
        budget = hard.get("budget_total")
        if budget and budget < 200:
            reasons.append(f"预算 ¥{budget:.0f} 过低，大多数商品无法覆盖")

        attr_constraints = {
            k: v for k, v in hard.items()
            if k not in {"budget_total", "delivery_days"}
        }
        if len(attr_constraints) >= 3:
            reasons.append(f"同时满足 {len(attr_constraints)} 项硬性要求（可能条件过严）")

        delivery = hard.get("delivery_days")
        if delivery and delivery <= 1:
            reasons.append(f"要求 {delivery} 天内送达，当前库存可能无法满足")

        if not reasons:
            reasons.append("当前约束组合下候选商品不足")

        return reasons


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _constraint_label(key: str, value: Any) -> str:
    """将约束键值对转为人类可读标签。"""
    labels = {
        "noise_cancelling": "主动降噪",
        "wireless": "无线连接",
        "mechanical": "机械键盘",
        "rgb": "RGB 灯效",
        "adjustable_lumbar": "可调腰托",
        "motorized": "电动升降",
        "refresh_rate_hz": f"刷新率≥{value}Hz",
        "weight_kg": f"重量≤{value}kg",
        "delivery_days": f"{value}天内到货",
        "brand": f"品牌={value}",
        "platform": f"平台={value}",
    }
    return labels.get(key, f"{key}={value}")


def _apply_relaxation(task: ShoppingTask, opt: RelaxationOption) -> ShoppingTask:
    """
    将松弛方案应用到任务，返回新 ShoppingTask（不修改原任务）。
    """
    import copy
    new_task = copy.deepcopy(task)

    changes = opt.modified_constraints

    if "budget_total" in changes:
        # 更新预算约束
        for c in new_task.constraints:
            if c.key == "budget_total":
                c.value = changes["budget_total"]
                break

    if "drop_soft" in changes:
        # 删除某个软约束
        drop_key = changes["drop_soft"]
        new_task.constraints = [
            c for c in new_task.constraints
            if not (c.key == drop_key and c.severity == ConstraintSeverity.SOFT)
        ]

    if "drop_hard" in changes:
        # 将硬约束降级为软约束（保留但不强制）
        drop_key = changes["drop_hard"]
        for c in new_task.constraints:
            if c.key == drop_key and c.severity == ConstraintSeverity.HARD:
                c.severity = ConstraintSeverity.SOFT
                break

    return new_task
