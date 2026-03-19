"""
Explainer — 方案解释生成器。

支持四种解释风格：
  BRIEF    一句话总结（默认首轮）
  DETAILED 逐项说明选择理由
  TRADEOFF 为什么不选其他方案
  ALERT    风险/注意事项提示
"""

from __future__ import annotations

from typing import Optional

from shopping_agent.common.types import (
    CandidatePlan,
    ExplainStyle,
    ShoppingTask,
    VerificationStatus,
)


class Explainer:
    def explain(
        self,
        plan: CandidatePlan,
        task: ShoppingTask,
        alternative_plans: Optional[list[CandidatePlan]] = None,
        style_hint: str = "brief",
    ) -> str:
        """
        生成方案解释文本。

        style_hint: "brief" | "detailed" | "tradeoff" | "alert"
        """
        style = ExplainStyle(style_hint) if style_hint in ExplainStyle._value2member_map_ \
            else ExplainStyle.BRIEF

        parts = []

        # 主推方案摘要
        parts.append(self._build_summary(plan, task))

        # 根据风格添加详细内容
        if style in (ExplainStyle.DETAILED, ExplainStyle.TRADEOFF):
            parts.append(self._build_item_details(plan))

        if style == ExplainStyle.TRADEOFF and alternative_plans:
            parts.append(self._build_tradeoff(plan, alternative_plans))

        # 警告信息（始终添加 BLOCK/WARN）
        warnings = self._build_warnings(plan)
        if warnings:
            parts.append(warnings)

        # Trade-off 注记
        if plan.tradeoff_notes:
            notes = "\n".join(f"  ⚠️ {note.note}" for note in plan.tradeoff_notes)
            parts.append(f"注意事项：\n{notes}")

        # 备选方案提示
        if alternative_plans:
            parts.append(self._build_alternatives_hint(alternative_plans))

        if plan.phased_purchase_options:
            parts.append(self._build_phased_options(plan))

        return "\n\n".join(filter(None, parts))

    def _build_summary(self, plan: CandidatePlan, task: ShoppingTask) -> str:
        items_summary = "、".join(
            f"{item.bundle_slot}（{item.product.title[:15]}...，¥{item.product.final_price:.0f}）"
            for item in plan.items
        )
        summary = (
            f"为您推荐以下方案（综合得分 {plan.overall_score:.0%}）：\n"
            f"{items_summary}\n"
            f"合计：¥{plan.net_price:.2f}"
            + (f"（预算 ¥{task.get_hard_constraints().get('budget_total', '不限')}）"
               if task.get_hard_constraints().get("budget_total") else "")
        )
        if plan.persona_summary:
            summary += (
                f"\n这套方案在用户画像上的贴合度约为 {plan.persona_alignment_score:.0%}，"
                f"{plan.persona_summary}。"
            )
        if plan.bundle_type == "bundle_plan" and plan.budget_allocation:
            allocation = "、".join(
                f"{slot} {ratio:.0%}" for slot, ratio in plan.budget_allocation.items()
            )
            summary += f"\n预算分配：{allocation}。"
        return summary

    def _build_item_details(self, plan: CandidatePlan) -> str:
        lines = ["各品类选择理由："]
        for item in plan.items:
            p = item.product
            reason = item.reason
            if item.persona_reason:
                reason = f"{reason}；画像上：{item.persona_reason}"
            lines.append(
                f"  • [{item.bundle_slot}] {p.title}\n"
                f"    价格：¥{p.final_price:.2f} | 评分：{p.rating} | "
                f"平台：{p.platform} | 理由：{reason}"
            )
        return "\n".join(lines)

    def _build_tradeoff(
        self, plan: CandidatePlan, alternatives: list[CandidatePlan]
    ) -> str:
        if not alternatives:
            return ""
        lines = ["与其他方案对比："]
        for alt in alternatives[:2]:
            price_diff = alt.net_price - plan.net_price
            sign = "+" if price_diff > 0 else ""
            lines.append(
                f"  • 方案 {alt.plan_id[:6]}：总价 ¥{alt.net_price:.2f}（{sign}{price_diff:.0f}元），"
                f"得分 {alt.overall_score:.0%}"
            )
        return "\n".join(lines)

    def _build_warnings(self, plan: CandidatePlan) -> str:
        warnings = []
        for issue in plan.verification_issues:
            warnings.append(f"  ⚠️ {issue}")
        # 库存预警
        for item in plan.items:
            if item.product.stock_count and item.product.stock_count < 5:
                warnings.append(f"  ⚠️ {item.product.title[:20]} 库存仅剩 {item.product.stock_count} 件")
        if warnings:
            return "风险提示：\n" + "\n".join(warnings)
        return ""

    def _build_alternatives_hint(self, alternatives: list[CandidatePlan]) -> str:
        if not alternatives:
            return ""
        hints = [f"省钱方案（¥{alternatives[0].net_price:.2f}）" if alternatives else ""]
        if len(alternatives) > 1:
            hints.append(f"升级方案（¥{alternatives[1].net_price:.2f}）")
        return f"💡 还有 {len(alternatives)} 套备选方案可供参考：{'、'.join(filter(None, hints))}。如需查看请告诉我。"

    def _build_phased_options(self, plan: CandidatePlan) -> str:
        lines = ["分阶段购买建议："]
        for option in plan.phased_purchase_options[:2]:
            if option.get("route") == "分阶段升级":
                lines.append(
                    f"  • 分阶段升级：先买 {', '.join(option.get('phase_1_slots', []))} "
                    f"(约¥{option.get('phase_1_budget', 0):.0f})，"
                    f"后续补 {', '.join(option.get('phase_2_slots', []))} "
                    f"(约¥{option.get('phase_2_budget', 0):.0f})"
                )
            else:
                lines.append(
                    f"  • {option.get('route', '路线')}：{option.get('goal', '')}，"
                    f"预算约¥{option.get('budget', 0):.0f}"
                )
        return "\n".join(lines)
