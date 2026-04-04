"""
VerificationPipeline — 五重并行校验管道。

校验器（独立可测试）：
  1. BudgetChecker      预算校验
  2. ConstraintChecker  硬约束满足度校验
  3. CompatibilityChecker 兼容性校验（基于候选图的 INCOMPATIBLE 边）
  4. FreshnessChecker   数据时效校验
  5. RiskChecker        高风险动作校验

结果分三档：
  PASS  方案可用
  WARN  方案可用，但需告知用户
  BLOCK 方案不可用，必须重规划
  STALE 数据过期，需重新拉取后重新校验
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from shopping_agent.common.constants import HIGH_RISK_AMOUNT_THRESHOLD
from shopping_agent.common.types import (
    CandidatePlan,
    CheckResult,
    ShoppingTask,
    VerificationReport,
    VerificationStatus,
)


class BaseChecker:
    name: str = "BaseChecker"

    def check(self, plan: CandidatePlan, task: ShoppingTask) -> CheckResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. 预算校验
# ---------------------------------------------------------------------------

class BudgetChecker(BaseChecker):
    name = "BudgetChecker"

    def check(self, plan: CandidatePlan, task: ShoppingTask) -> CheckResult:
        budget = task.get_hard_constraints().get("budget_total")
        if budget is None:
            return CheckResult(
                checker_name=self.name,
                status=VerificationStatus.PASS,
                message="无预算约束，跳过预算校验。",
            )

        net = plan.net_price
        if net <= budget:
            return CheckResult(
                checker_name=self.name,
                status=VerificationStatus.PASS,
                message=f"方案总价 ¥{net:.2f} 在预算 ¥{budget:.2f} 以内。",
            )
        elif net <= budget * 1.05:
            # 超出 5% 以内：警告
            return CheckResult(
                checker_name=self.name,
                status=VerificationStatus.WARN,
                message=f"方案总价 ¥{net:.2f} 略超预算 ¥{budget:.2f}（超出 {(net/budget-1)*100:.1f}%）。",
                severity="warning",
                fix_suggestion="可以尝试将某个品类换为更低价的选择。",
            )
        else:
            return CheckResult(
                checker_name=self.name,
                status=VerificationStatus.BLOCK,
                message=f"方案总价 ¥{net:.2f} 超出预算 ¥{budget:.2f}（超出 {(net/budget-1)*100:.1f}%）。",
                severity="error",
                fix_suggestion="请放宽预算，或减少品类数量。",
            )


# ---------------------------------------------------------------------------
# 2. 硬约束校验
# ---------------------------------------------------------------------------

class ConstraintChecker(BaseChecker):
    name = "ConstraintChecker"

    def check(self, plan: CandidatePlan, task: ShoppingTask) -> CheckResult:
        hard = task.get_hard_constraints()
        violations = []

        for item in plan.items:
            p = item.product

            # 库存校验
            if not p.in_stock:
                violations.append(f"{p.title[:20]} 已无货")

            # 履约时效校验
            max_days = hard.get("delivery_days")
            if (max_days and p.logistics and
                    p.logistics.delivery_days and
                    p.logistics.delivery_days > max_days):
                violations.append(
                    f"{p.title[:20]} 预计 {p.logistics.delivery_days} 天到货，"
                    f"超过要求的 {max_days} 天"
                )

            # 必须官方店校验（如果有此要求）
            if (hard.get("official_store_only") and
                    p.logistics and not p.logistics.is_official_store):
                violations.append(f"{p.title[:20]} 非官方店铺")

        if not violations:
            return CheckResult(
                checker_name=self.name,
                status=VerificationStatus.PASS,
                message="所有硬约束均已满足。",
            )

        return CheckResult(
            checker_name=self.name,
            status=VerificationStatus.BLOCK,
            message=f"硬约束违反：{'; '.join(violations)}",
            severity="error",
            fix_suggestion="请检查库存情况或调整履约时效要求。",
        )


# ---------------------------------------------------------------------------
# 3. 兼容性校验
# ---------------------------------------------------------------------------

class CompatibilityChecker(BaseChecker):
    name = "CompatibilityChecker"

    def check(self, plan: CandidatePlan, task: ShoppingTask) -> CheckResult:
        # 检查方案内商品品类组合是否有已知不兼容
        categories = [item.product.category.lower() for item in plan.items]

        from shopping_agent.product.candidate_graph import _INCOMPATIBLE_RULES
        incompatible_found = []

        for rule_a, rule_b, reason in _INCOMPATIBLE_RULES:
            has_a = any(rule_a in cat for cat in categories)
            has_b = any(rule_b in cat for cat in categories)
            if has_a and has_b:
                incompatible_found.append(reason)

        if not incompatible_found:
            return CheckResult(
                checker_name=self.name,
                status=VerificationStatus.PASS,
                message="方案内商品组合兼容性检查通过。",
            )

        return CheckResult(
            checker_name=self.name,
            status=VerificationStatus.BLOCK,
            message=f"兼容性问题：{'; '.join(incompatible_found)}",
            severity="error",
            fix_suggestion="请选择兼容的硬件组合。",
        )


# ---------------------------------------------------------------------------
# 4. 数据时效校验
# ---------------------------------------------------------------------------

class FreshnessChecker(BaseChecker):
    name = "FreshnessChecker"

    def check(self, plan: CandidatePlan, task: ShoppingTask) -> CheckResult:
        stale_items = []
        for item in plan.items:
            if not item.product.is_fresh():
                stale_items.append(item.product.title[:20])

        if not stale_items:
            return CheckResult(
                checker_name=self.name,
                status=VerificationStatus.PASS,
                message="所有商品数据均在有效期内。",
            )

        return CheckResult(
            checker_name=self.name,
            status=VerificationStatus.STALE,
            message=f"以下商品数据已过期，需重新拉取：{', '.join(stale_items)}",
            severity="warning",
            fix_suggestion="系统将自动刷新商品价格和库存。",
        )


# ---------------------------------------------------------------------------
# 5. 风险控制校验
# ---------------------------------------------------------------------------

class RiskChecker(BaseChecker):
    name = "RiskChecker"

    def check(self, plan: CandidatePlan, task: ShoppingTask) -> CheckResult:
        warnings = []

        # 高金额预警
        if plan.net_price > HIGH_RISK_AMOUNT_THRESHOLD:
            warnings.append(
                f"订单金额 ¥{plan.net_price:.2f} 较高，请确认后再下单。"
            )

        # 低库存预警
        for item in plan.items:
            if item.product.stock_count and item.product.stock_count < 3:
                warnings.append(
                    f"{item.product.title[:20]} 库存仅剩 {item.product.stock_count} 件，手慢无。"
                )

        # 低评分预警
        for item in plan.items:
            if item.product.rating and item.product.rating < 4.0:
                warnings.append(
                    f"{item.product.title[:20]} 评分较低（{item.product.rating}），建议谨慎。"
                )

        if not warnings:
            return CheckResult(
                checker_name=self.name,
                status=VerificationStatus.PASS,
                message="无风险提示。",
            )

        return CheckResult(
            checker_name=self.name,
            status=VerificationStatus.WARN,
            message="\n".join(warnings),
            severity="warning",
        )


# ---------------------------------------------------------------------------
# 主校验管道
# ---------------------------------------------------------------------------

class VerificationPipeline:
    def __init__(self):
        self._checkers: list[BaseChecker] = [
            BudgetChecker(),
            ConstraintChecker(),
            CompatibilityChecker(),
            FreshnessChecker(),
            RiskChecker(),
        ]

    def verify(
        self,
        plan: CandidatePlan,
        task: ShoppingTask,
        parallel: bool = True,
    ) -> VerificationReport:
        """
        并行执行所有校验器，汇总报告。
        """
        results: list[CheckResult] = []

        if parallel:
            with ThreadPoolExecutor(max_workers=len(self._checkers)) as executor:
                futures = {
                    executor.submit(checker.check, plan, task): checker
                    for checker in self._checkers
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        checker = futures[future]
                        results.append(CheckResult(
                            checker_name=checker.name,
                            status=VerificationStatus.WARN,
                            message=f"校验器异常: {e}",
                            severity="warning",
                        ))
        else:
            for checker in self._checkers:
                results.append(checker.check(plan, task))

        passed = not any(r.status == VerificationStatus.BLOCK for r in results)

        # 把校验问题写入 plan
        plan.verification_status = (
            VerificationStatus.PASS if passed else VerificationStatus.BLOCK
        )
        plan.verification_issues = [
            r.message for r in results
            if r.status in (VerificationStatus.BLOCK, VerificationStatus.WARN)
        ]

        return VerificationReport(
            plan_id=plan.plan_id,
            passed=passed,
            results=results,
        )
