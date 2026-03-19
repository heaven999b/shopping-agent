"""
CandidateGraphBuilder — 候选商品图构建器。

图定义：
  节点类型：PRODUCT / CATEGORY / BUNDLE_SLOT / ATTRIBUTE
  边类型：
    SUBSTITUTE    替代关系（同类竞品）
    COMPLEMENT    互补关系（组合常一起买）
    COMPATIBLE    兼容关系（技术规格可配合）
    INCOMPATIBLE  不兼容（必须拦截）
    BELONGS_TO    SKU 属于品类
    HAS_ATTR      SKU 拥有属性

规模控制：每个 BUNDLE_SLOT 最多保留 MAX_CANDIDATES_PER_SLOT 个候选。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from shopping_agent.common.constants import MAX_CANDIDATES_PER_SLOT
from shopping_agent.common.types import BundlePlan, CandidatePlan, PlanItem, Product, ShoppingTask


# 预定义的常见互补关系（生产环境从共购数据中挖掘）
_COMPLEMENT_RULES: list[tuple[str, str]] = [
    ("monitor", "keyboard"),
    ("monitor", "mouse"),
    ("keyboard", "mouse"),
    ("laptop", "mouse"),
    ("laptop", "bag"),
    ("cpu", "motherboard"),
    ("cpu", "cooling"),
    ("gpu", "psu"),
]

# 预定义的不兼容规则（生产环境从知识图谱中加载）
_INCOMPATIBLE_RULES: list[tuple[str, str, str]] = [
    # (category_a, category_b, reason)
    ("amd_cpu", "intel_motherboard", "平台不兼容：AMD CPU 需要 AM4/AM5 主板"),
    ("intel_cpu", "amd_motherboard", "平台不兼容：Intel CPU 需要 LGA1700/1200 主板"),
    ("ddr4_ram", "ddr5_motherboard", "内存代际不兼容：DDR4 内存不能插 DDR5 主板"),
]


class CandidateGraphBuilder:
    def build(
        self,
        task: ShoppingTask,
        products: list[Product],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        构建候选图，返回邻接表表示。

        结构：
          {
            "product_id": [
              {"target_id": "...", "edge_type": "SUBSTITUTE", "weight": 0.8, "reason": "..."},
              ...
            ]
          }
        """
        graph: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # 按品类分组，限制每个坑位的候选数量
        category_buckets = self._bucket_by_category(products)
        trimmed_products = []
        for category, bucket in category_buckets.items():
            # 按评分+销量排序，保留 Top-N
            sorted_bucket = sorted(
                bucket,
                key=lambda p: (p.rating or 0) * 0.6 + min(p.sales_volume / 10000, 1.0) * 0.4,
                reverse=True,
            )
            trimmed_products.extend(sorted_bucket[:MAX_CANDIDATES_PER_SLOT])

        # 建立 SUBSTITUTE 边（同品类商品互为替代）
        self._add_substitute_edges(graph, category_buckets)

        # 建立 COMPLEMENT 边（跨品类互补）
        self._add_complement_edges(graph, category_buckets)

        # 建立 INCOMPATIBLE 边（不兼容检测）
        self._add_incompatible_edges(graph, trimmed_products)

        return dict(graph)

    def _bucket_by_category(
        self, products: list[Product]
    ) -> dict[str, list[Product]]:
        buckets: dict[str, list[Product]] = defaultdict(list)
        for p in products:
            buckets[p.category].append(p)
        return buckets

    def _add_substitute_edges(
        self,
        graph: dict[str, list[dict[str, Any]]],
        buckets: dict[str, list[Product]],
    ) -> None:
        for category, products in buckets.items():
            top_n = products[:MAX_CANDIDATES_PER_SLOT]
            for i, p_a in enumerate(top_n):
                for p_b in top_n[i + 1:]:
                    similarity = self._calc_similarity(p_a, p_b)
                    edge = {
                        "target_id": p_b.product_id,
                        "edge_type": "SUBSTITUTE",
                        "weight": similarity,
                        "reason": f"同为{category}品类，可互相替代",
                    }
                    graph[p_a.product_id].append(edge)
                    # 无向边
                    graph[p_b.product_id].append({
                        **edge,
                        "target_id": p_a.product_id,
                    })

    def _add_complement_edges(
        self,
        graph: dict[str, list[dict[str, Any]]],
        buckets: dict[str, list[Product]],
    ) -> None:
        for cat_a, cat_b in _COMPLEMENT_RULES:
            if cat_a not in buckets or cat_b not in buckets:
                continue
            # 取每个品类的 Top-3 建立互补关系
            for p_a in buckets[cat_a][:3]:
                for p_b in buckets[cat_b][:3]:
                    graph[p_a.product_id].append({
                        "target_id": p_b.product_id,
                        "edge_type": "COMPLEMENT",
                        "weight": 0.7,
                        "reason": f"{cat_a} 和 {cat_b} 常一起购买",
                    })

    def _add_incompatible_edges(
        self,
        graph: dict[str, list[dict[str, Any]]],
        products: list[Product],
    ) -> None:
        """
        检测不兼容组合并标记。
        当前基于品类关键词匹配，生产环境应接入规格知识图谱。
        """
        for p_a in products:
            for p_b in products:
                if p_a.product_id == p_b.product_id:
                    continue
                reason = self._check_incompatible(p_a, p_b)
                if reason:
                    graph[p_a.product_id].append({
                        "target_id": p_b.product_id,
                        "edge_type": "INCOMPATIBLE",
                        "weight": 1.0,
                        "reason": reason,
                    })

    def _check_incompatible(self, p_a: Product, p_b: Product) -> str:
        """返回不兼容原因，无问题返回空字符串。"""
        cat_a = p_a.category.lower()
        cat_b = p_b.category.lower()
        for rule_a, rule_b, reason in _INCOMPATIBLE_RULES:
            if rule_a in cat_a and rule_b in cat_b:
                return reason
        return ""

    def _calc_similarity(self, p_a: Product, p_b: Product) -> float:
        """简单计算两个商品的相似度（0~1）。"""
        score = 0.0
        # 同品牌加分
        if p_a.brand == p_b.brand:
            score += 0.3
        # 价格接近加分
        if p_a.price > 0 and p_b.price > 0:
            price_ratio = min(p_a.price, p_b.price) / max(p_a.price, p_b.price)
            score += price_ratio * 0.4
        # 同平台加分
        if p_a.platform == p_b.platform:
            score += 0.1
        return min(1.0, score + 0.2)  # 基础相似度 0.2


# ---------------------------------------------------------------------------
# Graph-Based Recovery — 基于候选图的方案修复器
# ---------------------------------------------------------------------------

class GraphBasedRecovery:
    """
    当校验器阻断方案时，通过候选图中的 SUBSTITUTE 边寻找替代商品，
    自动修复方案使其通过校验。

    修复策略（按优先级）：
      1. BUDGET_BLOCK  → 在方案中找最贵的商品，沿 SUBSTITUTE 边找更便宜的替代品
      2. CONSTRAINT_BLOCK → 找违反约束的商品，沿 SUBSTITUTE 边找满足约束的替代品
      3. STALE         → 跳过（数据问题，不在此修复）

    每次最多替换 max_swaps 个商品，避免无限修复。
    """

    def __init__(self, max_swaps: int = 2):
        self.max_swaps = max_swaps

    def try_repair(
        self,
        plan: CandidatePlan,
        task: ShoppingTask,
        graph: dict[str, list[dict[str, Any]]],
        product_index: dict[str, Product],
        block_reason: str,
    ) -> Optional[CandidatePlan]:
        """
        尝试修复阻断的方案，返回修复后的新方案（原方案不变）。
        若无法修复则返回 None。

        参数：
          plan          — 待修复的方案
          task          — 当前任务（含硬约束）
          graph         — 候选图邻接表 {product_id: [edge_dicts]}
          product_index — product_id → Product 的索引
          block_reason  — BLOCK 原因关键词（"budget"/"constraint"/"stock"）
        """
        import uuid
        from shopping_agent.common.types import TradeoffNote

        items = list(plan.items)  # shallow copy for mutation
        swaps_done = 0
        hard = task.get_hard_constraints()
        budget = hard.get("budget_total")

        for attempt in range(self.max_swaps):
            current_total = sum(item.product.final_price for item in items)

            # 找需要替换的商品
            target_item_idx = self._find_item_to_replace(
                items, block_reason, current_total, budget, hard
            )
            if target_item_idx is None:
                break

            target_item = items[target_item_idx]
            old_product = target_item.product

            # 在 SUBSTITUTE 边中找替代品
            substitute = self._find_substitute(
                old_product, graph, product_index, block_reason,
                budget_remaining=budget - (current_total - old_product.final_price) if budget else None,
                hard_constraints=hard,
            )
            if substitute is None:
                break

            # 执行替换
            new_item = PlanItem(
                bundle_slot=target_item.bundle_slot,
                product=substitute,
                reason=f"图修复: 由 {old_product.title[:15]} 替换为 {substitute.title[:15]}（{block_reason}约束修复）",
                alternatives=target_item.alternatives,
            )
            items[target_item_idx] = new_item
            swaps_done += 1

        if swaps_done == 0:
            return None

        # 重新计算方案分数
        new_total = sum(item.product.final_price for item in items)
        constraint_score = 1.0 if (budget is None or new_total <= budget) else max(0.0, 1.0 - (new_total - budget) / budget)

        common_kwargs = dict(
            plan_id=str(uuid.uuid4()),
            task_id=plan.task_id,
            items=items,
            total_price=round(new_total, 2),
            total_discount=round(sum(i.product.coupon_discount for i in items), 2),
            constraint_score=round(constraint_score, 3),
            preference_score=round(plan.preference_score * 0.9, 3),  # 替换品略降偏好分
            value_score=round(plan.value_score, 3),
            tradeoff_notes=[TradeoffNote(
                dimension="recovery",
                note=f"已自动替换 {swaps_done} 件商品以满足{block_reason}约束",
                severity="info",
            )],
            explanation=f"[图修复方案] 在原方案基础上替换了 {swaps_done} 件商品。\n{plan.explanation}",
        )
        if isinstance(plan, BundlePlan):
            return BundlePlan(
                **common_kwargs,
                persona_alignment_score=plan.persona_alignment_score,
                style_coherence_score=plan.style_coherence_score,
                scenario_fit_score=plan.scenario_fit_score,
                bundle_completeness_score=plan.bundle_completeness_score,
                long_term_fit_score=plan.long_term_fit_score,
                phased_purchase_score=plan.phased_purchase_score,
                persona_summary=plan.persona_summary,
                bundle_type=plan.bundle_type,
                bundle_objective=plan.bundle_objective,
                budget_allocation=plan.budget_allocation,
                phased_purchase_options=plan.phased_purchase_options,
                slot_coverage=plan.slot_coverage,
                compatibility_score=plan.compatibility_score,
                phased_upgrade_plan=plan.phased_upgrade_plan,
            )
        return CandidatePlan(**common_kwargs)

    def build_product_index(self, products: list[Product]) -> dict[str, Product]:
        """从商品列表构建 product_id → Product 索引。"""
        return {p.product_id: p for p in products}

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _find_item_to_replace(
        self,
        items: list[PlanItem],
        block_reason: str,
        current_total: float,
        budget: Optional[float],
        hard: dict[str, Any],
    ) -> Optional[int]:
        """找出最需要替换的商品 index。"""
        if "budget" in block_reason.lower() and budget and current_total > budget:
            # 找最贵的商品
            return max(range(len(items)), key=lambda i: items[i].product.final_price)

        if "stock" in block_reason.lower() or "库存" in block_reason:
            # 找无货商品
            for i, item in enumerate(items):
                if not item.product.in_stock:
                    return i

        if "delivery" in block_reason.lower() or "配送" in block_reason:
            max_days = hard.get("delivery_days")
            if max_days:
                for i, item in enumerate(items):
                    if (item.product.logistics and
                            item.product.logistics.delivery_days and
                            item.product.logistics.delivery_days > max_days):
                        return i

        # 通用：找最贵的
        return max(range(len(items)), key=lambda i: items[i].product.final_price)

    def _find_substitute(
        self,
        product: Product,
        graph: dict[str, list[dict[str, Any]]],
        product_index: dict[str, Product],
        block_reason: str,
        budget_remaining: Optional[float],
        hard_constraints: dict[str, Any],
    ) -> Optional[Product]:
        """
        沿 SUBSTITUTE 边找满足条件的替代品。
        """
        edges = graph.get(product.product_id, [])
        substitute_edges = [e for e in edges if e["edge_type"] == "SUBSTITUTE"]

        # 按相似度权重降序排列
        substitute_edges.sort(key=lambda e: e.get("weight", 0), reverse=True)

        for edge in substitute_edges:
            candidate = product_index.get(edge["target_id"])
            if candidate is None or candidate.product_id == product.product_id:
                continue

            # 预算约束
            if budget_remaining is not None and candidate.final_price > budget_remaining:
                continue

            # 库存约束
            if not candidate.in_stock:
                continue

            # 配送约束
            max_days = hard_constraints.get("delivery_days")
            if (max_days and candidate.logistics and
                    candidate.logistics.delivery_days and
                    candidate.logistics.delivery_days > max_days):
                continue

            # 对 budget block，必须比原来便宜
            if "budget" in block_reason.lower() and candidate.final_price >= product.final_price:
                continue

            return candidate

        return None
