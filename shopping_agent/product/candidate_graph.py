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
from typing import Any

from shopping_agent.common.constants import MAX_CANDIDATES_PER_SLOT
from shopping_agent.common.types import Product, ShoppingTask


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
