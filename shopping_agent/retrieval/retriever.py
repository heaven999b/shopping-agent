"""
HybridRetriever — 多通道商品召回器。

四路召回（按优先级）：
  1. 属性精确过滤：按硬约束过滤（预算/时效/必要属性）
  2. 关键词匹配：在 title/brand 上做词袋匹配
  3. TF-IDF 语义向量检索：覆盖关键词遗漏的语义相关商品
  4. 用户画像重排：偏好品牌提权，价格敏感度调整

数据后端：ProductCatalog（加载自 data/products.json）。
生产环境可替换为电商 API，接口不变。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from shopping_agent.common.constants import RETRIEVAL_TOP_K
from shopping_agent.common.exceptions import NoProductFoundError
from shopping_agent.common.types import Product, ShoppingTask, UserProfile
from shopping_agent.data.loader import ProductCatalog, get_catalog
from shopping_agent.retrieval.vector_index import TFIDFVectorIndex

logger = logging.getLogger(__name__)


# 约束键名 → 商品属性名 的映射表
# tasks.json 的 constraints 键名可能和 products.json 的属性名不同
_CONSTRAINT_TO_ATTR: dict[str, str] = {
    "noise_cancelling": "noise_cancelling",
    "wireless": "wireless",
    "mechanical": "mechanical",
    "rgb": "rgb",
    "adjustable_lumbar": "adjustable_lumbar",
    "motorized": "motorized",
    # 范围型约束：{key}_min / {key}_max 会被解析为 attr 的 _min/_max
    "refresh_rate_hz_min": "refresh_rate_hz_min",
    "weight_kg_max": "weight_kg_max",
}

# 不属于商品属性的约束键（预算/时效/品牌等，单独处理）
_NON_ATTR_CONSTRAINT_KEYS = {
    "budget_total",
    "delivery_days",
    "brand",
    "brand_preference",
    "platform",
}


class HybridRetriever:
    """
    多通道召回器。

    参数 catalog 可由外部注入（用于测试 / 自定义数据集），
    默认使用全局单例（自动加载 data/products.json）。
    """

    def __init__(
        self,
        catalog: Optional[ProductCatalog] = None,
        vector_index: Optional[TFIDFVectorIndex] = None,
    ):
        self._catalog = catalog      # 延迟初始化，首次 retrieve 时加载
        self._vector_index = vector_index  # 延迟构建

    @property
    def catalog(self) -> ProductCatalog:
        if self._catalog is None:
            self._catalog = get_catalog()
        return self._catalog

    @property
    def vector_index(self) -> TFIDFVectorIndex:
        """懒构建 TF-IDF 索引（首次访问时从 catalog 构建）。"""
        if self._vector_index is None:
            self._vector_index = TFIDFVectorIndex()
        if not self._vector_index.is_built:
            all_products = self.catalog.search(top_k=9999)
            self._vector_index.build(all_products)
        return self._vector_index

    def retrieve(
        self,
        task: ShoppingTask,
        user_profile: Optional[UserProfile] = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[Product]:
        """
        多通道召回，返回候选商品列表（已去重、已重排）。
        """
        hard = task.get_hard_constraints()
        budget = hard.get("budget_total")
        delivery_days = hard.get("delivery_days")

        # 从约束中提取品牌过滤
        brand_filter: Optional[list[str]] = None
        if "brand" in hard and hard["brand"]:
            brand_filter = [hard["brand"]] if isinstance(hard["brand"], str) else hard["brand"]

        # 从约束中提取商品属性过滤
        required_attrs = self._extract_attr_constraints(hard)

        # 每品类单独召回，再合并
        all_candidates: list[Product] = []
        per_cat_k = max(top_k // max(len(task.categories), 1) + 5, 10)

        for category in task.categories:
            # 先按硬约束召回
            candidates = self.catalog.search(
                categories=[category],
                budget_max=budget,
                delivery_days_max=delivery_days,
                required_attrs=required_attrs or None,
                brands=brand_filter,
                top_k=per_cat_k,
            )

            # 若硬约束下结果不足，放宽属性约束重试（保留预算/时效）
            if len(candidates) < 3 and required_attrs:
                candidates = self.catalog.search(
                    categories=[category],
                    budget_max=budget,
                    delivery_days_max=delivery_days,
                    brands=brand_filter,
                    top_k=per_cat_k,
                )

            # 关键词补充：用 query 中的词做模糊召回，填补空缺品类
            if len(candidates) < 2:
                keyword = self._extract_keyword(task.raw_query, category)
                if keyword:
                    extra = self.catalog.search(
                        categories=[category],
                        keyword=keyword,
                        budget_max=budget,
                        top_k=5,
                    )
                    candidates = self._merge_dedupe(candidates, extra)

            # TF-IDF 语义向量补充：覆盖关键词匹配遗漏的语义相关商品
            if len(candidates) < per_cat_k:
                try:
                    vec_results = self.vector_index.search_products(
                        query=task.raw_query,
                        top_k=per_cat_k,
                        category_filter=category,
                        score_threshold=0.01,
                    )
                    # 预算过滤（向量检索不带预算约束，需手动过滤）
                    if budget:
                        vec_results = [p for p in vec_results if p.final_price <= budget]
                    candidates = self._merge_dedupe(candidates, vec_results)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("TF-IDF search failed for category=%s: %s", category, exc)

            all_candidates.extend(candidates)

        if not all_candidates:
            raise NoProductFoundError(
                f"在约束条件下（预算={budget}，品类={task.categories}）未找到商品。"
            )

        # 全局去重
        seen: set[str] = set()
        unique: list[Product] = []
        for p in all_candidates:
            if p.product_id not in seen:
                seen.add(p.product_id)
                unique.append(p)

        # 用户画像重排
        unique = self._rerank(unique, user_profile, budget)

        return unique[:top_k]

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _extract_attr_constraints(self, hard: dict[str, Any]) -> dict[str, Any]:
        """将任务约束字典转换为商品属性过滤条件。"""
        attrs: dict[str, Any] = {}
        for key, value in hard.items():
            if key in _NON_ATTR_CONSTRAINT_KEYS:
                continue
            mapped = _CONSTRAINT_TO_ATTR.get(key, key)
            attrs[mapped] = value
        return attrs

    def _extract_keyword(self, query: str, category: str) -> Optional[str]:
        """
        从原始 query 中提取与品类相关的关键词。
        简单实现：去除常用停用词后取最长词。
        """
        stopwords = {"帮我", "买", "一个", "一台", "一套", "推荐", "需要", "想要",
                     "用的", "以内", "不超过", "预算", "元", "块", "价格", "越便宜越好"}
        words = [w for w in query.split() if w not in stopwords and len(w) > 1]
        return words[0] if words else None

    def _merge_dedupe(self, primary: list[Product], extra: list[Product]) -> list[Product]:
        """合并两个列表并去重（primary 优先）。"""
        seen = {p.product_id for p in primary}
        return primary + [p for p in extra if p.product_id not in seen]

    def _rerank(
        self,
        candidates: list[Product],
        user_profile: Optional[UserProfile],
        budget: Optional[float],
    ) -> list[Product]:
        """
        基于用户画像重排候选列表。

        得分 = rating_score + brand_bonus - price_penalty
          rating_score  = rating / 5.0
          brand_bonus   = brand_weight * 0.3（偏好品牌加权）
          price_penalty = price_sensitivity * (final_price / budget) * 0.2
        """
        def score(p: Product) -> float:
            rating_score = (p.rating or 3.0) / 5.0

            brand_bonus = 0.0
            if user_profile and user_profile.brand_weights:
                brand_bonus = user_profile.brand_weights.get(p.brand, 0.0) * 0.3

            price_penalty = 0.0
            if user_profile and budget:
                sensitivity = getattr(user_profile, "price_sensitivity", 0.5)
                price_penalty = sensitivity * (p.final_price / budget) * 0.2

            return rating_score + brand_bonus - price_penalty

        return sorted(candidates, key=score, reverse=True)
