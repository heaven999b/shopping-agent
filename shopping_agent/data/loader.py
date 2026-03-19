"""
ProductCatalog — JSON 商品目录加载与检索索引。

将 data/products.json 加载为内存索引，提供：
  - 按品类过滤
  - 关键词匹配（title / brand）
  - 属性过滤（exact + range）
  - 预算过滤
  - 评分排序

这是 HybridRetriever 的数据后端。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from shopping_agent.common.types import LogisticsInfo, Product, ProductAttribute


# 默认数据目录（相对于项目根）
_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _find_data_file(filename: str) -> Path:
    """从几个候选位置找 data 文件。"""
    candidates = [
        _DEFAULT_DATA_DIR / filename,
        Path("data") / filename,
        Path(__file__).parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"找不到数据文件 {filename}，候选路径: {[str(c) for c in candidates]}"
    )


def _dict_to_product(d: dict) -> Product:
    """将 JSON 字典转换为 Product 对象。"""
    attrs = [
        ProductAttribute(name=k, value=v)
        for k, v in d.get("attributes", {}).items()
    ]
    logistics = LogisticsInfo(
        platform=d["platform"],
        delivery_days=d.get("delivery_days"),
        delivery_fee=0.0,
        supports_return=True,
        return_days=7,
        is_official_store=d.get("is_official_store", False),
    )
    persona_tags = d.get("persona_tags") or _infer_persona_tags(d)
    return Product(
        product_id=d["product_id"],
        title=d["title"],
        platform=d["platform"],
        price=float(d["price"]),
        original_price=d.get("original_price"),
        coupon_discount=float(d.get("coupon_discount", 0.0)),
        category=d.get("category", ""),
        brand=d.get("brand", ""),
        attributes=attrs,
        in_stock=d.get("in_stock", True),
        logistics=logistics,
        rating=d.get("rating"),
        review_count=d.get("review_count", 0),
        sales_volume=d.get("sales_volume", 0),
        persona_tags=persona_tags,
    )


def _infer_persona_tags(d: dict) -> dict[str, Any]:
    """根据商品元数据推断最小 persona tags，便于逐步迁移到显式标注。"""
    category = d.get("category", "")
    brand = (d.get("brand") or "").lower()
    price = float(d.get("price", 0.0))
    title = d.get("title", "").lower()
    color = str(d.get("attributes", {}).get("color", "")).lower()

    style_signal: list[str] = []
    identity_fit: list[str] = []
    symbolic_value = "utilitarian"
    visibility_level = "moderate"

    premium_brands = {"sony", "bose", "apple", "samsung", "dell", "lg"}
    playful_brands = {"logitech", "razer", "aoc"}

    if brand in premium_brands or price >= 1800:
        style_signal.extend(["premium", "professional"])
        symbolic_value = "taste_signaling"
    if brand in playful_brands or "rgb" in title or "游戏" in title:
        style_signal.extend(["tech", "playful"])
        visibility_level = "attention_grabbing"
    if color in {"黑色", "silver", "银色", "gray", "灰色"}:
        style_signal.append("clean")
    if category in {"monitor", "laptop", "chair", "desk"}:
        identity_fit.extend(["office", "professional"])
    if category in {"keyboard", "mouse", "headset"}:
        identity_fit.extend(["creator", "gamer"])
    if "无线" in title or category == "headset":
        identity_fit.append("traveler")

    if not style_signal:
        style_signal.append("practical")
    if not identity_fit:
        identity_fit.append("general")

    return {
        "style_signal": list(dict.fromkeys(style_signal)),
        "identity_fit": list(dict.fromkeys(identity_fit)),
        "visibility_level": visibility_level,
        "symbolic_value": symbolic_value,
    }


class ProductCatalog:
    """
    内存商品目录，提供高效的多维过滤接口。

    用法：
        catalog = ProductCatalog.load()
        results = catalog.search(
            categories=["headset"],
            budget_max=2000,
            required_attrs={"noise_cancelling": True},
            keyword="Sony",
        )
    """

    def __init__(self, products: list[Product]):
        self._products = products
        # 品类倒排索引
        self._by_category: dict[str, list[Product]] = {}
        for p in products:
            self._by_category.setdefault(p.category, []).append(p)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ProductCatalog":
        """从 JSON 文件加载商品目录。"""
        if path:
            data_path = Path(path)
        else:
            data_path = _find_data_file("products.json")

        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)

        products = [_dict_to_product(d) for d in raw]
        return cls(products)

    # ------------------------------------------------------------------
    # 检索接口
    # ------------------------------------------------------------------

    def search(
        self,
        categories: Optional[list[str]] = None,
        keyword: Optional[str] = None,
        budget_max: Optional[float] = None,
        delivery_days_max: Optional[int] = None,
        required_attrs: Optional[dict[str, Any]] = None,
        brands: Optional[list[str]] = None,
        top_k: int = 20,
    ) -> list[Product]:
        """
        多条件检索，返回按评分降序排列的候选列表。

        参数：
          categories       — 品类白名单（模糊匹配）
          keyword          — 在 title/brand 中做关键词包含匹配
          budget_max       — final_price 上限
          delivery_days_max — 配送天数上限
          required_attrs   — 必须满足的属性 kv，支持 bool / str exact / "_min"后缀范围
          brands           — 品牌白名单（精确）
          top_k            — 返回数量上限
        """
        candidates = list(self._products)

        # 1. 品类过滤（支持模糊匹配，如 "headset" 匹配 "headset"）
        if categories:
            candidates = [
                p for p in candidates
                if any(self._category_match(p.category, cat) for cat in categories)
            ]

        # 2. 品牌过滤
        if brands:
            normalized_brands = [b.lower() for b in brands]
            candidates = [
                p for p in candidates
                if p.brand.lower() in normalized_brands
            ]

        # 3. 关键词过滤
        if keyword:
            kw = keyword.lower()
            candidates = [
                p for p in candidates
                if kw in p.title.lower() or kw in p.brand.lower()
            ]

        # 4. 预算过滤
        if budget_max is not None:
            candidates = [p for p in candidates if p.final_price <= budget_max]

        # 5. 配送时效过滤
        if delivery_days_max is not None:
            candidates = [
                p for p in candidates
                if p.logistics and p.logistics.delivery_days is not None
                   and p.logistics.delivery_days <= delivery_days_max
            ]

        # 6. 属性过滤
        if required_attrs:
            candidates = [
                p for p in candidates
                if self._attrs_match(p, required_attrs)
            ]

        # 7. 按综合分排序（rating * log(review_count+1)）
        candidates.sort(
            key=lambda p: (p.rating or 0) * self._log1p(p.review_count),
            reverse=True,
        )

        return candidates[:top_k]

    def get_by_id(self, product_id: str) -> Optional[Product]:
        for p in self._products:
            if p.product_id == product_id:
                return p
        return None

    def categories(self) -> list[str]:
        return list(self._by_category.keys())

    def count(self) -> int:
        return len(self._products)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _category_match(product_cat: str, query_cat: str) -> bool:
        """品类匹配：支持精确和包含两种方式。"""
        return (product_cat.lower() == query_cat.lower() or
                query_cat.lower() in product_cat.lower() or
                product_cat.lower() in query_cat.lower())

    @staticmethod
    def _attrs_match(product: Product, required: dict[str, Any]) -> bool:
        """
        检查商品是否满足所有必要属性。

        支持：
          key: value          → 精确匹配（bool / str）
          key + "_min": v     → product.attr[key] >= v
          key + "_max": v     → product.attr[key] <= v
        """
        attr_dict = {a.name: a.value for a in product.attributes}

        for key, required_value in required.items():
            if key.endswith("_min"):
                base_key = key[:-4]
                actual = attr_dict.get(base_key)
                if actual is None or float(actual) < float(required_value):
                    return False
            elif key.endswith("_max"):
                base_key = key[:-4]
                actual = attr_dict.get(base_key)
                if actual is None or float(actual) > float(required_value):
                    return False
            else:
                actual = attr_dict.get(key)
                if actual is None:
                    return False
                if isinstance(required_value, bool):
                    if bool(actual) != required_value:
                        return False
                elif str(actual).lower() != str(required_value).lower():
                    return False

        return True

    @staticmethod
    def _log1p(x: int) -> float:
        import math
        return math.log1p(x)


# ---------------------------------------------------------------------------
# 任务数据加载
# ---------------------------------------------------------------------------

def load_benchmark_tasks(path: Optional[str] = None) -> list[dict]:
    """
    从 data/tasks.json 加载 benchmark 任务列表。
    每个任务是 dict，包含 task_id / query / constraints / expected 等字段。
    """
    if path:
        data_path = Path(path)
    else:
        data_path = _find_data_file("tasks.json")

    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 全局单例（延迟加载）
# ---------------------------------------------------------------------------

_catalog_instance: Optional[ProductCatalog] = None


def get_catalog() -> ProductCatalog:
    """获取全局单例 ProductCatalog（延迟加载）。"""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = ProductCatalog.load()
    return _catalog_instance
