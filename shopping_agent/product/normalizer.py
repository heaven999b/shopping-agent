"""
ProductNormalizer — 商品数据标准化。

职责：
  - 属性单位统一（g→kg，cm→mm 等）
  - 价格字段规范化（确保 final_price 计算正确）
  - 缺失字段填充默认值
  - 标题噪声清洗
"""

from __future__ import annotations

import re

from shopping_agent.common.types import Product, ProductAttribute


_UNIT_CONVERSIONS = {
    ("g", "kg"): lambda v: v / 1000,
    ("mm", "cm"): lambda v: v / 10,
    ("cm", "m"): lambda v: v / 100,
    ("ml", "l"): lambda v: v / 1000,
}


class ProductNormalizer:
    def normalize(self, product: Product) -> Product:
        """对单个商品做标准化处理，返回原对象（in-place 修改）。"""
        self._clean_title(product)
        self._normalize_price(product)
        self._normalize_attributes(product)
        self._fill_defaults(product)
        return product

    def _clean_title(self, product: Product) -> None:
        title = product.title
        # 移除常见营销噪声词
        noise_patterns = [
            r"【.*?】", r"\[.*?\]", r"（限时.*?）",
            r"正品保障", r"官方旗舰", r"全国联保",
        ]
        for pattern in noise_patterns:
            title = re.sub(pattern, "", title)
        product.title = title.strip()

    def _normalize_price(self, product: Product) -> None:
        # 确保价格为正数
        if product.price <= 0:
            product.price = product.original_price or 0.0

        # 优惠券折扣不超过原价的 50%（防止数据异常）
        if product.original_price and product.coupon_discount:
            max_discount = product.original_price * 0.5
            product.coupon_discount = min(product.coupon_discount, max_discount)

    def _normalize_attributes(self, product: Product) -> None:
        normalized_attrs = []
        for attr in product.attributes:
            attr = self._convert_unit(attr)
            normalized_attrs.append(attr)
        product.attributes = normalized_attrs

    def _convert_unit(self, attr: ProductAttribute) -> ProductAttribute:
        """统一单位，例如 g→kg。"""
        if attr.unit is None or not isinstance(attr.value, (int, float)):
            return attr
        for (from_unit, to_unit), converter in _UNIT_CONVERSIONS.items():
            if attr.unit == from_unit:
                attr.value = round(converter(attr.value), 3)
                attr.unit = to_unit
                break
        return attr

    def _fill_defaults(self, product: Product) -> None:
        if not product.brand:
            # 从标题提取品牌（简单 heuristic）
            parts = product.title.split()
            if parts:
                product.brand = parts[0]

        if product.rating is None:
            product.rating = 4.0  # 无评分时给默认值

        if not product.category:
            product.category = "other"
