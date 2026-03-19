"""
HybridRetriever — 多通道商品召回器。

三路召回：
  1. 关键词检索（lexical）：精确匹配品牌/型号/关键词
  2. 语义检索（dense）：理解语义相似性
  3. 属性过滤（attribute）：基于结构化约束过滤

生产环境接入电商搜索 API（京东/淘宝开放平台等）。
当前实现为 mock，接口与生产版本一致。
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

from shopping_agent.common.constants import RETRIEVAL_TOP_K
from shopping_agent.common.exceptions import NoProductFoundError
from shopping_agent.common.types import (
    LogisticsInfo,
    Product,
    ProductAttribute,
    ShoppingTask,
    UserProfile,
)


class HybridRetriever:
    def __init__(self):
        # 生产环境：注入各平台 API 客户端
        pass

    def retrieve(
        self,
        task: ShoppingTask,
        user_profile: Optional[UserProfile] = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[Product]:
        """
        多通道召回，返回标准化前的原始候选商品列表。
        """
        hard_constraints = task.get_hard_constraints()
        budget = hard_constraints.get("budget_total")
        delivery_days = hard_constraints.get("delivery_days")

        all_candidates: list[Product] = []

        for category in task.categories:
            candidates = self._retrieve_category(
                category=category,
                budget=budget,
                delivery_days=delivery_days,
                user_profile=user_profile,
                top_k=top_k // len(task.categories) + 1,
            )
            all_candidates.extend(candidates)

        if not all_candidates:
            raise NoProductFoundError(
                f"在约束条件下（预算={budget}，品类={task.categories}）未找到商品。"
            )

        # 去重 + 按相关性粗排
        seen_ids = set()
        unique_candidates = []
        for p in all_candidates:
            if p.product_id not in seen_ids:
                seen_ids.add(p.product_id)
                unique_candidates.append(p)

        # 用户画像重排：偏好品牌提权
        if user_profile and user_profile.brand_weights:
            unique_candidates.sort(
                key=lambda p: (
                    user_profile.brand_weights.get(p.brand, 0.5),
                    p.rating or 0,
                ),
                reverse=True,
            )

        return unique_candidates[:top_k]

    def _retrieve_category(
        self,
        category: str,
        budget: Optional[float],
        delivery_days: Optional[int],
        user_profile: Optional[UserProfile],
        top_k: int,
    ) -> list[Product]:
        """
        单品类召回（mock 实现）。
        生产环境替换为真实 API 调用。
        """
        # Mock：生成模拟商品数据
        brands = ["Sony", "LG", "Samsung", "Xiaomi", "Huawei", "Apple",
                  "Logitech", "IKEA", "Herman Miller"]
        platforms = ["JD", "Tmall", "Taobao"]
        products = []

        for i in range(min(top_k, 15)):
            brand = random.choice(brands)
            platform = random.choice(platforms)
            base_price = random.uniform(200, budget * 0.8 if budget else 5000)
            rating = round(random.uniform(4.0, 5.0), 1)

            product = Product(
                product_id=str(uuid.uuid4()),
                title=f"{brand} {category} 旗舰款 {i + 1}号",
                platform=platform,
                price=round(base_price, 2),
                original_price=round(base_price * 1.1, 2),
                coupon_discount=round(base_price * 0.05, 2),
                category=category,
                brand=brand,
                attributes=[
                    ProductAttribute(name="重量", value=random.uniform(0.5, 3.0), unit="kg"),
                    ProductAttribute(name="颜色", value=random.choice(["黑色", "白色", "银色"])),
                ],
                in_stock=random.random() > 0.1,
                stock_count=random.randint(1, 500),
                logistics=LogisticsInfo(
                    platform=platform,
                    delivery_days=random.randint(1, 5),
                    delivery_fee=0.0 if base_price > 99 else 8.0,
                    supports_return=True,
                    return_days=7,
                    is_official_store=random.random() > 0.4,
                ),
                rating=rating,
                review_count=random.randint(100, 50000),
                sales_volume=random.randint(500, 100000),
            )

            # 物流约束过滤
            if delivery_days and product.logistics:
                if product.logistics.delivery_days > delivery_days:
                    continue

            products.append(product)

        return products
