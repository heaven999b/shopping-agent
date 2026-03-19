"""
pytest 公共夹具（Fixtures）。

所有测试均使用内存数据（不依赖磁盘文件），
通过工厂函数快速创建 Product / ShoppingTask / CandidatePlan。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from shopping_agent.common.types import (
    CandidatePlan,
    Constraint,
    ConstraintSeverity,
    LogisticsInfo,
    PlanItem,
    Product,
    ProductAttribute,
    ShoppingTask,
    TaskType,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Product 工厂
# ---------------------------------------------------------------------------

def make_product(
    product_id: str = "p001",
    title: str = "索尼 WH-1000XM5 降噪耳机",
    platform: str = "JD",
    price: float = 1999.0,
    coupon_discount: float = 0.0,
    category: str = "headset",
    brand: str = "Sony",
    in_stock: bool = True,
    delivery_days: int = 2,
    rating: float = 4.8,
    review_count: int = 5000,
    sales_volume: int = 10000,
    attributes: dict | None = None,
    ttl_seconds: int = 3600,
) -> Product:
    attrs = []
    for k, v in (attributes or {}).items():
        attrs.append(ProductAttribute(name=k, value=v))
    logistics = LogisticsInfo(
        platform=platform,
        delivery_days=delivery_days,
        is_official_store=True,
    )
    return Product(
        product_id=product_id,
        title=title,
        platform=platform,
        price=price,
        coupon_discount=coupon_discount,
        category=category,
        brand=brand,
        attributes=attrs,
        in_stock=in_stock,
        logistics=logistics,
        rating=rating,
        review_count=review_count,
        sales_volume=sales_volume,
        ttl_seconds=ttl_seconds,
    )


@pytest.fixture
def product_headset() -> Product:
    return make_product(
        product_id="p001",
        title="索尼 WH-1000XM5 蓝牙降噪耳机",
        price=1999.0,
        category="headset",
        brand="Sony",
        attributes={"noise_cancelling": True, "wireless": True},
    )


@pytest.fixture
def product_cheap_headset() -> Product:
    return make_product(
        product_id="p002",
        title="QCY T10 蓝牙耳机",
        price=129.0,
        category="headset",
        brand="QCY",
        rating=4.2,
        attributes={"noise_cancelling": False, "wireless": True},
    )


@pytest.fixture
def product_monitor() -> Product:
    return make_product(
        product_id="p003",
        title="LG 27英寸 4K 显示器",
        price=2499.0,
        category="monitor",
        brand="LG",
        attributes={"resolution": "4K", "refresh_rate_hz": 60},
    )


@pytest.fixture
def product_out_of_stock() -> Product:
    return make_product(
        product_id="p004",
        title="某品牌缺货耳机",
        price=500.0,
        category="headset",
        brand="Unknown",
        in_stock=False,
    )


@pytest.fixture
def sample_products(product_headset, product_cheap_headset, product_monitor, product_out_of_stock) -> list[Product]:
    return [product_headset, product_cheap_headset, product_monitor, product_out_of_stock]


# ---------------------------------------------------------------------------
# ShoppingTask 工厂
# ---------------------------------------------------------------------------

def make_task(
    task_id: str = "t001",
    task_type: TaskType = TaskType.SINGLE,
    categories: list[str] | None = None,
    raw_query: str = "帮我买一个降噪耳机，预算2000以内",
    budget: float | None = 2000.0,
    hard_constraints: dict | None = None,
    soft_constraints: dict | None = None,
    uncertainty_score: float = 0.2,
) -> ShoppingTask:
    constraints = []
    if budget is not None:
        constraints.append(Constraint(
            key="budget_total",
            value=budget,
            severity=ConstraintSeverity.HARD,
        ))
    for k, v in (hard_constraints or {}).items():
        constraints.append(Constraint(key=k, value=v, severity=ConstraintSeverity.HARD))
    for k, v in (soft_constraints or {}).items():
        constraints.append(Constraint(key=k, value=v, severity=ConstraintSeverity.SOFT))

    return ShoppingTask(
        task_id=task_id,
        task_type=task_type,
        categories=categories or ["headset"],
        raw_query=raw_query,
        constraints=constraints,
        uncertainty_score=uncertainty_score,
    )


@pytest.fixture
def task_headset_budget() -> ShoppingTask:
    """预算 2000 内的降噪耳机任务。"""
    return make_task(
        task_id="t001",
        categories=["headset"],
        budget=2000.0,
        hard_constraints={"noise_cancelling": True},
        raw_query="帮我买一个降噪耳机，预算2000以内",
    )


@pytest.fixture
def task_bundle() -> ShoppingTask:
    """耳机 + 显示器组合任务。"""
    return make_task(
        task_id="t002",
        task_type=TaskType.BUNDLE,
        categories=["headset", "monitor"],
        budget=5000.0,
        raw_query="帮我配一套桌面：耳机和显示器，预算5000",
    )


@pytest.fixture
def task_no_budget() -> ShoppingTask:
    """无预算约束的任务。"""
    return make_task(
        task_id="t003",
        categories=["headset"],
        budget=None,
        raw_query="推荐一款好耳机",
    )


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------

@pytest.fixture
def user_profile() -> UserProfile:
    return UserProfile(
        user_id="u001",
        brand_weights={"Sony": 0.9, "LG": 0.7},
        price_sensitivity=0.6,
        prefer_fast_delivery=True,
        prefer_official_store=True,
    )


# ---------------------------------------------------------------------------
# CandidatePlan 工厂
# ---------------------------------------------------------------------------

def make_plan(
    plan_id: str = "plan001",
    task_id: str = "t001",
    products: list[Product] | None = None,
    total_price: float | None = None,
) -> CandidatePlan:
    if products is None:
        products = [make_product()]
    items = [
        PlanItem(bundle_slot=p.category, product=p, reason="测试方案")
        for p in products
    ]
    tp = total_price if total_price is not None else sum(p.final_price for p in products)
    return CandidatePlan(
        plan_id=plan_id,
        task_id=task_id,
        items=items,
        total_price=tp,
        constraint_score=0.9,
        preference_score=0.8,
        value_score=0.75,
    )


@pytest.fixture
def plan_within_budget(product_headset) -> CandidatePlan:
    return make_plan(plan_id="plan001", products=[product_headset])


@pytest.fixture
def plan_over_budget() -> CandidatePlan:
    expensive = make_product(product_id="p_exp", price=9999.0)
    return make_plan(plan_id="plan002", products=[expensive], total_price=9999.0)
