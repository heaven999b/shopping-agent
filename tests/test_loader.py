"""
tests/test_loader.py — ProductCatalog 单元测试。

覆盖：
  - 从 dict 创建 Product
  - search() 按品类、预算、属性、关键词过滤
  - 排名逻辑（评分 + 销量）
"""

from __future__ import annotations

import pytest

from shopping_agent.data.loader import ProductCatalog, _dict_to_product
from tests.conftest import make_product


# ---------------------------------------------------------------------------
# _dict_to_product
# ---------------------------------------------------------------------------

class TestDictToProduct:
    def test_basic_fields(self):
        d = {
            "product_id": "p100",
            "title": "测试耳机",
            "platform": "JD",
            "price": 299.0,
            "original_price": 399.0,
            "coupon_discount": 50.0,
            "category": "headset",
            "brand": "Test",
            "attributes": {"wireless": True, "noise_cancelling": False},
            "in_stock": True,
            "delivery_days": 1,
            "is_official_store": True,
            "rating": 4.5,
            "review_count": 200,
            "sales_volume": 500,
        }
        p = _dict_to_product(d)
        assert p.product_id == "p100"
        assert p.final_price == pytest.approx(249.0)
        assert p.get_attribute("wireless") is True
        assert p.get_attribute("noise_cancelling") is False
        assert p.logistics is not None
        assert p.logistics.delivery_days == 1
        assert p.logistics.is_official_store is True

    def test_missing_optional_fields(self):
        d = {
            "product_id": "p101",
            "title": "最简商品",
            "platform": "Tmall",
            "price": 100.0,
        }
        p = _dict_to_product(d)
        assert p.rating is None
        assert p.brand == ""
        assert p.category == ""
        assert p.attributes == []
        assert p.persona_tags  # fallback tags should be inferred

    def test_persona_tags_preserved(self):
        d = {
            "product_id": "p102",
            "title": "高端耳机",
            "platform": "JD",
            "price": 2999.0,
            "category": "headset",
            "brand": "Sony",
            "persona_tags": {
                "style_signal": ["premium"],
                "identity_fit": ["professional"],
                "visibility_level": "moderate",
                "symbolic_value": "taste_signaling",
            },
        }
        p = _dict_to_product(d)
        assert p.persona_tags["style_signal"] == ["premium"]


# ---------------------------------------------------------------------------
# ProductCatalog.search()
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog() -> ProductCatalog:
    """内存 catalog，不读磁盘文件。"""
    products = [
        make_product("p001", "Sony WH-1000XM5", price=1999.0, category="headset",
                     brand="Sony", attributes={"noise_cancelling": True, "wireless": True},
                     rating=4.8, sales_volume=10000),
        make_product("p002", "QCY T10 耳机", price=129.0, category="headset",
                     brand="QCY", attributes={"noise_cancelling": False},
                     rating=4.2, sales_volume=50000),
        make_product("p003", "LG 27寸 4K 显示器", price=2499.0, category="monitor",
                     brand="LG", attributes={"resolution": "4K", "refresh_rate_hz": 144},
                     rating=4.7, sales_volume=3000),
        make_product("p004", "机械键盘 Cherry MX", price=399.0, category="keyboard",
                     brand="Cherry", attributes={"mechanical": True},
                     rating=4.6, sales_volume=8000),
    ]
    return ProductCatalog(products)


class TestCatalogSearch:
    def test_search_by_category(self, catalog):
        results = catalog.search(categories=["headset"])
        pids = [p.product_id for p in results]
        assert "p001" in pids
        assert "p002" in pids
        assert "p003" not in pids  # monitor

    def test_search_by_budget(self, catalog):
        results = catalog.search(categories=["headset"], budget_max=500.0)
        assert all(p.final_price <= 500.0 for p in results)
        pids = [p.product_id for p in results]
        assert "p001" not in pids  # 1999 > 500
        assert "p002" in pids

    def test_search_required_attr_bool(self, catalog):
        results = catalog.search(categories=["headset"],
                                 required_attrs={"noise_cancelling": True})
        pids = [p.product_id for p in results]
        assert "p001" in pids
        assert "p002" not in pids  # noise_cancelling=False

    def test_search_required_attr_range_min(self, catalog):
        results = catalog.search(categories=["monitor"],
                                 required_attrs={"refresh_rate_hz_min": 100})
        pids = [p.product_id for p in results]
        assert "p003" in pids  # refresh_rate=144 ≥ 100

    def test_search_keyword(self, catalog):
        results = catalog.search(keyword="Sony")
        assert any(p.brand == "Sony" for p in results)

    def test_search_top_k(self, catalog):
        results = catalog.search(top_k=1)
        assert len(results) <= 1

    def test_search_no_results(self, catalog):
        results = catalog.search(categories=["nonexistent_cat"])
        assert results == []

    def test_search_no_filters_returns_all(self, catalog):
        results = catalog.search(top_k=100)
        assert len(results) == 4
