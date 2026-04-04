"""
tests/test_retriever.py — HybridRetriever 单元测试。

覆盖：
  - 基础召回：品类过滤、预算过滤
  - 无结果时抛出 NoProductFoundError
  - 用户画像重排（偏好品牌优先）
  - TF-IDF 向量通道补充（低命中场景）
"""

from __future__ import annotations

import pytest

from shopping_agent.common.exceptions import NoProductFoundError
from shopping_agent.common.types import UserProfile
from shopping_agent.data.loader import ProductCatalog
from shopping_agent.retrieval.retriever import HybridRetriever
from shopping_agent.retrieval.vector_index import TFIDFVectorIndex
from tests.conftest import make_product, make_task


@pytest.fixture
def small_catalog() -> ProductCatalog:
    products = [
        make_product("p001", "Sony WH-1000XM5 降噪耳机", price=1999.0, category="headset",
                     brand="Sony", attributes={"noise_cancelling": True}, rating=4.8),
        make_product("p002", "QCY T10 蓝牙耳机", price=129.0, category="headset",
                     brand="QCY", attributes={"noise_cancelling": False}, rating=4.2),
        make_product("p003", "LG 27寸显示器", price=2499.0, category="monitor",
                     brand="LG", rating=4.7),
    ]
    return ProductCatalog(products)


@pytest.fixture
def retriever(small_catalog) -> HybridRetriever:
    # 同时构建 vector index
    idx = TFIDFVectorIndex()
    idx.build(list(small_catalog._products))  # _products is list[Product]
    return HybridRetriever(catalog=small_catalog, vector_index=idx)


class TestHybridRetrieverBasic:
    def test_retrieve_by_category(self, retriever):
        task = make_task(categories=["headset"], budget=5000.0)
        results = retriever.retrieve(task, top_k=10)
        assert len(results) >= 1
        assert all(p.category == "headset" for p in results)

    def test_retrieve_budget_filter(self, retriever):
        task = make_task(categories=["headset"], budget=500.0)
        results = retriever.retrieve(task, top_k=10)
        # p001 (1999) should be excluded; p002 (129) included
        pids = [p.product_id for p in results]
        assert "p001" not in pids
        assert "p002" in pids

    def test_retrieve_no_products_raises(self, retriever):
        task = make_task(categories=["nonexistent"], budget=100.0)
        with pytest.raises(NoProductFoundError):
            retriever.retrieve(task, top_k=5)

    def test_retrieve_deduplication(self, retriever):
        """同一商品不应重复出现。"""
        task = make_task(categories=["headset"], budget=5000.0)
        results = retriever.retrieve(task, top_k=20)
        pids = [p.product_id for p in results]
        assert len(pids) == len(set(pids))


class TestHybridRetrieverRerank:
    def test_preferred_brand_ranked_higher(self, retriever, user_profile):
        """偏好品牌 Sony 应排在 QCY 前面。"""
        task = make_task(categories=["headset"], budget=5000.0)
        results = retriever.retrieve(task, user_profile=user_profile, top_k=10)
        brands = [p.brand for p in results]
        if "Sony" in brands and "QCY" in brands:
            assert brands.index("Sony") < brands.index("QCY")

    def test_persona_alignment_can_raise_professional_product(self, small_catalog):
        products = list(small_catalog._products)
        products[0].persona_tags = {
            "style_signal": ["premium", "clean"],
            "identity_fit": ["professional"],
            "visibility_level": "moderate",
            "symbolic_value": "taste_signaling",
        }
        products[1].persona_tags = {
            "style_signal": ["playful", "tech"],
            "identity_fit": ["gamer"],
            "visibility_level": "attention_grabbing",
            "symbolic_value": "utilitarian",
        }
        idx = TFIDFVectorIndex()
        idx.build(products)
        retriever = HybridRetriever(catalog=small_catalog, vector_index=idx)

        profile = UserProfile(
            user_id="u_persona",
            identity_goal={"professional": 0.9},
            budget_sensitivity_profile={"premium": 0.8},
            brand_orientation={"brand_signal": 0.7},
            aesthetic_preference={"clean": 0.8, "premium": 0.8},
            persona_stability=0.9,
        )
        task = make_task(categories=["headset"], budget=5000.0)
        results = retriever.retrieve(task, user_profile=profile, top_k=10)
        assert results[0].product_id == "p001"


class TestTFIDFVectorIndex:
    def test_build_and_search(self):
        products = [
            make_product("p1", "蓝牙降噪头戴式耳机", category="headset",
                         attributes={"noise_cancelling": True}),
            make_product("p2", "27寸4K显示器", category="monitor"),
            make_product("p3", "机械键盘 RGB", category="keyboard",
                         attributes={"mechanical": True}),
        ]
        idx = TFIDFVectorIndex()
        idx.build(products)
        assert idx.is_built
        assert idx.product_count == 3

        results = idx.search("降噪耳机", top_k=3)
        assert len(results) >= 1
        top_product, top_score = results[0]
        assert top_product.product_id == "p1"
        assert top_score > 0

    def test_category_filter(self):
        products = [
            make_product("p1", "蓝牙降噪耳机", category="headset"),
            make_product("p2", "27寸显示器", category="monitor"),
        ]
        idx = TFIDFVectorIndex()
        idx.build(products)

        results = idx.search_products("耳机", top_k=5, category_filter="headset")
        assert all(p.category == "headset" for p in results)

    def test_empty_build(self):
        idx = TFIDFVectorIndex()
        idx.build([])
        assert not idx.is_built
        # search on unbuilt index returns empty
        assert idx.search("query") == []

    def test_rebuild(self):
        products_v1 = [make_product("p1", "耳机A", category="headset")]
        products_v2 = [
            make_product("p1", "耳机A", category="headset"),
            make_product("p2", "显示器B", category="monitor"),
        ]
        idx = TFIDFVectorIndex()
        idx.build(products_v1)
        assert idx.product_count == 1

        idx.rebuild(products_v2)
        assert idx.product_count == 2
