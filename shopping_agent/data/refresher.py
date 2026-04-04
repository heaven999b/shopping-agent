"""
DataRefresher — 商品数据时效管理器。

职责：
  1. 检测过期商品（Product.is_fresh() 返回 False）
  2. 触发数据刷新（MockAPIAdapter → 生产环境替换为真实电商 API）
  3. 维护 catalog 版本，支持热更新（不重启服务）
  4. CatalogManager：带自动刷新的 ProductCatalog 包装器

架构说明：
  真实 API 只需实现 BaseProductAPI 接口（fetch_product / fetch_by_category），
  其余 TTL 管理、版本追踪、刷新调度逻辑无需修改。
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from shopping_agent.common.constants import PRODUCT_DATA_TTL
from shopping_agent.common.types import Product
from shopping_agent.data.loader import ProductCatalog, _dict_to_product, _find_data_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 抽象 API 接口（生产环境实现此接口对接真实平台）
# ---------------------------------------------------------------------------

class BaseProductAPI(ABC):
    """电商平台 API 抽象接口。生产环境继承此类接入真实平台。"""

    @abstractmethod
    def fetch_product(self, product_id: str) -> Optional[dict]:
        """拉取单个商品最新数据。返回 None 表示商品已下架。"""

    @abstractmethod
    def fetch_by_category(self, category: str, top_k: int = 50) -> list[dict]:
        """按品类拉取最新商品列表（用于全量刷新）。"""


class MockProductAPI(BaseProductAPI):
    """
    Mock API 实现（基于 products.json，模拟价格波动 ±5% + 库存变化）。

    替换为真实 API 时：继承 BaseProductAPI 并实现 fetch_product / fetch_by_category，
    构造 DataRefresher 时传入新的 api 实例即可。
    """

    def __init__(self, data_path: Optional[str] = None):
        path = Path(data_path) if data_path else _find_data_file("products.json")
        with open(path, encoding="utf-8") as f:
            self._raw: list[dict] = json.load(f)
        self._index: dict[str, dict] = {d["product_id"]: d for d in self._raw}

    def fetch_product(self, product_id: str) -> Optional[dict]:
        """模拟价格 ±5% 波动、库存随机变化。"""
        raw = self._index.get(product_id)
        if raw is None:
            return None
        updated = dict(raw)
        # 模拟价格波动
        fluctuation = random.uniform(0.95, 1.05)
        updated["price"] = round(raw["price"] * fluctuation, 2)
        if "original_price" in raw and raw["original_price"]:
            updated["original_price"] = round(raw["original_price"] * fluctuation, 2)
        # 模拟库存
        updated["in_stock"] = random.random() > 0.05  # 95% 概率有货
        return updated

    def fetch_by_category(self, category: str, top_k: int = 50) -> list[dict]:
        results = [d for d in self._raw if d.get("category") == category]
        # 每个都施加价格波动
        updated = []
        for d in results[:top_k]:
            fetched = self.fetch_product(d["product_id"])
            if fetched:
                updated.append(fetched)
        return updated


# ---------------------------------------------------------------------------
# DataRefresher — TTL 管理 + 刷新调度
# ---------------------------------------------------------------------------

class DataRefresher:
    """
    商品数据刷新器。

    工作模式：
      - 按需刷新（refresh_stale）：检测 is_fresh() 失败的商品，批量刷新
      - 全量刷新（refresh_all）：重新拉取所有品类

    线程安全：内部使用 RLock 保护 _products 字典。
    """

    def __init__(
        self,
        api: Optional[BaseProductAPI] = None,
        ttl_seconds: int = PRODUCT_DATA_TTL,
    ):
        self._api = api or MockProductAPI()
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        # product_id → Product（当前已知的最新版本）
        self._products: dict[str, Product] = {}
        self._refresh_count = 0
        self._last_full_refresh: Optional[datetime] = None

    def register(self, products: list[Product]) -> None:
        """将初始商品列表注册到刷新器。"""
        with self._lock:
            for p in products:
                self._products[p.product_id] = p

    def refresh_stale(self) -> tuple[list[Product], list[str]]:
        """
        检测并刷新过期商品。

        返回 (refreshed_products, failed_ids)
        """
        with self._lock:
            stale = [p for p in self._products.values() if not p.is_fresh()]

        if not stale:
            return [], []

        refreshed, failed = [], []
        for product in stale:
            try:
                raw = self._api.fetch_product(product.product_id)
                if raw is None:
                    # 商品已下架
                    logger.info(f"商品 {product.product_id} 已下架，从目录移除")
                    with self._lock:
                        self._products.pop(product.product_id, None)
                    failed.append(product.product_id)
                    continue

                raw["fetched_at"] = datetime.now().isoformat()  # 不影响 _dict_to_product
                updated = _dict_to_product(raw)
                with self._lock:
                    self._products[updated.product_id] = updated
                refreshed.append(updated)
                self._refresh_count += 1
            except Exception as e:
                logger.warning(f"刷新商品 {product.product_id} 失败: {e}")
                failed.append(product.product_id)

        if refreshed:
            logger.info(f"已刷新 {len(refreshed)} 个过期商品，失败 {len(failed)} 个")
        return refreshed, failed

    def get_all(self) -> list[Product]:
        """返回当前所有（含最新版）商品。"""
        with self._lock:
            return list(self._products.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._products)
            stale_count = sum(1 for p in self._products.values() if not p.is_fresh())
        return {
            "total_products": total,
            "stale_products": stale_count,
            "fresh_products": total - stale_count,
            "total_refreshes": self._refresh_count,
            "last_full_refresh": self._last_full_refresh.isoformat() if self._last_full_refresh else None,
        }


# ---------------------------------------------------------------------------
# CatalogManager — 带自动刷新的 ProductCatalog 包装器
# ---------------------------------------------------------------------------

class CatalogManager:
    """
    ProductCatalog 的生产级包装器。

    新增能力：
      - 自动检测并刷新过期商品（每次 search() 时惰性触发）
      - 支持热重载（reload()）
      - 提供 stats() 监控接口

    用法（替换 get_catalog()）：
        manager = CatalogManager.create()
        products = manager.search(categories=["headset"], budget_max=2000)
    """

    def __init__(
        self,
        catalog: ProductCatalog,
        refresher: DataRefresher,
        auto_refresh: bool = True,
        refresh_interval_s: int = 60,
    ):
        self._catalog = catalog
        self._refresher = refresher
        self._auto_refresh = auto_refresh
        self._refresh_interval = refresh_interval_s
        self._last_refresh_check: float = time.time()
        self._lock = threading.RLock()

        # 注册所有商品到刷新器
        refresher.register(catalog._products)

    @classmethod
    def create(
        cls,
        data_path: Optional[str] = None,
        api: Optional[BaseProductAPI] = None,
        auto_refresh: bool = True,
    ) -> "CatalogManager":
        """工厂方法：加载 JSON + 初始化刷新器。"""
        catalog = ProductCatalog.load(data_path)
        refresher = DataRefresher(api=api)
        return cls(catalog, refresher, auto_refresh=auto_refresh)

    def search(self, **kwargs) -> list[Product]:
        """代理 ProductCatalog.search()，自动触发过期检测。"""
        if self._auto_refresh:
            self._maybe_refresh()
        return self._catalog.search(**kwargs)

    def reload(self, data_path: Optional[str] = None) -> None:
        """热重载商品目录（不停机更新）。"""
        with self._lock:
            new_catalog = ProductCatalog.load(data_path)
            self._catalog = new_catalog
            self._refresher.register(new_catalog._products)
        logger.info(f"商品目录已热重载，共 {self._catalog.count()} 个商品")

    def stats(self) -> dict[str, Any]:
        return {
            "catalog": {
                "total": self._catalog.count(),
                "categories": self._catalog.categories(),
            },
            "refresher": self._refresher.stats(),
        }

    def _maybe_refresh(self) -> None:
        """若距上次检查超过 refresh_interval，触发一次过期检测。"""
        now = time.time()
        if now - self._last_refresh_check < self._refresh_interval:
            return
        self._last_refresh_check = now
        refreshed, _ = self._refresher.refresh_stale()
        if refreshed:
            # 将刷新后的商品更新到 catalog 内存中
            with self._lock:
                for p in refreshed:
                    # 更新 _products 列表中的对应项
                    for i, existing in enumerate(self._catalog._products):
                        if existing.product_id == p.product_id:
                            self._catalog._products[i] = p
                            break
                    # 更新品类索引
                    if p.category in self._catalog._by_category:
                        bucket = self._catalog._by_category[p.category]
                        for i, existing in enumerate(bucket):
                            if existing.product_id == p.product_id:
                                bucket[i] = p
                                break


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_manager_instance: Optional[CatalogManager] = None
_manager_lock = threading.Lock()


def get_catalog_manager(auto_refresh: bool = True) -> CatalogManager:
    """获取全局 CatalogManager 单例（线程安全的延迟初始化）。"""
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = CatalogManager.create(auto_refresh=auto_refresh)
    return _manager_instance
