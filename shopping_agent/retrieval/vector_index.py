"""
TF-IDF 向量索引 — 商品语义召回通道。

功能：
  - 对每件商品的"文本摘要"（标题 + 品牌 + 属性键值）构建 TF-IDF 矩阵
  - 查询时做 cosine 相似度检索，返回 top-k 候选
  - 支持按品类过滤，减少跨品类噪声
  - 懒构建（首次 search 时才 fit），支持 rebuild() 热更新

集成方式：
  HybridRetriever.retrieve() 在关键词召回结果不足时，
  调用 TFIDFVectorIndex.search() 补充语义召回结果。

依赖：scikit-learn（仅用 TfidfVectorizer 和 cosine_similarity）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shopping_agent.common.types import Product

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _product_to_text(product: Product) -> str:
    """
    将 Product 序列化为可索引文本串。

    格式：<title> <brand> <category> <attr_key=attr_value> ...
    中文字符空格切分（sklearn TF-IDF 默认 tokenizer 对中文不友好，
    但我们的数据混合中英文，空格分词够用；生产环境可接入 jieba）。
    """
    parts = [product.title, product.brand, product.category]
    for attr in product.attributes:
        parts.append(f"{attr.name}={attr.value}")
    return " ".join(str(p) for p in parts if p)


class TFIDFVectorIndex:
    """
    基于 TF-IDF 的商品向量索引。

    用法：
        index = TFIDFVectorIndex()
        index.build(products)
        results = index.search("头戴式降噪耳机", top_k=10)
        results = index.search("机械键盘", category_filter="keyboard", top_k=5)
    """

    def __init__(
        self,
        min_df: int = 1,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        """
        参数：
            min_df: 词项最少出现文档数（过滤极低频词）
            max_features: 词典最大词汇量
            ngram_range: n-gram 范围，(1,2) 同时使用 unigram + bigram
        """
        self._vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer="char_wb",   # 字符级 n-gram，对中文更友好
            sublinear_tf=True,    # log(1+tf) 压缩高频词权重
        )
        self._products: list[Product] = []
        self._tfidf_matrix = None   # shape: (n_products, n_features)
        self._built = False
        self._category_index: dict[str, list[int]] = {}  # category → row indices

    # ------------------------------------------------------------------
    # 构建索引
    # ------------------------------------------------------------------

    def build(self, products: list[Product]) -> None:
        """
        用给定的商品列表构建（或重建）TF-IDF 索引。
        可随时调用以热更新（catalog 刷新后）。
        """
        if not products:
            logger.warning("TFIDFVectorIndex.build() called with empty product list")
            return

        self._products = products
        corpus = [_product_to_text(p) for p in products]

        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self._built = True

        # 按品类建倒排，用于 category_filter
        self._category_index = {}
        for i, p in enumerate(products):
            cat = p.category.lower()
            self._category_index.setdefault(cat, []).append(i)

        logger.info(
            "TFIDFVectorIndex built: %d products, vocab_size=%d",
            len(products),
            len(self._vectorizer.vocabulary_),
        )

    def rebuild(self, products: list[Product]) -> None:
        """别名，与 build() 等价（语义上强调"重建"）。"""
        self.build(products)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[Product, float]]:
        """
        TF-IDF cosine 相似度检索。

        参数：
            query: 用户原始查询或任意文本
            top_k: 返回候选数
            category_filter: 若指定，只在该品类内检索（模糊匹配）
            score_threshold: 相似度下限，低于此值的结果被过滤

        返回：
            list of (Product, score) 按相似度降序排列
        """
        if not self._built or self._tfidf_matrix is None:
            logger.warning("TFIDFVectorIndex not built, returning empty results")
            return []

        # 查询向量
        query_vec = self._vectorizer.transform([query])  # shape: (1, n_features)

        # 确定检索范围（全量 or 品类子集）
        if category_filter:
            candidate_indices = self._get_category_indices(category_filter)
            if not candidate_indices:
                # 品类无匹配时退回全量检索
                candidate_indices = list(range(len(self._products)))
        else:
            candidate_indices = list(range(len(self._products)))

        if not candidate_indices:
            return []

        # 取子矩阵做 cosine 相似度
        sub_matrix = self._tfidf_matrix[candidate_indices]  # (n_candidates, n_features)
        scores = cosine_similarity(query_vec, sub_matrix)[0]  # shape: (n_candidates,)

        # 排序，取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[Product, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < score_threshold:
                break
            product = self._products[candidate_indices[idx]]
            results.append((product, score))

        return results

    def search_products(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> list[Product]:
        """
        search() 的简化版，只返回 Product 列表（忽略 score）。
        HybridRetriever 使用此接口。
        """
        return [p for p, _ in self.search(query, top_k, category_filter, score_threshold)]

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _get_category_indices(self, category_filter: str) -> list[int]:
        """模糊匹配品类名，返回对应行索引。"""
        cat_lower = category_filter.lower()
        # 精确匹配
        if cat_lower in self._category_index:
            return self._category_index[cat_lower]
        # 前缀/子串匹配
        indices: list[int] = []
        for cat, idxs in self._category_index.items():
            if cat_lower in cat or cat in cat_lower:
                indices.extend(idxs)
        return indices

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def product_count(self) -> int:
        return len(self._products)

    def stats(self) -> dict:
        return {
            "built": self._built,
            "product_count": len(self._products),
            "vocab_size": len(self._vectorizer.vocabulary_) if self._built else 0,
            "categories": list(self._category_index.keys()),
        }
