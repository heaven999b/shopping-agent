"""shopping_agent.product — 商品标准化与候选图。"""

from shopping_agent.product.candidate_graph import CandidateGraphBuilder, GraphBasedRecovery
from shopping_agent.product.normalizer import ProductNormalizer

__all__ = ["CandidateGraphBuilder", "GraphBasedRecovery", "ProductNormalizer"]
