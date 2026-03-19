"""shopping_agent.retrieval — 商品多路召回。"""

__all__ = ["HybridRetriever"]


def __getattr__(name: str):
    if name == "HybridRetriever":
        from shopping_agent.retrieval.retriever import HybridRetriever

        return HybridRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
