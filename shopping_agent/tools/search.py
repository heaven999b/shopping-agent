"""
SearchTool — 商品搜索工具。

生产环境对接：京东开放平台 / 淘宝联盟 API / PDD API
当前实现：mock stub + Claude 网络搜索（作为补充）
"""

from __future__ import annotations

from typing import Any

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only in minimal environments
    anthropic = None

from shopping_agent.common.constants import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from shopping_agent.common.exceptions import ToolUnavailableError
from shopping_agent.tools.base import BaseTool, with_retry


class SearchTool(BaseTool):
    tool_name = "search"

    def __init__(self):
        self._client = anthropic.Anthropic() if anthropic is not None else None

    @with_retry
    def execute(self, query: str, category: str = "",
                max_price: float = None, platform: str = "JD") -> dict[str, Any]:
        """
        搜索商品。生产环境替换为真实 API 调用。
        此处使用 Claude 的 web_search 工具作为演示。
        """
        if self._client is None:
            raise ToolUnavailableError("anthropic package is not installed")

        search_query = f"site:jd.com OR site:tmall.com {category} {query}"
        if max_price:
            search_query += f" 价格不超过{max_price}元"

        response = self._client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=DEFAULT_MAX_TOKENS,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"搜索商品信息：{search_query}，返回前5个结果的商品名称、价格、评分。"
            }],
        )

        # 提取搜索结果（简化处理）
        result_text = ""
        for block in response.content:
            if block.type == "text":
                result_text = block.text
                break

        return {
            "tool": self.tool_name,
            "query": query,
            "platform": platform,
            "raw_results": result_text,
            "status": "ok",
        }
