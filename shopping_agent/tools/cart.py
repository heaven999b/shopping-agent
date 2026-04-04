"""
CartTool — 加购/下单工具。

高风险操作：金额超过阈值时必须经过 RiskChecker 确认后才能调用。
生产环境对接：电商平台开放 API（需 OAuth 授权）。
"""

from __future__ import annotations

from typing import Any

from shopping_agent.common.constants import HIGH_RISK_AMOUNT_THRESHOLD
from shopping_agent.common.exceptions import HighRiskActionError
from shopping_agent.common.types import CandidatePlan
from shopping_agent.tools.base import BaseTool, with_retry


class CartTool(BaseTool):
    tool_name = "cart"

    @with_retry
    def add_to_cart(self, product_id: str, quantity: int = 1,
                    platform: str = "JD") -> dict[str, Any]:
        """加购操作（stub）。"""
        # 生产环境：调用平台 API
        return {
            "tool": self.tool_name,
            "action": "add_to_cart",
            "product_id": product_id,
            "quantity": quantity,
            "platform": platform,
            "status": "ok",
            "cart_id": f"cart_{product_id[:8]}",
        }

    def place_order(
        self,
        plan: CandidatePlan,
        confirmed_by_user: bool = False,
    ) -> dict[str, Any]:
        """
        下单操作。金额超过阈值时必须有用户确认。
        """
        if plan.net_price > HIGH_RISK_AMOUNT_THRESHOLD and not confirmed_by_user:
            raise HighRiskActionError(
                action="place_order",
                reason=f"订单金额 ¥{plan.net_price:.2f} 超过 ¥{HIGH_RISK_AMOUNT_THRESHOLD}，"
                       f"需要用户明确确认后才能下单。",
            )

        # 生产环境：调用平台下单 API
        order_ids = []
        for item in plan.items:
            order_ids.append(f"order_{item.product.product_id[:8]}")

        return {
            "tool": self.tool_name,
            "action": "place_order",
            "plan_id": plan.plan_id,
            "order_ids": order_ids,
            "total_amount": plan.net_price,
            "status": "ok",
        }
