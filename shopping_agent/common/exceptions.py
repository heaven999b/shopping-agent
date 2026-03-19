"""Custom exceptions for SHOP-PLAN Shopping Agent."""


class ShopPlanError(Exception):
    """所有自定义异常的基类。"""


# ---------------------------------------------------------------------------
# 任务 / 意图解析
# ---------------------------------------------------------------------------

class IntentParseError(ShopPlanError):
    """无法从用户输入中解析出有效的购物意图。"""


class ConstraintConflictError(ShopPlanError):
    """检测到无法自动解决的约束冲突，需要用户介入。"""


# ---------------------------------------------------------------------------
# 检索 / 商品
# ---------------------------------------------------------------------------

class RetrievalError(ShopPlanError):
    """商品检索失败（网络、API 限流等）。"""


class NoProductFoundError(ShopPlanError):
    """在给定约束下没有找到任何候选商品。"""


class ProductNormalizationError(ShopPlanError):
    """商品数据标准化失败。"""


class StaleProductDataError(ShopPlanError):
    """商品数据已超过 TTL，需要重新拉取。"""
    def __init__(self, product_id: str, message: str = ""):
        self.product_id = product_id
        super().__init__(message or f"Product {product_id} data is stale.")


# ---------------------------------------------------------------------------
# 规划
# ---------------------------------------------------------------------------

class PlanningError(ShopPlanError):
    """规划器无法在给定约束下生成有效方案。"""


class InfeasibleConstraintError(ShopPlanError):
    """约束组合不可满足（如预算过低无法购买任何符合条件的商品）。"""


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

class VerificationError(ShopPlanError):
    """方案校验失败，不应该执行。"""


class BudgetExceededError(VerificationError):
    """方案总价超出预算。"""
    def __init__(self, total: float, budget: float):
        self.total = total
        self.budget = budget
        super().__init__(f"Plan total {total:.2f} exceeds budget {budget:.2f}.")


class IncompatibilityError(VerificationError):
    """方案中存在不兼容的商品组合。"""
    def __init__(self, product_a: str, product_b: str, reason: str):
        self.product_a = product_a
        self.product_b = product_b
        super().__init__(f"{product_a} and {product_b} are incompatible: {reason}")


# ---------------------------------------------------------------------------
# 工具执行
# ---------------------------------------------------------------------------

class ToolExecutionError(ShopPlanError):
    """工具调用失败。"""
    def __init__(self, tool_name: str, reason: str, retryable: bool = True):
        self.tool_name = tool_name
        self.retryable = retryable
        super().__init__(f"Tool '{tool_name}' failed: {reason}")


class ToolRateLimitError(ToolExecutionError):
    """工具 API 限流，需要切换备用源或等待重试。"""
    def __init__(self, tool_name: str):
        super().__init__(tool_name, "rate limited", retryable=True)


class ToolUnavailableError(ToolExecutionError):
    """工具服务不可用（如下单接口挂了），需要降级处理。"""
    def __init__(self, tool_name: str):
        super().__init__(tool_name, "service unavailable", retryable=False)


# ---------------------------------------------------------------------------
# 高风险动作
# ---------------------------------------------------------------------------

class HighRiskActionError(ShopPlanError):
    """尝试执行高风险动作（如大额下单）时未经用户确认。"""
    def __init__(self, action: str, reason: str):
        self.action = action
        super().__init__(f"High-risk action '{action}' blocked: {reason}")


# ---------------------------------------------------------------------------
# 会话 / 记忆
# ---------------------------------------------------------------------------

class SessionNotFoundError(ShopPlanError):
    """找不到指定的会话。"""


class MemoryAccessError(ShopPlanError):
    """记忆层读写失败。"""
