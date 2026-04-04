"""
Tools 基类与重试装饰器。

所有工具调用必须继承 BaseTool，统一处理：
  - 指数退避重试（最多 MAX_TOOL_RETRIES 次）
  - 限流时切换备用数据源
  - 关键工具不可用时降级为"信息不完整"警告
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from shopping_agent.common.constants import MAX_TOOL_RETRIES, TOOL_RETRY_BASE_DELAY
from shopping_agent.common.exceptions import ToolExecutionError, ToolRateLimitError


def with_retry(func: Callable) -> Callable:
    """
    指数退避重试装饰器。
    非重试异常（ToolExecutionError.retryable=False）直接抛出。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        last_exc = None
        for attempt in range(MAX_TOOL_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except ToolRateLimitError as e:
                last_exc = e
                if attempt < MAX_TOOL_RETRIES:
                    delay = TOOL_RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
            except ToolExecutionError as e:
                if not e.retryable or attempt >= MAX_TOOL_RETRIES:
                    raise
                last_exc = e
                delay = TOOL_RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
        raise last_exc
    return wrapper


class BaseTool:
    """所有工具的基类。"""
    tool_name: str = "base_tool"

    def execute(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError
