"""
可重复性辅助工具。

统一设置 Python / NumPy 随机种子，并暴露 deterministic 配置。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: Optional[int], deterministic: bool = True) -> None:
    """设置全局随机种子。"""
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        os.environ["SHOP_PLAN_DETERMINISTIC"] = "1"
    else:
        os.environ.pop("SHOP_PLAN_DETERMINISTIC", None)
