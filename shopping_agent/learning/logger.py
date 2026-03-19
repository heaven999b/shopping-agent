"""
BehaviorLogger — 行为日志记录器。

记录所有用户行为信号，用于：
  1. 事后归因分析（哪一步出了问题）
  2. 偏好更新的输入
  3. 离线评估与 A/B 实验

生产环境：写入 Kafka/数据库，此处为 in-memory stub。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from shopping_agent.common.types import FeedbackRecord


class BehaviorLogger:
    def __init__(self):
        self._logs: list[dict[str, Any]] = []

    def log(self, record: FeedbackRecord) -> None:
        """记录一条反馈信号。"""
        entry = {
            "session_id": record.session_id,
            "task_id": record.task_id,
            "plan_id": record.plan_id,
            "signal": record.signal.value,
            "context": record.context,
            "attribution_steps": [
                {
                    "module": a.get("module", ""),
                    "decision": a.get("decision", ""),
                }
                for a in record.attribution_trace[:5]  # 只记前 5 步
            ],
            "timestamp": record.timestamp.isoformat(),
        }
        self._logs.append(entry)

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        inputs: dict,
        outputs: dict,
        duration_ms: float,
        success: bool,
    ) -> None:
        """记录工具调用。"""
        self._logs.append({
            "type": "tool_call",
            "session_id": session_id,
            "tool_name": tool_name,
            "inputs_summary": str(inputs)[:200],
            "outputs_summary": str(outputs)[:200],
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

    def get_session_logs(self, session_id: str) -> list[dict]:
        return [log for log in self._logs if log.get("session_id") == session_id]

    def export_jsonl(self) -> str:
        """导出为 JSONL 格式（用于离线分析）。"""
        return "\n".join(json.dumps(log, ensure_ascii=False) for log in self._logs)
