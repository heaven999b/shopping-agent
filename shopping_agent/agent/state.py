"""
AgentState — 会话全局状态，贯穿整个任务执行闭环。

所有模块读写状态时应通过此对象，不应各自维护独立状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from shopping_agent.common.types import (
    CandidatePlan,
    ClarificationQuestion,
    FeedbackRecord,
    PlanWorkspace,
    Product,
    ShoppingTask,
    UserProfile,
    VerificationReport,
)


class WorkflowStep(str, Enum):
    """主流程的步骤标记，用于记录当前执行位置与错误恢复。"""
    INIT = "init"
    PARSE_INTENT = "parse_intent"
    LOAD_MEMORY = "load_memory"
    CLARIFY = "clarify"
    RETRIEVE = "retrieve"
    BUILD_GRAPH = "build_graph"
    PLAN = "plan"
    VERIFY = "verify"
    RESPOND = "respond"
    EXECUTE = "execute"
    LEARN = "learn"
    DONE = "done"
    ERROR = "error"


@dataclass
class ConversationTurn:
    """单轮对话记录。"""
    round_index: int
    user_input: str
    agent_response: str
    clarification_asked: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AttributionEntry:
    """
    单步决策的归因记录。
    每个模块执行时向 attribution_trace 追加一条，用于事后分析"是哪步出了问题"。
    """
    step: WorkflowStep
    module: str                       # e.g. "IntentParser", "Planner"
    decision: str                     # 做了什么决定
    rationale: str                    # 为什么这么决定
    inputs_summary: str               # 关键输入摘要
    outputs_summary: str              # 关键输出摘要
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """
    会话级全局状态。

    - 由 Orchestrator 初始化并传递给每个模块
    - 模块只读取/追加，不应替换整个 state 对象
    """
    session_id: str
    user_id: str

    # 当前任务
    task: Optional[ShoppingTask] = None

    # 用户画像（从 Memory 层加载）
    user_profile: Optional[UserProfile] = None

    # 当前执行步骤
    current_step: WorkflowStep = WorkflowStep.INIT

    # 多轮对话历史
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    current_round: int = 0

    # 澄清状态
    pending_clarifications: list[ClarificationQuestion] = field(default_factory=list)
    clarification_rounds_used: int = 0

    # 检索结果（原始候选池）
    retrieved_products: list[Product] = field(default_factory=list)

    # 候选图（由 CandidateGraphBuilder 填充）
    # 使用 dict 表示图的邻接信息，key=product_id，value=相关节点列表
    candidate_graph: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # 规划结果
    candidate_plans: list[CandidatePlan] = field(default_factory=list)
    selected_plan: Optional[CandidatePlan] = None
    current_workspace: Optional[PlanWorkspace] = None

    # 校验结果
    verification_reports: list[VerificationReport] = field(default_factory=list)

    # 已执行的工具调用记录
    tool_execution_log: list[dict[str, Any]] = field(default_factory=list)

    # 反馈记录（本会话内收集）
    feedback_records: list[FeedbackRecord] = field(default_factory=list)

    # 归因追踪（每个模块追加，用于学习层分析）
    attribution_trace: list[AttributionEntry] = field(default_factory=list)

    # 错误记录
    errors: list[dict[str, Any]] = field(default_factory=list)

    # 会话时间
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # ---------------------------------------------------------------------------
    # 便捷方法
    # ---------------------------------------------------------------------------

    def add_turn(self, user_input: str, agent_response: str,
                 clarification_asked: Optional[str] = None) -> None:
        turn = ConversationTurn(
            round_index=self.current_round,
            user_input=user_input,
            agent_response=agent_response,
            clarification_asked=clarification_asked,
        )
        self.conversation_history.append(turn)
        self.current_round += 1
        self.updated_at = datetime.now()

    def record_attribution(
        self,
        module: str,
        decision: str,
        rationale: str,
        inputs_summary: str = "",
        outputs_summary: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        entry = AttributionEntry(
            step=self.current_step,
            module=module,
            decision=decision,
            rationale=rationale,
            inputs_summary=inputs_summary,
            outputs_summary=outputs_summary,
            duration_ms=duration_ms,
        )
        self.attribution_trace.append(entry)

    def record_error(self, module: str, error: Exception) -> None:
        self.errors.append({
            "step": self.current_step.value,
            "module": module,
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
        })

    def transition(self, next_step: WorkflowStep) -> None:
        self.current_step = next_step
        self.updated_at = datetime.now()

    def get_last_user_input(self) -> Optional[str]:
        if self.conversation_history:
            return self.conversation_history[-1].user_input
        return None

    def is_done(self) -> bool:
        return self.current_step in (WorkflowStep.DONE, WorkflowStep.ERROR)
