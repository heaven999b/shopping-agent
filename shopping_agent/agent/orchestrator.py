"""
Orchestrator — 主流程协调器。

负责按顺序调度各模块，维护 AgentState，处理异常与重试。
实现两条闭环：
  1. 任务执行闭环（理解 → 澄清 → 检索 → 规划 → 校验 → 执行 → 反馈）
  2. 持续学习闭环（反馈 → 归因 → 偏好更新）

Option A 扩展：
  - enable_rl_mode() 将 clarification_policy 和 planner 切换到 RL 模式
  - train_rl() 触发 RLTrainer 的完整训练循环
  - 训练完成后策略自动注入 clarification_policy 和 planner
"""

from __future__ import annotations

import time
import uuid
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shopping_agent.rl.trainer import RLTrainer

from shopping_agent.agent.state import AgentState, WorkflowStep
from shopping_agent.common.constants import CLARIFICATION_MAX_ROUNDS
from shopping_agent.common.exceptions import (
    InfeasibleConstraintError,
    IntentParseError,
    NoProductFoundError,
    PlanningError,
    ShopPlanError,
    VerificationError,
)
from shopping_agent.common.types import (
    CandidatePlan,
    FeedbackRecord,
    FeedbackSignal,
)
from shopping_agent.interaction.clarification import ClarificationPolicy
from shopping_agent.interaction.intent_parser import IntentParser
from shopping_agent.learning.logger import BehaviorLogger
from shopping_agent.learning.preference_updater import PreferenceUpdater
from shopping_agent.memory.preference_memory import PreferenceMemory
from shopping_agent.planning.constraint_relaxer import ConstraintRelaxer
from shopping_agent.planning.explainer import Explainer
from shopping_agent.planning.planner import ConstraintAwarePlanner
from shopping_agent.product.candidate_graph import CandidateGraphBuilder
from shopping_agent.product.normalizer import ProductNormalizer
from shopping_agent.retrieval.retriever import HybridRetriever
from shopping_agent.verifier.pipeline import VerificationPipeline


class ShoppingAgentOrchestrator:
    """
    主协调器。

    使用方式：
        orchestrator = ShoppingAgentOrchestrator()
        response = orchestrator.run(user_id="u123", user_input="帮我配一套桌搭，预算5000")

        # 多轮对话
        response2 = orchestrator.continue_session(
            session_id=response["session_id"],
            user_input="预算可以放宽到6000"
        )
    """

    def __init__(self, use_rl: bool = False):
        # 初始化各模块
        self.intent_parser = IntentParser()
        self.clarification_policy = ClarificationPolicy(use_rl=use_rl)
        self.preference_memory = PreferenceMemory()
        self.retriever = HybridRetriever()
        self.normalizer = ProductNormalizer()
        self.graph_builder = CandidateGraphBuilder()
        self.planner = ConstraintAwarePlanner(use_rl=use_rl)
        self.verifier = VerificationPipeline()
        self.explainer = Explainer()
        self.behavior_logger = BehaviorLogger()
        self.preference_updater = PreferenceUpdater()
        self.constraint_relaxer = ConstraintRelaxer()

        # 会话存储（生产环境应替换为 Redis/DB）
        self._sessions: dict[str, AgentState] = {}
        self._use_rl = use_rl

    # ---------------------------------------------------------------------------
    # 公开接口
    # ---------------------------------------------------------------------------

    def run(self, user_id: str, user_input: str) -> dict:
        """
        开始一次新的购物任务。
        返回 agent 的回复与会话 ID。
        """
        session_id = str(uuid.uuid4())
        state = AgentState(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = state

        response = self._execute_workflow(state, user_input)
        return {"session_id": session_id, **response}

    def continue_session(self, session_id: str, user_input: str) -> dict:
        """
        在已有会话中继续对话（用户修正约束、回答澄清问题等）。
        """
        state = self._sessions.get(session_id)
        if state is None:
            raise ValueError(f"Session {session_id} not found.")

        response = self._execute_workflow(state, user_input)
        return {"session_id": session_id, **response}

    def record_feedback(self, session_id: str, signal: FeedbackSignal,
                        plan_id: Optional[str] = None, context: dict = None) -> None:
        """
        记录用户的显式/隐式反馈，触发学习闭环。
        """
        state = self._sessions.get(session_id)
        if state is None:
            return

        record = FeedbackRecord(
            session_id=session_id,
            task_id=state.task.task_id if state.task else "",
            plan_id=plan_id,
            signal=signal,
            context=context or {},
            attribution_trace=state.attribution_trace.copy(),
        )
        state.feedback_records.append(record)
        self.behavior_logger.log(record)
        self.preference_updater.update(state.user_id, record)

    # ---------------------------------------------------------------------------
    # RL 训练接口（Option A）
    # ---------------------------------------------------------------------------

    def enable_rl_mode(self) -> None:
        """
        将 clarification_policy 和 planner 切换到 RL 模式。
        可在运行时调用，无需重新初始化 orchestrator。
        """
        self._use_rl = True
        self.clarification_policy.set_rl_mode(True)
        self.planner.set_rl_mode(True)

    def train_rl(
        self,
        num_iterations: int = 200,
        episodes_per_iter: int = 32,
        eval_interval: int = 20,
        checkpoint_dir: str = "checkpoints",
    ) -> list[dict]:
        """
        启动 REINFORCE 训练循环。

        训练完成后，训练好的策略会自动注入 clarification_policy 和 planner，
        后续调用 run() 时自动使用 RL 策略（RL 模式已开启）。

        返回训练日志（每次迭代的 metrics）。
        """
        from shopping_agent.rl.trainer import RLTrainer

        trainer = RLTrainer(
            clarification_policy=self.clarification_policy,
            planner=self.planner,
            checkpoint_dir=checkpoint_dir,
        )
        logs = trainer.train(
            num_iterations=num_iterations,
            episodes_per_iter=episodes_per_iter,
            eval_interval=eval_interval,
        )
        # 训练后切换到 RL 模式（策略已通过 trainer 更新）
        self.enable_rl_mode()
        return logs

    # ---------------------------------------------------------------------------
    # 核心工作流
    # ---------------------------------------------------------------------------

    def _execute_workflow(self, state: AgentState, user_input: str) -> dict:
        """
        执行一轮任务工作流。
        根据当前 state 决定从哪个步骤开始（支持多轮对话续接）。
        """
        try:
            # Step 1: 解析意图
            state.transition(WorkflowStep.PARSE_INTENT)
            self._step_parse_intent(state, user_input)

            # Step 2: 加载用户记忆
            state.transition(WorkflowStep.LOAD_MEMORY)
            self._step_load_memory(state)

            # Step 3: 自适应澄清
            state.transition(WorkflowStep.CLARIFY)
            clarification_response = self._step_clarify(state)
            if clarification_response:
                # 本轮以澄清问题结束，等待用户回答
                return self._build_response(state, clarification_response, needs_input=True)

            # Step 4: 检索商品
            state.transition(WorkflowStep.RETRIEVE)
            self._step_retrieve(state)

            # Step 5: 构建候选图
            state.transition(WorkflowStep.BUILD_GRAPH)
            self._step_build_graph(state)

            # Step 6: 规划方案
            state.transition(WorkflowStep.PLAN)
            self._step_plan(state)

            # Step 7: 校验方案
            state.transition(WorkflowStep.VERIFY)
            self._step_verify(state)

            # Step 8: 生成回复
            state.transition(WorkflowStep.RESPOND)
            response_text = self._step_respond(state)

            state.transition(WorkflowStep.DONE)
            state.add_turn(user_input, response_text)
            return self._build_response(state, response_text)

        except (IntentParseError, NoProductFoundError, InfeasibleConstraintError) as e:
            state.record_error("Orchestrator", e)
            state.transition(WorkflowStep.ERROR)
            error_msg = self._handle_graceful_error(state, e)
            state.add_turn(user_input, error_msg)
            return self._build_response(state, error_msg, is_error=True)

        except ShopPlanError as e:
            state.record_error("Orchestrator", e)
            state.transition(WorkflowStep.ERROR)
            msg = f"抱歉，处理您的请求时遇到问题：{str(e)}"
            state.add_turn(user_input, msg)
            return self._build_response(state, msg, is_error=True)

    # ---------------------------------------------------------------------------
    # 各步骤实现
    # ---------------------------------------------------------------------------

    def _step_parse_intent(self, state: AgentState, user_input: str) -> None:
        t0 = time.time()

        if state.task is None:
            # 首轮：全新解析
            task = self.intent_parser.parse(user_input, state.user_profile)
        else:
            # 续轮：基于现有任务做增量更新（用户修正约束）
            task = self.intent_parser.parse_revision(
                user_input, state.task, state.conversation_history
            )

        state.task = task
        state.record_attribution(
            module="IntentParser",
            decision=f"解析为 {task.task_type.value} 任务，品类: {task.categories}",
            rationale="基于用户输入和对话历史",
            inputs_summary=user_input[:100],
            outputs_summary=f"uncertainty_score={task.uncertainty_score:.2f}, "
                            f"conflicts={len(task.conflict_pairs)}",
            duration_ms=(time.time() - t0) * 1000,
        )

    def _step_load_memory(self, state: AgentState) -> None:
        t0 = time.time()
        profile = self.preference_memory.load(state.user_id, state.task)
        state.user_profile = profile
        state.record_attribution(
            module="PreferenceMemory",
            decision="加载用户画像",
            rationale="为后续检索和规划提供个性化偏好",
            outputs_summary=f"brand_weights={list(profile.brand_weights.keys())[:3]}, "
                            f"price_sensitivity={profile.price_sensitivity:.2f}",
            duration_ms=(time.time() - t0) * 1000,
        )

    def _step_clarify(self, state: AgentState) -> Optional[str]:
        """
        判断是否需要澄清，如需要则生成问题并返回问题文本。
        返回 None 表示无需澄清，继续流程。

        RL 模式：调用 decide_rl() 单步决策（PROCEED 或 ASK_SLOT）。
        启发式模式：原有逻辑，按信息增益排序后取最高分问题。
        """
        if state.clarification_rounds_used >= CLARIFICATION_MAX_ROUNDS:
            return None
        if state.task.uncertainty_score < 0.3:
            return None

        if self._use_rl:
            question, action, log_prob = self.clarification_policy.decide_rl(
                task=state.task,
                conversation_round=state.clarification_rounds_used,
                user_profile=state.user_profile,
                explore=False,  # 推理时不探索
            )
            if question is None:
                # RL 策略决定 PROCEED
                state.record_attribution(
                    module="ClarificationPolicy(RL)",
                    decision="PROCEED — RL策略决定不再澄清",
                    rationale=f"action=PROCEED, log_prob={log_prob:.3f}",
                )
                return None

            state.pending_clarifications = [question]
            state.clarification_rounds_used += 1
            state.record_attribution(
                module="ClarificationPolicy(RL)",
                decision=f"ASK_SLOT: {question.slot}",
                rationale=f"RL策略决策, log_prob={log_prob:.3f}",
                outputs_summary=question.question,
            )
            return question.question

        # 启发式模式（原有逻辑）
        questions = self.clarification_policy.generate(
            state.task, state.user_profile
        )
        if not questions:
            return None

        best_question = max(questions, key=lambda q: q.info_gain)
        state.pending_clarifications = [best_question]
        state.clarification_rounds_used += 1

        state.record_attribution(
            module="ClarificationPolicy",
            decision=f"提问槽位: {best_question.slot}",
            rationale=f"info_gain={best_question.info_gain:.2f}, "
                      f"uncertainty_score={state.task.uncertainty_score:.2f}",
            outputs_summary=best_question.question,
        )
        return best_question.question

    def _step_retrieve(self, state: AgentState) -> None:
        t0 = time.time()
        products = self.retriever.retrieve(state.task, state.user_profile)
        # 标准化
        normalized = [self.normalizer.normalize(p) for p in products]
        state.retrieved_products = normalized
        state.record_attribution(
            module="HybridRetriever",
            decision=f"召回 {len(normalized)} 个商品",
            rationale="多通道检索 + 标准化",
            duration_ms=(time.time() - t0) * 1000,
        )

    def _step_build_graph(self, state: AgentState) -> None:
        t0 = time.time()
        graph = self.graph_builder.build(state.task, state.retrieved_products)
        state.candidate_graph = graph
        state.record_attribution(
            module="CandidateGraphBuilder",
            decision=f"构建候选图，节点数: {len(graph)}",
            rationale="建立替代/互补/兼容关系",
            duration_ms=(time.time() - t0) * 1000,
        )

    def _step_plan(self, state: AgentState) -> None:
        t0 = time.time()
        plans = self.planner.plan(
            task=state.task,
            candidate_graph=state.candidate_graph,
            user_profile=state.user_profile,
            retrieved_products=state.retrieved_products,
        )
        state.candidate_plans = plans
        state.record_attribution(
            module="ConstraintAwarePlanner",
            decision=f"生成 {len(plans)} 套候选方案",
            rationale="约束感知多目标优化",
            duration_ms=(time.time() - t0) * 1000,
        )

    def _step_verify(self, state: AgentState) -> None:
        t0 = time.time()
        valid_plans = []
        reports = []

        for plan in state.candidate_plans:
            report = self.verifier.verify(plan, state.task)
            reports.append(report)
            if report.passed:
                valid_plans.append(plan)

        # ----------------------------------------------------------------
        # Graph-based recovery: 当所有方案均被阻断时，
        # 尝试用候选图中的 SUBSTITUTE 边替换违规商品，生成修复方案
        # ----------------------------------------------------------------
        if not valid_plans and state.candidate_graph and state.retrieved_products:
            from shopping_agent.product.candidate_graph import GraphBasedRecovery
            recovery = GraphBasedRecovery(max_swaps=2)
            product_index = recovery.build_product_index(state.retrieved_products)
            recovered: list = []

            for plan, report in zip(state.candidate_plans, reports):
                if report.passed:
                    continue
                # 提取阻断原因（取第一个 BLOCK 的关键词）
                block_msgs = [r.message for r in report.results
                              if r.status.value == "block"]
                block_reason = (
                    "budget" if any("预算" in m or "超出" in m for m in block_msgs)
                    else "constraint" if any("库存" in m or "配送" in m for m in block_msgs)
                    else "budget"  # 默认尝试预算修复
                )
                repaired = recovery.try_repair(
                    plan, state.task, state.candidate_graph, product_index, block_reason
                )
                if repaired:
                    re_report = self.verifier.verify(repaired, state.task)
                    if re_report.passed:
                        recovered.append(repaired)
                        reports.append(re_report)

            if recovered:
                valid_plans = recovered
                state.record_attribution(
                    module="GraphBasedRecovery",
                    decision=f"图修复成功 {len(recovered)} 套方案",
                    rationale="通过 SUBSTITUTE 边替换违规商品",
                )

        state.verification_reports = reports
        state.candidate_plans = valid_plans

        if not valid_plans:
            # ── 约束松弛诊断：在抛出错误前尝试分析并记录松弛建议 ──
            last_report = reports[-1] if reports else None
            relax_result = self.constraint_relaxer.diagnose(state.task, last_report)
            state.record_attribution(
                module="ConstraintRelaxer",
                decision=f"诊断到 {len(relax_result.options)} 个松弛选项",
                rationale="; ".join(relax_result.infeasible_reasons[:2]),
            )
            # 若可自动松弛（仅软约束），静默执行并重试一次规划
            if relax_result.can_auto_relax and relax_result.recommended:
                relaxed_task = relax_result.apply_recommended(state.task)
                if relaxed_task is not None:
                    state.task = relaxed_task
                    state.record_attribution(
                        module="ConstraintRelaxer",
                        decision=f"自动松弛: {relax_result.recommended.description}",
                        rationale="软约束放弃，无需用户确认",
                    )
                    # 以松弛后的约束重新校验原始方案
                    for plan in state.candidate_plans or []:
                        re_report = self.verifier.verify(plan, relaxed_task)
                        if re_report.passed:
                            state.candidate_plans = [plan]
                            state.selected_plan = plan
                            return
            # 将松弛建议嵌入错误消息
            relax_hint = (
                f"\n\n建议：{relax_result.recommended.description}"
                if relax_result.recommended else ""
            )
            raise VerificationError(
                f"所有候选方案均未通过校验。{relax_hint}"
            )

        # 选分数最高的方案作为主推
        state.selected_plan = max(valid_plans, key=lambda p: p.overall_score)
        state.record_attribution(
            module="VerificationPipeline",
            decision=f"通过校验: {len(valid_plans)}/{len(state.candidate_plans + reports)} 套方案",
            rationale="预算/约束/兼容性/时效/风险五重校验 + 图修复",
            duration_ms=(time.time() - t0) * 1000,
        )

    def _step_respond(self, state: AgentState) -> str:
        return self.explainer.explain(
            plan=state.selected_plan,
            task=state.task,
            alternative_plans=state.candidate_plans[1:],
            style_hint="brief" if state.current_round == 1 else "detailed",
        )

    # ---------------------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------------------

    def _handle_graceful_error(self, state: AgentState, error: Exception) -> str:
        if isinstance(error, IntentParseError):
            return "抱歉，我没能理解您的购物需求。能告诉我您想买什么，大概预算是多少吗？"

        if isinstance(error, (NoProductFoundError, InfeasibleConstraintError)):
            # 调用约束松弛器，给出具体的、可操作的建议
            if state.task:
                relax_result = self.constraint_relaxer.diagnose(state.task)
                return relax_result.user_message()
            return "在您的条件下没有找到合适的商品，能稍微放宽一下约束吗？比如预算或品牌。"

        if isinstance(error, VerificationError):
            # VerificationError 中可能已包含松弛建议
            return str(error)

        return f"遇到了一个问题：{str(error)}"

    def _build_response(self, state: AgentState, message: str,
                        needs_input: bool = False, is_error: bool = False) -> dict:
        result = {
            "message": message,
            "needs_input": needs_input,
            "is_error": is_error,
            "current_step": state.current_step.value,
            "round": state.current_round,
        }
        if state.selected_plan:
            result["plan"] = {
                "plan_id": state.selected_plan.plan_id,
                "net_price": state.selected_plan.net_price,
                "overall_score": round(state.selected_plan.overall_score, 3),
                "items": [
                    {"slot": item.bundle_slot, "product": item.product.title,
                     "price": item.product.final_price}
                    for item in state.selected_plan.items
                ],
            }
        return result
