"""
SHOP-PLAN Shopping Agent — 命令行入口

用法：
  python main.py
  python main.py --user-id "user_001"

环境变量：
  ANTHROPIC_API_KEY  必须设置
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from shopping_agent.agent.orchestrator import ShoppingAgentOrchestrator
from shopping_agent.common.types import FeedbackSignal


def main():
    parser = argparse.ArgumentParser(description="SHOP-PLAN Shopping Agent")
    parser.add_argument("--user-id", default=None, help="用户 ID（默认随机生成）")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("错误：请设置环境变量 ANTHROPIC_API_KEY")
        sys.exit(1)

    user_id = args.user_id or f"user_{uuid.uuid4().hex[:8]}"
    orchestrator = ShoppingAgentOrchestrator()
    session_id = None

    print("=" * 60)
    print("  SHOP-PLAN 购物智能体")
    print("  输入购物需求，输入 'quit' 退出，输入 'feedback' 给出反馈")
    print("=" * 60)
    print(f"用户ID: {user_id}\n")

    while True:
        try:
            user_input = input("您: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower() == "feedback" and session_id:
            _handle_feedback(orchestrator, session_id)
            continue

        try:
            if session_id is None:
                result = orchestrator.run(user_id=user_id, user_input=user_input)
                session_id = result["session_id"]
            else:
                result = orchestrator.continue_session(
                    session_id=session_id, user_input=user_input
                )

            print(f"\nAgent: {result['message']}\n")

            if result.get("plan"):
                plan = result["plan"]
                print(f"  [方案 {plan['plan_id'][:8]}] 总价: ¥{plan['net_price']:.2f} "
                      f"得分: {plan['overall_score']:.0%}")
                for item in plan["items"]:
                    print(f"  - {item['slot']}: {item['product']} ¥{item['price']:.2f}")
                print()

            # 任务完成后重置 session，允许开始新任务
            if result.get("current_step") == "done" and not result.get("needs_input"):
                print("─" * 40)
                print("本次任务已完成。您可以开始新的购物需求。")
                print("─" * 40)
                session_id = None

        except Exception as e:
            print(f"\n[系统错误] {e}\n")


def _handle_feedback(orchestrator: ShoppingAgentOrchestrator, session_id: str):
    print("\n请选择反馈类型：")
    print("  1. 满意（加购/下单）")
    print("  2. 不满意（拒绝推荐）")
    print("  3. 取消")

    choice = input("选择(1/2/3): ").strip()
    if choice == "1":
        orchestrator.record_feedback(session_id, FeedbackSignal.EXPLICIT_POSITIVE)
        print("感谢反馈！已记录正向信号。\n")
    elif choice == "2":
        orchestrator.record_feedback(session_id, FeedbackSignal.EXPLICIT_NEGATIVE)
        print("感谢反馈！已记录负向信号，下次推荐会调整。\n")
    else:
        print("已取消。\n")


if __name__ == "__main__":
    main()
