# SHOP-PLAN: Shopping Agent

**Structured Hierarchical Optimization and Planning for Shopping Agents**

面向复杂消费决策的约束感知、记忆增强、工具校验式智能体。

---

## 架构概览

```
七层架构 + 两条闭环 + 三个核心方法模块
```

### 七层架构

| 层级 | 名称 | 核心模块 |
|------|------|---------|
| [1] User Interaction | 多轮对话、澄清问题生成 | `interaction/intent_parser.py`, `interaction/clarification.py` |
| [2] User Modeling & Memory | 短期状态、长期偏好 | `memory/preference_memory.py` |
| [3] Product Knowledge & Retrieval | 多源召回、标准化、候选图 | `retrieval/`, `product/` |
| [4] Planning & Decision | 约束满足、多目标优化 | `planning/planner.py`, `planning/explainer.py` |
| [5] Tool Execution | 搜索、比价、加购、下单 | `tools/` |
| [6] Risk & Governance | 五重并行校验 | `verifier/pipeline.py` |
| [7] Learning & Evaluation | 归因、偏好更新 | `learning/` |

### 三个核心方法模块

1. **Adaptive Clarification Policy** — 信息增益驱动，只问最有价值的问题
2. **Candidate Graph Planner** — 先构建替代/互补/兼容候选图，再做约束感知规划
3. **Verifier-based Decision Guardrail** — 五重并行校验（预算/约束/兼容性/时效/风险）

### 两条闭环

```
任务执行闭环：
  理解需求 → 澄清 → 检索 → 构建候选图 → 规划 → 校验 → 执行 → 用户修正 → 重规划

持续学习闭环：
  用户反馈 → 归因分析 → 偏好更新 → 策略优化 → 下次任务更好
```

---

## 核心数据结构

```python
ShoppingTask     # 意图解析结果（含约束、不确定性、冲突检测）
UserProfile      # 用户长期偏好画像
Product          # 标准化商品对象（跨平台统一格式）
CandidatePlan    # 候选方案（含得分、trade-off、校验状态）
AgentState       # 会话全局状态（含归因追踪）
```

---

## 项目结构

```
shopping_agent/
├── agent/              # orchestrator, state
├── interaction/        # intent_parser, clarification
├── memory/             # preference_memory
├── retrieval/          # hybrid_retriever
├── product/            # normalizer, candidate_graph
├── planning/           # planner, explainer
├── verifier/           # pipeline（5个独立checker）
├── tools/              # search, cart（含重试/降级）
├── learning/           # logger, preference_updater
└── common/             # types, exceptions, constants
```

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 API Key
export ANTHROPIC_API_KEY=your_key_here

# 运行
python main.py

# 指定用户 ID（跨会话保持偏好记忆）
python main.py --user-id user_001
```

### 示例交互

```
您: 帮我配一套桌搭，预算5000，需要显示器、键盘、鼠标
Agent: 为您推荐以下方案（综合得分 87%）：
  显示器（Sony 27寸 4K...，¥2,199）、键盘（Logitech MX Keys...，¥699）...
  合计：¥3,298（预算 ¥5,000）

您: 预算可以放宽到8000，想要升级显示器
Agent: 好的，根据您的调整，为您重新规划...

您: feedback
请选择反馈类型：
  1. 满意（加购/下单）
  2. 不满意（拒绝推荐）
```

---

## 设计原则

- **接口契约优先**：`ShoppingTask` 是所有模块的共享契约，先定版再开发
- **校验独立可测**：每个 Verifier 独立，支持并行执行，结果分 PASS/WARN/BLOCK/STALE
- **归因可追溯**：`attribution_trace` 记录每步决策依据，失败可溯源
- **工具降级优雅**：网络超时重试，限流切换备用源，关键接口失败通知用户
- **冷启动友好**：新用户返回空白画像，不影响主流程

---

## 扩展方向

1. **多平台工具接入**：对接京东/淘宝/拼多多开放 API
2. **长期记忆持久化**：接入 Redis（短期）+ PostgreSQL（长期画像）
3. **在线策略更新**：Clarification Policy 的 info_gain 权重通过 A/B 实验在线更新
4. **候选图知识增强**：从共购数据挖掘互补关系，从规格数据库加载兼容性规则