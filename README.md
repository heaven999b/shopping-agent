# SHOP-PLAN: Shopping Agent

**Structured Hierarchical Optimization and Planning for Shopping Agents**

面向复杂消费决策的约束感知、POMDP 驱动、强化学习增强型购物智能体。

---

## 架构概览

```
七层架构 + 两条闭环 + 四个核心方法模块
```

### 七层架构

| 层级 | 名称 | 核心模块 |
|------|------|---------|
| [1] User Interaction | 多轮对话、澄清问题生成 | `interaction/intent_parser.py`, `interaction/clarification.py` |
| [2] User Modeling & Memory | 短期状态、长期偏好、Bayesian 信念状态 | `memory/preference_memory.py`, `rl/belief.py` |
| [3] Product Knowledge & Retrieval | 多源召回（关键词 + TF-IDF 向量）、标准化、候选图 | `retrieval/`, `product/` |
| [4] Planning & Decision | 约束感知规划、约束松弛、候选图修复 | `planning/planner.py`, `planning/constraint_relaxer.py` |
| [5] Tool Execution | 搜索、比价、加购、下单 | `tools/` |
| [6] Risk & Governance | 五重并行校验 + 自动约束松弛 | `verifier/pipeline.py` |
| [7] Learning & Evaluation | RL 训练、行为克隆、归因追踪 | `rl/`, `learning/` |

### 四个核心方法模块

1. **Bayesian Belief State** — 对用户偏好维护后验分布，通过 VoI（期望信息增益）驱动澄清决策，替代启发式信息熵估计
2. **Hybrid Retrieval** — 关键词精确过滤 + TF-IDF 字符级 n-gram 语义向量召回，覆盖中文近义词
3. **Actor-Critic RL（GAE）** — 澄清策略 π_θ 和规划策略 π_φ 用 GAE（Generalized Advantage Estimation）替代原始 REINFORCE，配套线性 Critic V(s) 大幅降低梯度方差
4. **Constraint Relaxation** — 在检索/规划失败时自动诊断约束冲突，生成排序松弛方案（软约束优先 → 预算扩增 → 属性降级）

### 两条闭环

```
任务执行闭环：
  理解需求 → Bayesian 信念更新 → VoI 驱动澄清 → 混合检索 → 候选图构建
  → 约束感知规划 → 五重校验 → 约束松弛重试 → 图修复 → 执行 → 反馈

持续学习闭环：
  用户反馈 → 归因分析 → 偏好更新 → Actor-Critic (GAE) 策略优化 → 行为克隆预热
```

---

## 核心数据结构

```python
ShoppingTask     # 意图解析结果（含约束、不确定性、冲突检测）
UserProfile      # 用户长期偏好画像（SQLite 持久化）
Product          # 标准化商品对象（跨平台统一格式，TTL 刷新）
CandidatePlan    # 候选方案（含得分、trade-off、校验状态）
AgentState       # 会话全局状态（含归因追踪，SessionStore 持久化）
BeliefState      # POMDP 信念状态（Bayesian 后验，VoI 驱动）
```

---

## 项目结构

```
shopping_agent/
├── agent/              # orchestrator, state
├── interaction/        # intent_parser（双层 LLM+规则兜底），clarification
├── memory/             # preference_memory（SQLite 持久化）
├── retrieval/          # hybrid_retriever（关键词 + TF-IDF 向量）
├── product/            # normalizer, candidate_graph（图修复）
├── planning/           # planner, explainer, constraint_relaxer
├── verifier/           # pipeline（5个独立checker）
├── tools/              # search, cart（含重试/降级）
├── learning/           # logger, preference_updater, trajectory_logger
├── rl/                 # POMDP, policy（Actor + Critic），belief，pretrain，trainer
├── evaluation/         # benchmark, metrics
├── storage/            # db（SQLite WAL），profile_store，session_store
├── data/               # loader，refresher（TTL 数据刷新）
└── common/             # types, exceptions, constants

data/
├── products.json       # 27 件示例商品（8 品类）
└── tasks.json          # 10 条基准测试任务

tests/                  # 61 个单元测试（pytest）
```

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 API Key
export ANTHROPIC_API_KEY=your_key_here

# 对话模式
python main.py chat --user-id user_001

# 基准测试（启发式 vs RL 对比）
python main.py benchmark --compare

# RL 行为克隆预训练（冷启动）
python main.py train --pretrain --synthetic

# 数据目录信息
python main.py catalog
```

### 示例交互

```
您: 帮我配一套桌搭，预算5000，需要显示器、键盘、鼠标
Agent: 为您推荐以下方案（综合得分 87%）：
  显示器（LG 27寸 4K，¥2,199）、键盘（Logitech MX Keys，¥699）...
  合计：¥3,298（预算 ¥5,000）

您: 预算可以放宽到8000，想要升级显示器
Agent: 好的，根据您的调整，为您重新规划...

# 约束过严时自动给出松弛建议：
您: 帮我买降噪耳机，预算300，要主动降噪和机械感
Agent: 抱歉，当前条件下暂无完全匹配的商品：
  • 在约束条件（预算=300，品类=['headset']）下未找到任何商品
  为您推荐以下调整方案：
  1. 不强制要求「RGB 灯效」功能（保留主动降噪）
  2. 预算上浮 10%（预算调整为 ¥330）
```

---

## 强化学习模块

### POMDP 形式化

```
Clarification POMDP:
  State  s  = BeliefState（用户偏好后验分布）
  Action a  ∈ {ASK_slot_i} ∪ {PROCEED}
  Obs    o  = 用户回答（带噪声观测）
  Belief update: b_{t+1}(p) ∝ P(o | p) · b_t(p)
  Reward R  = α·success - β·turns - γ·dropout_prob
  VoI(slot) = H(b_t) - E[H(b_{t+1} | ask slot)]

Planning POMDP:
  State  s  = (budget_used/budget, slot_fill_ratio, constraint_sat, ...)
  Action a  ∈ {SELECT_BEST, SELECT_BUDGET_OPT, SELECT_SAFE}
  Reward R  = constraint_score + preference_score + value_score
```

### Actor-Critic 训练（GAE）

```python
# 替代原始 REINFORCE 的低方差优势估计
A_t^GAE(γ,λ) = Σ_{l≥0} (γλ)^l · δ_{t+l}
where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

# λ=0.95（PPO 默认），平衡偏差-方差权衡
# Critic: L = (V(s_t) - G_t)^2，线性函数近似
```

```bash
# 行为克隆预训练 + RL 微调
python -m shopping_agent.rl.pretrain --synthetic --epochs 20
python main.py train --rl --iterations 200
```

### 基准测试与消融实验

```
Full model:    RL Clarification + RL Planning (GAE)
Ablation A:    Heuristic Clarification + RL Planning
Ablation B:    RL Clarification + Heuristic Planning
Ablation C:    REINFORCE（无 Critic）vs GAE（有 Critic）
Baseline:      Heuristic Clarification + Heuristic Planning
```

```bash
python run_benchmark.py --compare      # 输出消融对比表
python run_benchmark.py --rl --save    # 保存详细结果
```

---

## 设计原则

- **接口契约优先**：`ShoppingTask` 是所有模块的共享契约，先定版再开发
- **POMDP 可信**：Belief State 有完整 Bayesian 更新，VoI 驱动澄清而非直觉启发式
- **低方差梯度**：GAE + 线性 Critic，不依赖 GPU，支持小数据集训练
- **校验独立可测**：每个 Verifier 独立，支持并行执行，结果分 PASS/WARN/BLOCK/STALE
- **失败优雅降级**：约束冲突 → 自动诊断 → 松弛建议 → 图修复 → 用户协商
- **归因可追溯**：`attribution_trace` 记录每步决策依据，失败可溯源
- **存储可替换**：SQLite 接口与 PostgreSQL 一致，`BaseProductAPI` 可接入真实电商 API

---

## 扩展方向

1. **Dense Retrieval**：FAISS + bi-encoder（sentence-transformers）替换 TF-IDF，覆盖语义相似
2. **分层 RL**：Options Framework，macro-policy（是否换策略）+ micro-policy（具体选品）
3. **多平台工具接入**：对接京东/淘宝/拼多多开放 API（实现 `BaseProductAPI`）
4. **Contextual Bandit**：用户画像实时更新（LinUCB / Thompson Sampling）
5. **对话状态跟踪（DST）**：集成 TripPy 或 SimpleTOD 做标准化状态追踪
