# SHOP-PLAN: Shopping Agent

**Structured Hierarchical Optimization and Planning for Shopping Agents**

面向复杂消费决策的长期记忆增强型组合规划购物智能体。

当前项目的主线已经从“推荐单品”扩展到两条更明确的系统能力：

- **系统化推荐（Systematic Recommendation）**：不只给单个商品，而是输出组合级 `BundlePlan`、分阶段购买路线、行动清单和 trade-off 提示。
- **长期用户成长建模（Long-horizon User Growth Modeling）**：不只记录静态偏好，而是持续维护 `owned_items`、`active_setups`、`upgrade_stage`、`purchase_rhythm`、`persona drift` 等长期状态。
- **计划工作台（Plan Workspace）**：orchestrator 会把推荐组织为 `workspace + artifacts`，便于前端展示成可持续推进的计划视图，而不只是聊天文本。

---

## Core Contribution

This project focuses on one main contribution line:

1. **Bundle-level planning instead of single-item recommendation**
2. **Phased purchase and upgrade-path support**
3. **Long-horizon user memory for evolving preferences**

RL, POMDP-style belief tracking, retrieval, and constraint handling are treated as
supporting components that improve this main planning line, rather than separate claims.

## 核心卖点

1. **系统化购买建议**
   从单品推荐升级为组合级规划，支持预算分配、风格一致性、长期适配和“一步到位 / 分阶段升级 / 保守路线”三类购买路径。
2. **长期用户成长建模**
   用户画像不再只是静态偏好表，而是会维护拥有状态、升级位置、购买节奏、aspiration signals 和 persona drift。
3. **Persona-aware + Drift-aware 决策**
   检索、规划、解释和澄清会联合考虑身份表达、风格偏好、预算人格，以及用户在交互中的变化轨迹。
4. **计划化输出而非纯对话输出**
   系统会返回结构化 `workspace`，其中包含 `growth_snapshot`、`bundle_recommendation`、`phase_plan`、`action_checklist`、`tradeoff_notes` 等 artifact。
5. **可评测、可恢复、可演进**
   benchmark 已覆盖 bundle、drift、长期升级与 phased purchase；session/workspace 可持久化恢复，便于长期陪伴式场景。

## 架构概览

```
七层架构 + 两条闭环 + 一个主方法主线
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

### 四层评分友好分层

1. **Input Layer**
   `intent_parser`, `clarification`, `user state`
2. **Retrieval Layer**
   `retrieval`, `product normalization`, `candidate graph`
3. **Decision Layer**
   `bundle_planner`, `explainer`, `workspace artifacts`
4. **Verification Layer**
   `verifier`, `constraint relaxer`, `risk handling`

Each layer is modular and can be independently replaced or extended.

### 主方法主线

**Memory-Augmented Bundle Planning**

项目的核心方法线收束为一个主问题：

- 用 `BundlePlan` 而不是单品作为主要决策单位
- 用长期用户状态而不是一次性偏好做组合评分
- 用 phased purchase / upgrade path 让推荐跨时间展开

围绕这条主线，当前系统使用以下支持组件：

1. **Belief-guided Clarification**
   用 belief / uncertainty 表示澄清价值，减少不必要问题数。
2. **Hybrid Retrieval**
   用关键词 + TF-IDF 做候选召回。
3. **Constraint Relaxation**
   在预算或属性冲突时给出可执行的松弛路径。
4. **RL-enhanced Policy**
   用 Actor-Critic 训练澄清与规划策略，作为增强模块，而不是主贡献本身。

### 两条闭环

```
任务执行闭环：
  理解需求 → Bayesian 信念更新 → VoI 驱动澄清 → 混合检索 → 候选图构建
  → 约束感知规划 → 五重校验 → 约束松弛重试 → 图修复 → 执行 → 反馈

持续学习闭环：
  用户反馈 → 归因分析 → 偏好更新 → RL-enhanced policy 优化 → 行为克隆预热
```

---

## 核心数据结构

```python
ShoppingTask     # 意图解析结果（含约束、不确定性、冲突检测）
UserProfile      # 用户长期成长画像（SQLite 持久化）
Product          # 标准化商品对象（跨平台统一格式，TTL 刷新）
BundlePlan       # 组合级方案对象（含 slot coverage、compatibility、阶段升级路径）
CandidatePlan    # 兼容旧接口的基类
PlanWorkspace    # 计划工作台（artifact 化输出，支持生命周期流转）
AgentState       # 会话全局状态（含归因追踪、workspace，SessionStore 持久化）
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
├── planning/           # bundle_planner, planner, explainer, constraint_relaxer
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
└── tasks.json          # 66 条基准任务（其中 38 条 bundle 任务）

tests/                  # 93 个单元测试（pytest）
```

---

## 快速开始

```bash
# 运行时依赖（最小）
pip install -r requirements.txt

# 开发 / 测试依赖（推荐用于复现仓库结果）
pip install -r requirements-dev.txt

# 设置 API Key
export ANTHROPIC_API_KEY=your_key_here

# 对话模式
python main.py chat --user-id user_001

# 基准测试（启发式 vs RL-enhanced 对比）
python main.py benchmark --compare

# Benchmark 脚本
python run_benchmark.py --mode pipeline   # 模块级 benchmark
python run_benchmark.py --mode e2e        # 端到端 benchmark（走公开入口）
python run_benchmark.py --mode e2e --save # 保存完整报告 + summary + metrics + per_task
python run_benchmark.py --baseline-suite  # 运行 full / naive / constraint-only / no-memory / single-item / no_bundle_scoring 对比

# RL-enhanced 策略训练（增强模块）
python main.py train --pretrain --synthetic

# 数据目录信息
python main.py catalog

# 计划工作台操作
python main.py plan --session-id <id> --show
python main.py plan --session-id <id> --accept --route "分阶段升级"
python main.py plan --session-id <id> --advance-phase
python main.py plan --session-id <id> --complete

# 一键复现
bash run.sh
```

### 复现与依赖说明

- `requirements.txt`
  运行时最小依赖，适合 CLI 和核心逻辑。
- `requirements-dev.txt`
  开发与测试依赖，包含 `pytest` 和 `scikit-learn`，推荐用于 fresh env 复现。
- `pyproject.toml`
  提供统一项目元数据、可选依赖和 pytest 配置。

当前仓库支持两种检索模式：

- 安装了 `scikit-learn`：使用标准 TF-IDF 向量检索实现。
- 未安装 `scikit-learn`：自动降级到内置的轻量 TF-IDF 实现，基础测试仍可运行。

### 测试

```bash
# 推荐：fresh env 下完整复现
pip install -r requirements-dev.txt
pytest
```

当前仓库包含 `93` 个 pytest 单元测试，并通过 GitHub Actions 在多 Python 版本下执行。

### 复现闭环

```bash
# 1. 安装依赖
pip install -r requirements-dev.txt

# 2. 跑测试
pytest -q

# 3. 跑端到端 benchmark 并保存结果
python run_benchmark.py --mode e2e --save

# 4. 查看 logs/ 下的固定产物
# - benchmark_<mode>_<ts>.json
# - benchmark_<mode>_<ts>_summary.json
# - benchmark_<mode>_<ts>_metrics.json
# - benchmark_<mode>_<ts>_per_task.json
```

如果只想快速验证仓库是否“能跑通”，可以直接执行：

```bash
bash run.sh
```

### Benchmark 模式

- `pipeline`
  直接评估检索、规划、校验链路，适合做模块消融和稳定对比。
- `e2e`
  通过 `ShoppingAgentOrchestrator.run()/continue_session()` 驱动任务，包含意图解析和自动澄清回合，更接近真实 agent 行为。

当前 benchmark 除了成功率、覆盖率、预算满足率外，还会输出更细的诊断指标：

- 解析品类匹配率 `avg_parser_category_match`
- 最终方案品类匹配率 `avg_plan_category_match`
- 澄清预期对齐率 `clarification_alignment_rate`
- 可行性判断对齐率 `feasibility_alignment_rate`
- 意图收敛分 `avg_intent_resolution_score`
- 执行就绪分 `avg_execution_readiness_score`
- 方案画像匹配均值 `avg_plan_persona_alignment_score`
- 单品画像理由覆盖率 `avg_persona_reason_coverage`
- 风格统一分 `avg_style_coherence_score`
- 组合完整度 `avg_bundle_completeness_score`
- 组合兼容分 `avg_compatibility_score`
- bundle 原生决策分 `avg_bundle_decision_score`
- 长期适配分 `avg_long_term_fit_score`
- 分阶段购买分 `avg_phased_purchase_score`
- drift 检测率 `drift_detection_rate`
- drift 对齐分 `avg_drift_alignment_score`
- 失败桶分布 `failure_bucket_breakdown`
- 任务家族分层汇总 `task_family_summary`

其中：

- `intent_resolution_score`
  衡量任务理解是否收敛，综合解析品类匹配和澄清对齐。
- `execution_readiness_score`
  衡量方案是否真正可交付，综合最终品类匹配、可行性对齐和约束命中。

benchmark 导出报告现在带固定 schema：

- `report_schema_version`
- `metrics`
- `per_task`
- `report_sections.summary_metrics`
- `report_sections.task_family_summary`
- `report_sections.per_task_results`

如果使用 `--save`，当前会固定导出四类文件：

- `benchmark_<mode>_<ts>.json`
- `benchmark_<mode>_<ts>_summary.json`
- `benchmark_<mode>_<ts>_metrics.json`
- `benchmark_<mode>_<ts>_per_task.json`

仓库里也附带了一个静态样例：[examples/benchmark_report_sample.json](/Users/yihaiwen/Documents/New%20project/repo/examples/benchmark_report_sample.json)。

任务如果未显式声明 `task_family`，系统会按 `clarification_heavy`、`bundle`、`constraint_dense`、`drift`、`comparison`、`general` 自动归类。
默认 benchmark 任务集中现在也包含长期升级与分阶段购买样例，例如 `upgrade_path` 和 `phased_purchase`。
当前默认任务集包含 `66` 条任务，其中 `38` 条是 bundle 任务，并额外覆盖 phased purchase、upgrade path、clarification-heavy、drift、budget-edge 和 relation-sensitive 场景。

### Baselines

当前仓库支持一组评分友好的 baseline 对比：

- `full_agent`
  完整系统
- `naive_retrieval`
  近似只保留基础召回，不使用 graph / verifier / clarification
- `constraint_only`
  保留约束驱动，但去掉 persona 与长期记忆信号
- `no_memory`
  保留 bundle/planning/persona，但去掉长期用户成长状态（owned_items / active_setups / upgrade_stage / purchase_rhythm）
- `single_item`
  将多品类任务退化为单品推荐
- `no_bundle_scoring`
  保留多商品输出，但关闭 bundle-native 决策分，验证组合级打分本身的必要性

运行方式：

```bash
python run_benchmark.py --baseline-suite --save
```

保存后会额外生成：

- `baseline_suite.json`
- `baseline_suite.md`
- `baseline_suite.csv`

默认输出会形成一张对比表，重点观察：

- `success_rate`
- `bundle_success_rate`
- `avg_bundle_completeness_score`
- `avg_compatibility_score`
- `avg_bundle_decision_score`
- `avg_long_term_fit_score`
- `avg_phased_purchase_score`
- 预算压力近似值 `RegretRisk`

当前更推荐把 `bundle_success_rate` 作为 bundle 主表核心指标来读：

- `full_agent vs single_item`
  验证组合级规划本身是否必要
- `full_agent vs no_memory`
  验证长期用户成长建模是否真正提升 long-horizon 任务完成度
- `full_agent vs no_bundle_scoring`
  验证 bundle-native 决策分是否带来额外增益

其中 `baseline_suite.md / .csv` 会优先使用 `bundle_summary`，也就是 bundle / upgrade / phased_purchase 子集上的主表结果，而不是简单全任务平均。

### Recommended Research Question

如果按论文/评审视角来收束，当前最推荐的问题表述是：

> How can an agent perform long-horizon bundle planning under evolving user memory and phased purchase constraints?

在这个表述里：

- `BundlePlan` 是主决策对象
- 长期用户状态是主状态表示
- `RL-enhanced policy` 只负责优化澄清与规划动作

### 用户记忆

- 长期画像通过 SQLite 持久化。
- 隐式/显式交互信号会写入持久化信号表，用于后续画像演化。
- 上下文化加载时会参考近期交互，对相关品牌和品类做轻量偏置。
- 当前还支持最小 persona state：`identity_goal`、`budget_sensitivity_profile`、`brand_orientation`、`aesthetic_preference`、`persona_stability`。
- 画像还会记录最近一次 persona drift 和 transition log，用于表示用户在预算、风格、品牌取向上的变化轨迹。
- 长期成长状态现在会额外维护 `owned_items`、`active_setups`、`upgrade_stage`、`purchase_rhythm`、`aspiration_signals`，用于描述用户已经拥有什么、正处在哪个升级阶段，以及下一步更适合补什么。

### 系统化推荐与计划工作台

- `BundlePlan` 已经成为组合级方案对象，支持 `slot_coverage`、`compatibility_score`、`bundle_objective`、`budget_allocation`、`style_coherence_score`、`bundle_completeness_score`、`long_term_fit_score`、`phased_upgrade_plan`。
- `BundlePlan` 当前会显式计算 `bundle_decision_score`，不再只是多个 item 分数汇总；该分数会联合考虑 slot coverage、relation-based compatibility、style coherence、long-term fit 与 phased purchase。
- `CandidatePlan` 目前仍保留为兼容基类，方便已有 verifier / cart / explainer 链路逐步迁移。
- orchestrator 会返回结构化 `workspace`，而不只是自然语言回答。
- 当前 `workspace` 默认包含这些 artifact：
  - `growth_snapshot`
  - `bundle_recommendation`
  - `phase_plan`
  - `action_checklist`
  - `tradeoff_notes`
- `action_checklist` 会把方案拆成“当前该买什么 / 后续再补什么 / 可选升级项”。
- `workspace` 还支持最小生命周期流转：`active`、`accepted`、`completed`、`archived`。
- 当用户接受“分阶段升级”路线时，系统会额外生成 `next_phase_handoff`，为下一阶段推荐保留交接信息。
- `workspace` 会跟 session 一起持久化恢复，因此前端可以把它当作计划工作台，而不是一次性聊天结果。
- CLI 也支持直接操作 workspace：查看当前计划、接受方案、推进 phase、标记完成、归档。
- 一个最小闭环是：先通过 `chat` 生成 `session_id`，再用 `python main.py plan --session-id <id> --accept --route "分阶段升级"` 接受 Phase 1，之后用 `--advance-phase` 和 `--complete` 推进。

### Failure Analysis

当前系统最常见的失败类型包括：

1. **Ambiguous user intent**
   多轮澄清不足时，可能把单品需求误解为组合需求，或反之。
2. **Over-constrained budget**
   预算与属性要求同时过严时，会进入无可行方案或仅能松弛的状态。
3. **Missing product relations**
   当 catalog 中缺少互补/替代关系时，组合完整度会下降。
4. **Persona misinterpretation**
   当用户表达含糊或 drift 很快时，可能出现风格匹配偏差。
5. **Bundle incompleteness / incompatibility**
   当关系边不足或组合预算过紧时，可能出现 slot 未补齐或整体兼容性不足。

Future work will focus on stronger intent disambiguation, richer product relation modeling, and more robust long-horizon state tracking.

### Persona-aware 规划

- 商品数据支持 `persona_tags`；未显式标注时，catalog loader 会根据品牌、价格、颜色和品类推断最小 tags。
- 检索重排会引入 `persona_alignment_score`，优先保留更贴近用户形象与风格表达的候选。
- 规划器主入口现在是 `BundlePlanner`，输出的 `BundlePlan` 包含 `persona_alignment_score`、`persona_summary`，单个 `PlanItem` 也会带 `persona_reason`。
- 解释器会在详细说明里展示“为什么这套方案更贴近当前用户画像”，而不只给价格和评分。
- 对多品类任务，规划器会输出组合级字段：`bundle_type`、`bundle_objective`、`budget_allocation`、`style_coherence_score`、`scenario_fit_score`、`bundle_completeness_score`。
- 当前还支持最小版 phased purchase：方案里会附带“一步到位 / 分阶段升级 / 保守路线”三类购买路径建议。
- orchestrator 响应现在还会返回 `workspace`，其中按 artifact 组织为 `growth_snapshot`、`bundle_recommendation`、`phase_plan`、`action_checklist`、`tradeoff_notes`，便于前端做卡片式计划视图。

### Drift-aware 澄清

- 当 `persona_stability` 偏低时，澄清策略会额外考虑 `identity_goal`、`aesthetic_preference`、`budget_flexibility` 这类问题，而不只问硬约束缺口。
- 用户主动修正预算、风格、身份表达或品牌取向时，系统会把这些变化写入 drift log，并在后续上下文化加载中轻量反映出来。

### RL-Enhanced Policy 指标

RL 训练与评估现在输出的是增强策略指标，而不是项目唯一目标：

- `avg_intent_resolution_score`
- `avg_execution_readiness_score`
- `avg_phase_completion_score`
- `avg_drift_adaptation_score`
- `drift_detection_rate`

这些指标用于把增强策略和 bundle-planning 主线对齐，而不是把 RL 本身当作项目主贡献。

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

## RL-Enhanced Policy Module

### Decision Formulation

```
State S =
  (task constraints, long-term user memory, belief/uncertainty, setup status)

Action A =
  {ask_question, retrieve_candidates, propose_bundle, relax_constraint}

Bundle scoring objective =
  utility(bundle, constraints)
  + alignment(bundle, long_term_memory)
  + phase_value(bundle, purchase_rhythm)
  - risk(bundle)

RL-enhanced policy:
  仅用于优化 ask / proceed / planning action 的选择，
  不改变项目的主问题定义。
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
# 行为克隆预训练 + RL-enhanced 微调
python -m shopping_agent.rl.pretrain --synthetic --epochs 20
python main.py train --rl --iterations 200
```

### 基准测试与消融实验

```
Full system:         Bundle planning + long-term memory + phased purchase
Baseline A:          Naive Retrieval
Baseline B:          Constraint-only Agent
Baseline C:          Single-item Agent
Policy ablation:     Heuristic policy vs RL-enhanced policy
```

```bash
python run_benchmark.py --compare      # 输出消融对比表
python run_benchmark.py --rl --save    # 保存详细结果
```

---

## 设计原则

- **接口契约优先**：`ShoppingTask` 是所有模块的共享契约，先定版再开发
- **主线优先**：所有增强模块都服务于 long-horizon bundle planning 主线
- **Belief-guided**：Belief State 用于降低澄清不确定性，而不是单独作为贡献点
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
