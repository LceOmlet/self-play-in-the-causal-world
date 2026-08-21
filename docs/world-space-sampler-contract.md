# WorldSpace 采样器合法性合同（CANDIDATE）

> 工作流：先 grill-with-docs，再 integrate-upstream-first，最后才允许写采样器。
> 本文件先冻结采样器必须满足的合法性与任务覆盖合同，禁止在此前盲目编码 `world_space.py`。

---

## 1. 工作流

```text
Step 1. grill-with-docs
  - 逐条核验已有 docs 与代码中未决条款
  - 找出互相冲突、尚未冻结的 owner

Step 2. integrate-upstream-first
  - 采样器必须建立在已有 upstream 数据与 owner 上
  - 不复制 CLadder / bnlearn / seeds.py 中已有语义

Step 3. 写合法性合同
  - 先定义“匿名 seed 合法”与“能采样更多任务”

Step 4. 才允许实现 sampler
```

---

## 2. Upstream 输入清单

采样器只允许使用这些已有输入：

| 输入 | 来源 | 用途 |
|---|---|---|
| bnlearn `.bif` | `data/worlds/bnlearn/*.bif` | world topology + 真实 CPT |
| CLadder meta-model 子集 | `data/worlds/cladder/meta-models-subset.json` | world motif + 参数 + truth |
| query/task/hiding registry | `src/cpt_world/registry.py` | 固定维度与兼容关系 |
| seed manifest | `data/seeds/candidate-v1.json` | 现有 6 个 seed 回归 |
| seed loader/validator | `src/cpt_world/seeds.py` | 匿名化与 manifest 合法性 |

禁止：新建第二套 graph/CPT/query 语义。

---

## 3. 匿名 seed 合法性

一个采样出的匿名 seed 必须同时满足：

### 3.1 world 合法

```text
- 有限离散 DAG，无环
- 每个变量有非空有限域
- 每个 CPT 行非负、和为 1
- CPT 行数与父节点取值组合一致
- hard-do 对每个可干预变量定义完整
```

### 3.2 query 与 world 匹配

```text
- query 要求的所有锚点变量存在于 world
- outcome 状态属于 outcome 变量域
- treatment ≠ outcome
- 结构前提满足：
  ate               → treatment/outcome 存在
  counterfactual_transition_bounds → treatment/outcome 存在
  backadj           → treatment/outcome 存在
  mediator_set      → treatment→outcome 至少一条有向路径
  best_intervention → decision target 是 outcome 的有向祖先；两者实验阶段只读
```

### 3.3 task 与 query 匹配

```text
- task head ∈ registry
- query/task 兼容表通过
- 答案 schema 与 task head 一致
- scorer 对 world truth 定义唯一
```

### 3.4 hiding 合法

```text
- visible schema 只含 opaque labels 与 state labels
- 不含内部变量名、图、CPT、角色、相关变量集
- 所有非锚变量角色隐藏
- manipulability 只通过动作合法性体现
```

### 3.5 匿名化合法

```text
- opaque label 为 3 字母 token
- 同一 seed 内 labels 唯一
- 跨 seed labels 不碰撞
- symbol orbit 下答案与分数不变
```

### 3.6 非退化

```text
- 对每个 treatment 锚点：P(Y|do X) != P(Y|X)
- 无 full-joint 自动返回
- 无 passive 观测短路
- 非锚变量或非锚干预必须真实影响答案/最优动作
```

---

## 4. “能采样更多任务”的最低要求

采样器不能只是重放现有 6 个 seed。

对每个固定的 `(hiding, query, task)` 组合：

```text
- 至少产生 2 个非同构 legal world
- 不同 world 的 query truth 或最优动作不完全相同
- 不同 world 的最优首动作或策略树不完全相同
```

全空间要求：

```text
|sampled legal seed| > 6
且覆盖当前 6 个 seed 作为回归子集。
```

---

## 5. 采样器接口合同

```text
sample_world(grammar, seed) -> world_spec
legal_world(world_spec) -> bool
supports_query(world_spec, query) -> bool
supports_task(query, task) -> bool
supports_hiding(world_spec, hiding) -> bool
degenerate(world_spec, query) -> bool
assemble_seed(world_spec, hiding, query, task) -> seed | REJECT
```

约束：

```text
- 确定性：同 grammar+seed 必须复现同一 world_spec
- 上游优先：只包装第 2 节输入
- 失败即拒绝：任一合法性/匹配/退化检查失败，不修不补，直接 REJECT
```

---

## 6. 编码前 gate

以下条件全部满足前，不写 `world_space.py`：

```text
G1 已核验 v5.2 与 registry 无未决冲突
G2 已固定 query/task/hiding 与 world 的匹配规则
G3 已证明现有 6 个 seed 能被同一 grammar 表示
G4 已定义 world grammar 的全部合法参数范围
G5 已定义“非同构/不同答案/不同策略”判据
G6 已指定采样器的确定性 seed 方案
```

---

## 7. 当前状态

```text
STATUS: QUERY_TRUTH_OWNERS_IMPLEMENTED_FOR_CURRENT_QUERY_SET
registry.py: IMPLEMENTED
  - query / task 注册表带 truth_owner_status 与 scorer_owner_status
  - registered 与 implemented 显式分离

world_space.py: IMPLEMENTED（声明分布 + 稳定数值 query 采样）
  - WorldSpec / WorldGrammar（含 state_names 与完整 CPT 行）
  - 声明分布：node count / domain size / topological order / edge subset
  - edge_effect ~ Uniform(-1/2,1/2)，root_prior ~ Uniform(0,1)
  - 两者 rationalize 为 exact Fraction；精度由 rational_denominator_bound 控制
  - seed 只作为 random.Random 种子，不做坐标展开
  - sample_world：从声明分布采样结构 + CPT
  - profile_task_targets：报告 target 分布与每条边正负效应计数
  - sample_task_world：acceptance-rejection 只取数值稳定且 target 非零实例
  - legal_world
  - load_bnlearn_world（cancer / earthquake / asia / survey）
  - load_cladder_world（meta-models-subset 全部模型）
  - iter_upstream_worlds（fixtures）/ iter_world_space
  - legal_query_anchors / supports_query / supports_task / supports_hiding
  - assemble_seed（锚点、manipulability/readable 掩码与 observation_bandwidth 显式可控）

rendering.py: IMPLEMENTED
  - render_seed_task_prompt / render_seed_initial_messages
  - 按 query/task 约束暴露锚点，隐藏 world_source / 图 / CPT / 内部名
  - 主采样 seed 持有 observation_bandwidth；运行时不能覆盖

world_runtime.py: IMPLEMENTED
  - visible-only intervene / observe JSON parser；支持多值 state_i 与 selected measure
  - exact hard-do / natural ancestral sampling；只物化 batch 中实际出现的 requested joint counts，遗漏格的 count 精确为 0
  - OutcomeTape v2 按 canonical (target,state,index,node) 定址；与改名/measure/batch 拆分无关
  - WorldSpecEpisode 串联 prompt → 多轮实验 → batch feedback → terminal scorer
  - max_samples 口径为 IID 原子样本行，单轮消耗 batch_size

query_truth.py: IMPLEMENTED（candidate，未经最终审核）
  - ate、counterfactual_transition_bounds、backadj_minimal_sets、mediator_set、best_intervention
  - selected-measure exact law 使用变量消元；原 full-joint 枚举保留为 reference owner 并逐 Fraction 对拍
  - counterfactual_transition_bounds 的两个终局模式由同一 world/query/K/M 实例成对发出，不另建采样器
  - 通用 WorldSpec，exact Fraction
REGRESSION:
  - n=2,3 的声明分布支持集覆盖全部 binary DAG
  - max_domain_size=5 的多值 world CPT 合法且和为 1
  - 30 sampled worlds deterministic + legal
  - 8 upstream fixture worlds 全部加载为同一 WorldSpec 且 legal
  - candidate-v1.json 的 6 个 seed 全部可渲染且不泄漏内部信息
  - sampled numerical seeds 只枚举结构合法锚点组合，且 target 非零
  - ate truth 复现 CLadder groundtruth
  - best_intervention truth 复现 Cancer/Survey seed
  - backdoor adjustment sets 复现 CLadder motif；mediator_set 复现 Asia/Cancer 路径
  - target_query / decision / discovery parser 与 raw scorer 已实现
  - ATE / collider / decision 三种 sampled 模式端到端运行
  - backadj / mediator 两种 discovery 模式由同一 sampler 发出并端到端运行
  - 多值 hard-do、selected joint、action-keyed split/interleave、预算均有回归
  - unittest 与 ruff check/format 由当前仓库回归命令验证

REMAINING:
  - budget 的每 seed 冻结值未定
  - 难度 band 冻结（当前只有连续 profile，无阈值）
  - 不做 planner / reference policy / 准入门
  - 退化问题：待 query/task 模式确定后复查，不单独做模块
  - G3/G5 需要更严格证明后升级
```
