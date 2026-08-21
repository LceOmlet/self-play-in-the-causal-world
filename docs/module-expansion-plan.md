# CPT-World 模块化与扩域计划（CANDIDATE）

> 目标：把 seed 构造归纳成模块，先保证任务合法，再逐步扩任务族维度。

---

## 1. 模块划分

| 模块 | 职责 |
|---|---|
| `world_sampler` | 枚举/采样 world 子空间，保证输出 world spec 合法 |
| `world_loaders` | 把 bnlearn `.bif`、CLadder meta-model 转成同一 world spec |
| `registry` | 注册 world / query / task / hiding 选项与兼容关系 |
| `seed_assembler` | 由 (world, hiding, query, task) 组装候选 seed |
| `legality` | 世界合法、query 合法、hiding 合法、task 合法 |
| `degeneration` | do=obs、full-joint 捷径、passive 短路、干预无关 |
| `renderer` | visible schema、opaque labels、state anonymization |
| `answerability` | 在采样 K/M 前，对五类任务按查询特定完整实验面的精确反馈等价类区分 answerable / unanswerable；范围只限当前候选世界族 |
| `difficulty_profile` | 结构与 target 分布的难度坐标，不设阈值 |

当前已完成：

- `src/cpt_world/registry.py`：query/task/hiding/world source 注册表与 legality 初检。
- `src/cpt_world/seeds.py`：候选 seed manifest loader、validator、seed_triple。
- `data/seeds/candidate-v1.json`：6 个候选 seed。
- `data/worlds/**`：bnlearn BIF 与 CLadder meta-model 子集。

---

## 2. 扩域顺序

### Phase 1：world 维度

先扩 CLadder 已观测 motif 与 bnlearn 网络：

```text
当前: confounding, diamondcut, collision
下一步: chain, fork, mediation, diamond
再下一步: earthquake.bif（已下载，尚未入 manifest）
之后: child / alarm / insurance（多值离散）
```

### Phase 2：查询维度

```text
当前: ate, counterfactual_transition_bounds, backadj_minimal_sets, best_intervention, mediator_set
下一步: intervention_target_selection（LeGIT）
之后: path_specific_effect / minimal_adjustment_set（需先冻结答案 schema）
```

### Phase 3：任务维度

```text
当前: target_query, discovery, decision
下一步: decision_policy（多于单点干预）
之后: discovery_class / set_answer_with_abstention（需先冻结 scorer）
```

### Phase 4：隐藏方式维度

```text
当前: mechanism_hidden + role_hidden + relevant_set_hidden + no_full_joint
下一步: partial_D0（给定受控观测数据）
之后: noisy_readout（需 owner 扩展）
```

---

## 3. 每次扩域必须满足

1. 新 world / query / task 先进入 registry；
2. 新 seed 由 assembler 生成，不用手写；
3. 先跑 legality，再跑 degeneration；
4. 只有通过检查的组合进入候选 manifest；
5. 每个新 seed 都做匿名化与 symbol orbit；
6. 通过后加入 `candidate-v1.json` 与回归测试。

---

## 4. 当前动作

```text
DONE: registry.py 模块
DONE: seed_triple 与 legality 测试
DONE: WorldSpec 统一表示（有限域 + 完整 CPT 行）
DONE: bnlearn BIF 与 CLadder meta-model 加载为同一 WorldSpec
DONE: WorldGrammar 声明分布；seed 仅作 RNG seed；effect/prior 为连续对称分布并有理化
DONE: iter_world_space 先枚举 upstream fixtures 再采样 sampled_dag，输出均 legal
DONE: rendering.py 实现 seed 任务渲染边界；6 个 pinned seed 均可渲染
DONE: legal_query_anchors 限制 sampled seeds 只枚举结构合法锚点组合
DONE: registry 显式区分 registered / implemented / diagnostic_only
DONE: query_truth.py 实现 ate / counterfactual_transition_bounds / backadj / mediator_set / best_intervention；collider 条件对比仅保留为 ATE 诊断函数
DONE: task_scoring.py 实现 target_query / decision / discovery parser 与 raw scorer
DONE: 五类 query 均进入同一个 iter_sampled_seeds 主管线；结构 discovery 复用 sample_world，不新增 sampler
DONE: task_answerability 在采样 K/M 前按查询特定完整实验面做候选族内精确可答性划分；反事实区间要求不可区分世界端点完全相同；决策任务要求不可区分世界至少共享一个零后悔部署动作；task_difficulty_profile 与可答性分开
NEXT: 对 answerable sampled tasks 做直观演示与难度 profile
```
