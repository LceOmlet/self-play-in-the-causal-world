# CPT-World Seed 选项清单（CANDIDATE）

> 当前已固定在 `src/cpt_world/seeds.py` 与 `data/seeds/candidate-v1.json` 中的选项。
> seed = (CPT-World, 隐藏方式, 查询方式, 任务)。

---

## 1. CPT-World 选项

### 1.1 CLadder world（候选）

| graph_id | 结构 |
|---|---|
| confounding | V1→X, V1→Y, X→Y |
| diamondcut | V1→X, V1→V3, X→Y, V3→Y |
| collision | X→V3←Y |

### 1.2 bnlearn BIF world（候选）

| 网络 | 节点 / 边 | 变量域 |
|---|---:|---|
| cancer | 5 / 4 | 全二元 |
| asia | 8 / 8 | 全二元 |
| survey | 6 / 6 | Age(3)、Travel(3)，其余二元 |

---

## 2. 世界隐藏方式

“隐藏世界”决定模型**不知道什么、能看到什么、只能通过什么获得信息**。
匿名化只是其中的一个渲染子项，不是隐藏方式本身。

| 隐藏方式 | 模型不知道 | 模型能看到 / 能获得 |
|---|---|---|
| mechanism_hidden | DAG、CPT、true world index | 变量集合与取值域 |
| role_hidden | 变量因果角色（confounder/mediator/collider/outcome） | query 必需的锚点 |
| relevant_set_hidden | 哪些变量与 query 有关 | 所有变量一律 opaque |
| evidence_by_intervention_only | 初始 D0 为空，无观测数据 | 只有 `do + measure` 的局部 counts |
| no_full_joint | 未选择 measure 的变量 | 仅 `measure` 子集的 joint counts |
| manipulability_via_action_legality | 哪些变量可干预及原因 | 动作合法 / 非法反馈 |

---

## 2.1 匿名化：只是隐藏方式的渲染实现

| 匿名化项 | 作用 | 是否属于世界隐藏 |
|---|---|---|
| opaque_labels | 变量名替换为 3 字母 token | 辅助 `role_hidden` |
| state_anonymization | 状态名替换为 state_i | 辅助 `mechanism_hidden` |
| surface_order | 变量展示顺序置换 | 抗表面捷径，不是语义隐藏 |

---

## 3. 查询方式选项

| 查询方式 | 定义 | 当前 seed |
|---|---|---|
| ate | \(E[Y|do(X=1)]-E[Y|do(X=0)]\) | SEED-CL-CONF-ATE |
| backadj_minimal_sets | X→Y 的最小 backdoor adjustment sets | SEED-CL-DIAMONDCUT-BACKADJ |
| best_intervention | 使 \(P(Y=y^*|do(X=v))\) 最优的单变量干预 | SEED-BN-CANCER-BESTINT, SEED-BN-SURVEY-BESTINT |
| mediator_set | X→Y 路径上的 mediators 及偏序 | SEED-BN-ASIA-MEDIATOR |

---

## 4. 任务选项

| 任务 head | 答案对象 | 当前 seed |
|---|---|---|
| target_query | 单个连续效应数值 | SEED-CL-CONF-ATE |
| discovery | adjustment sets | SEED-CL-DIAMONDCUT-BACKADJ |
| discovery | mediators + order | SEED-BN-ASIA-MEDIATOR |
| decision | 单变量干预 | SEED-BN-CANCER-BESTINT, SEED-BN-SURVEY-BESTINT |

---

## 5. 当前完整矩阵

| seed_id | world | 世界隐藏方式 | 查询方式 | 任务 |
|---|---|---|---|---|
| SEED-CL-CONF-ATE | CLadder confounding | mechanism_hidden + role_hidden + relevant_set_hidden + evidence_by_intervention_only + no_full_joint | ate | target_query |
| SEED-CL-DIAMONDCUT-BACKADJ | CLadder diamondcut | 同上 | backadj_minimal_sets | discovery |
| SEED-BN-CANCER-BESTINT | bnlearn cancer | 同上 | best_intervention | decision |
| SEED-BN-ASIA-MEDIATOR | bnlearn asia | 同上 | mediator_set | discovery |
| SEED-BN-SURVEY-BESTINT | bnlearn survey | 同上 | best_intervention | decision |

---

## 6. 状态

- `src/cpt_world/seeds.py`：已固定 `CandidateSeedSpec`、manifest loader、validator、seed_triple。
- `data/seeds/candidate-v1.json`：6 个候选 seed。
- `tests/test_candidate_real_seeds.py`：4 项检查 + seed 三元组断言。
- 全量测试：57 passed；ruff check/format passed。
