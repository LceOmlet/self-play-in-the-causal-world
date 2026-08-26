# CPT-World 世界子空间与采样器（CANDIDATE）

> 修正：扩域不是“往 manifest 里加更多已有 world”，而是
> **用最短语言描述离散 seed 的子空间，再写一个最简、最泛化的采样器**。
> 隐藏方式、查询方式、任务类型保持当前 seed 已有方式，但必须能与采样/枚举出的 world 稳定合法匹配。

---

## 1. 正确的关系

```text
seed = (world, hiding, query, task)

hiding  ≈ 固定
query   ≈ 固定
task    ≈ 固定

world   ∈ WorldSpace
```

扩域发生在 `WorldSpace`：

```text
WorldSpace := { world | 由最短 grammar 生成，且 legal(world) }
```

已有 6 个 seed 只是 `WorldSpace` 中的 6 个点。
以后不是调这 6 个点，而是在 `WorldSpace` 中采样。

---

## 2. WorldSpace 的最短语言

一个 world 用下面的紧凑 spec 描述：

```text
world_spec := {
  family,          # sampled_dag / cladder_meta_model / bnlearn_bif
  topology,        # sampled：dag-nN-o<order>-e<edges>-f<effect>；upstream：原始结构名
  variables,       # 内部变量名序列
  domains,         # 每个变量的有限取值数
  edges,           # DAG 边列表
  cpt              # 完整 rational CPT 行
}
```

Parametric grammar 声明以下概率模型：

```text
node_counts           # {2,3,4} 均匀
max_domain_size       # 默认 5；域大小从 {2,...,5} 均匀
topological order     # n! 个拓扑序均匀
forward edge subset   # 该拓扑序所有前向边子集均匀
root_prior            # Uniform(0,1)，有理化到 exact Fraction
edge_effect           # Uniform(-1/2,1/2)，有理化到 exact Fraction
rational_denominator_bound  # exact rational 的精度界，不是效应网格
```

`seed` 只作为 `random.Random(seed)` 的种子，不参与任何坐标展开、数组下标或网格选择。

`signed multiplicative mechanism`：

```text
q_p = (1/2 + e_p) / (1/2 - e_p)
preferred_p = parent_state mod domain_size
child raw weight = ∏ q_p for parents preferring that state
normalize to a CPT row
```

`sample_task_world` 对数值型 query 做 acceptance-rejection：只接受数值稳定且 causal target 非零的 CPT draw，不设 goodness 阈值；无稳定 draw 时 fail closed。

`profile_task_targets` 只采样并报告可廉价精确计算的 target 分布与每条边的正负效应计数，不做过滤。个体反事实不走该通用入口；它必须使用精确求解探针，超时明确记为 `unresolved`，Fréchet 外界不得替代任务 target。

---

## 3. 最简最泛化采样器

采样器只有一个入口：

```text
sample_world(grammar, seed) -> world_spec
```

内部流程：

```text
rng = random.Random(seed)

1. n = rng.choice(grammar.node_counts)
2. domains = [rng.randint(2, max_domain_size) for _ in range(n)]
3. order = list(range(n)); rng.shuffle(order)
4. edge subset = rng.getrandbits(k) over the k forward pairs
5. root prior = uniform draw from (0,1), rationalized
6. edge effect = uniform draw from (-1/2,1/2), rationalized
7. build signed multiplicative CPT rows
8. validate legal_world
```

完备性来自声明分布的支持集：拓扑序 × 前向边子集覆盖全部 n 节点 DAG；均匀性来自 `random.Random` 的声明分布，不来自 seed 坐标展开。

---

## 4. 固定维度与匹配规则

当前固定：

```text
hiding:
  mechanism_hidden
  role_hidden
  relevant_set_hidden
  evidence_by_intervention_only
  no_full_joint
  manipulability_via_action_legality

query:
  ate
  backadj_minimal_sets
  best_intervention
  mediator_set

task:
  target_query
  discovery
  decision
```

每个 world 采样后必须过匹配器：

```text
supports_query(world, query) -> bool
supports_task(query, task)  -> bool
supports_hiding(world, hiding) -> bool
```

例如：

- `ate` 要求 world 中存在 treatment 与 outcome；
- `mediator_set` 要求 treatment 与 outcome 之间存在至少一条有向路径；
- `best_intervention` 枚举 outcome 的有向祖先作为 decision target；该部署目标与 outcome 在实验阶段只读，其他合法变量才可作为实验 target。

不满足匹配的 world 直接丢弃，不进入任务构造。

---

## 5. 组装器

```text
assemble(world_spec, hiding, query, task):
    if not legal(world_spec): reject
    if not supports_query(world_spec, query): reject
    if not supports_task(query, task): reject
    if not supports_hiding(world_spec, hiding): reject
    if degenerate(world_spec, query): reject
    return anonymized_seed(world_spec, hiding, query, task)
```

只有通过全部匹配和退化检查的 world 才变成 seed。

---

## 6. 难度与“类似现实任务”

难度不再由 A/P/N 定义，而由采样空间本身给出：

| 空间参数 | 影响 |
|---|---|
| 变量数 N | 需要探索的变量规模 |
| 域大小 | 状态空间复杂度 |
| 边数 / 路径数 | 结构探索难度 |
| edge_effect_grid | 统计分辨难度 |
| 拓扑序 + 边子集 | 是否出现 confounder/collider/多路径 |

当前参数 grammar 覆盖二元与多值 DAG；真实上游网络仍作为 `WorldSpace` 显式点保留，并作为回归 fixture。

---

## 7. 当前动作

```text
DONE:
  - WorldSpec / WorldGrammar
  - 声明分布：node count / domain size / topological order / edge subset
  - edge_effect ~ Uniform(-1/2,1/2)，exact rational
  - root_prior ~ Uniform(0,1)，exact rational
  - signed multiplicative multi-parent CPT mechanism
  - seed 仅作为 random.Random 种子，无坐标展开
  - sample_world 从声明分布采样结构 + CPT
  - profile_task_targets 报告可廉价精确计算的 target 分布与每条边正负效应计数；个体反事实由精确求解探针单独统计
  - sample_task_world acceptance-rejection 只取数值稳定且 target 非零实例
  - 当前 sampled seeds 包含 ate / backadj_minimal_sets / best_intervention / mediator_set
  - 数值任务使用 sample_task_world；结构 discovery 任务直接复用 sample_world
  - sampled seed 均匀采样非空 manipulability width K 与同宽子集
  - sampled seed 独立均匀采样 observation bandwidth M
  - 显式被动 observe 与 hard-do 共用轮数、原子样本计数和 selected readout
  - legal_world / upstream fixtures / rendering / query_truth 同前

NOT_DONE:
  - budget 的每 seed 冻结值
  - 难度 band 冻结（当前只有连续 profile，无阈值）
  - 不做 planner / reference policy / 准入门
  - 退化检查待查询/任务模式确定后复查，不单独做模块
```
