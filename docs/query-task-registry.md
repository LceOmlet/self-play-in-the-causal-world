# Query / Task 注册表（CANDIDATE）

> 线性过程第 3 步之前的正式登记。
> 核心规则：**registered 只表示设计已登记；implemented 才表示有可执行 owner。两者不是一回事。**

## 1. Query 注册表

| query_type | anchors | answer_kind | 兼容 task_heads | truth owner |
|---|---|---|---|---|
| `ate` | treatment, outcome | single_effect | `target_query` | implemented（`query_truth.py`，通用 WorldSpec） |
| `counterfactual_transition_bounds` | treatment, outcome | interval_or_compatible_value | `target_query` | implemented（`query_truth.py` → `counterfactual_solver.py`，完整相容世界上的精确界） |
| `backadj_minimal_sets` | treatment, outcome | set | `discovery` | implemented（`query_truth.py`） |
| `best_intervention` | decision_target, outcome | intervention | `decision` | implemented（`query_truth.py`，通用 WorldSpec） |
| `mediator_set` | treatment, outcome | set_with_order | `discovery` | implemented（`query_truth.py`） |

## 2. Task head 注册表

| task_head | answer_kind | scorer owner |
|---|---|---|
| `target_query` | numeric_value_or_interval | diagnostic_only（`task_scoring.py`，effect / interval endpoint error） |
| `discovery` | set_or_set_with_order | diagnostic_only（`task_scoring.py`，set F1 / order F1） |
| `decision` | intervention | diagnostic_only（`task_scoring.py`，exact regret） |

## 3. 已实现 owner 的边界

`src/cpt_world/query_truth.py` 现在实现：

```text
ate_effect(world, treatment, outcome)
counterfactual_transition_bounds(world, treatment, outcome)
backdoor_adjustment_sets(world, treatment, outcome)
collider_bias_effect(world, treatment, outcome, collider)  # ATE diagnostic only
mediator_set_truth(world, treatment, outcome)
best_intervention_truth(world, outcome, objective, decision_target)
compute_query_truth(world, seed)
```

能力边界：

- 任意有限离散 `WorldSpec`；
- exact `Fraction`；
- `ate` 默认 `state_1` vs `state_0`，可由显式状态索引覆盖；
- `counterfactual_transition_bounds` 返回目标状态从 baseline 未发生到 treatment 发生的 sharp interval；在与完整 DAG、全部 CPT 行和公开因果语义相容的全部机制完成上优化，不选择隐藏 SCM；Fréchet 仅作诊断外界；
- 同一 counterfactual world/query 由主采样管线成对渲染为 `sharp_interval` 和 `compatible_value`；二者共享 K、M 与符号表，后者只检查点是否兼容并报告到区间的连续距离；
- `backadj` 采用标准 back-door criterion，返回 inclusion-minimal 集合；
- collider 条件 do 对比不再是独立 query，只作为 ATE 的诊断切片；
- `mediator_set` 返回所有 X→Y 有向路径上的中间变量与路径边偏序；
- `best_intervention` 只比较显式 `decision_target` 的各状态；该部署变量在实验阶段 readonly，实验 do target 由独立的 manipulability 掩码给出；ties 按状态顺序确定。

`src/cpt_world/task_scoring.py` 现在实现 target_query、decision、discovery 的 terminal parser 和 raw scoring：

```text
target_query:
  parsed effect -> truth effect
  abs_error, squared_error
  parsed lower/upper -> truth lower/upper
  mean absolute/squared endpoint error
  parsed compatible value -> truth interval
  compatible, distance_to_interval

decision:
  parsed (target, value) -> exact optimal probability
  maximize regret = optimal_probability - chosen_probability
  minimize regret = chosen_probability - optimal_probability

backadj_minimal_sets:
  parsed adjustment-set family -> truth minimal-set family
  precision / recall / F1 / exact_match

mediator_set:
  parsed mediators + order -> truth mediators + path-edge order
  mediator F1；order precision / recall / F1；exact matches
```

reward scalarization 未冻结，仍是 raw diagnostics。

五个 query type 均由同一个 `iter_sampled_seeds` 主管线发出。ATE、反事实区间与实验决策
使用既有数值稳定 CPT draw；后门调整与中介路径直接复用同一次结构世界采样，
不建立第二个 sampler。

## 4. 仍未实现

```text
- 难度 band 的冻结（当前只有连续难度 profile，无阈值）
- 不做 planner / reference policy / 准入门
```
