# Query / Task 注册表（CANDIDATE）

> 线性过程第 3 步之前的正式登记。
> 核心规则：**registered 只表示设计已登记；implemented 才表示有可执行 owner。两者不是一回事。**

## 1. Query 注册表

| query_type | anchors | answer_kind | 兼容 task_heads | truth owner |
|---|---|---|---|---|
| `ate` | treatment, outcome | categorical_effect_vector | `target_query` | implemented（`query_truth.py`，通用 WorldSpec） |
| `individual_counterfactual_probability` | treatment, outcome | identified_interval | `target_query` | implemented（`query_truth.py` → `counterfactual_solver.py`，给定个体 factual evidence 后的完整相容世界 exact / epsilon-sharp 认证界） |
| `backadj_minimal_sets` | treatment, outcome | set | `discovery` | implemented（`query_truth.py`） |
| `best_intervention` | decision_target, outcome | complete interventional value profile | `decision` | implemented（`query_truth.py`，通用 WorldSpec） |
| `mediator_set` | treatment, outcome | set_with_order | `discovery` | implemented（`query_truth.py`） |

## 2. Task head 注册表

| task_head | answer_kind | scorer owner |
|---|---|---|
| `target_query` | effect_vector_or_interval | diagnostic_only（`task_scoring.py`，effect error / counterfactual endpoint error） |
| `discovery` | set_or_set_with_order | diagnostic_only（`task_scoring.py`，set F1 / order F1） |
| `decision` | intervention | diagnostic_only（`task_scoring.py`，exact raw / span-normalized regret） |

## 3. 已实现 owner 的边界

`src/cpt_world/query_truth.py` 现在实现：

```text
ate_effect(world, treatment, outcome)
categorical_treatment_effect(world, treatment, outcome)
counterfactual_transition_bounds(world, treatment, outcome)
individual_counterfactual_probability_bounds(world, treatment, outcome, factual evidence)
validate_individual_counterfactual_probability(world, treatment, outcome, scalar prediction)
backdoor_adjustment_sets(world, treatment, outcome)
collider_bias_effect(world, treatment, outcome, collider)  # ATE diagnostic only
mediator_set_truth(world, treatment, outcome)
best_intervention_truth(world, outcome, objective, decision_target)
compute_query_truth(world, seed)
```

能力边界：

- 任意有限离散 `WorldSpec`；
- exact `Fraction`；
- `ate` 均匀采样 treatment 的有序状态对，返回 outcome 所有类别的概率差向量；各分量之和为零，二分类时与旧标量定义等价；
- `individual_counterfactual_probability` 的可见任务给出一个体在指定 factual treatment 下的 factual outcome，要求模型返回该个体在另一 treatment 下达到目标 outcome 的概率之 sharp identified interval；
- 隐藏验证器在与完整 DAG、全部 CPT 行和公开因果语义相容的全部 Markovian 有限机制上计算条件概率区间，不选择隐藏 SCM；模型返回 `lower` 与 `upper` 两个端点；
- 精确闭合时返回 `exact`；否则仅在最终条件概率尺度上两个端点误差都不超过 `0.002` 时返回安全外括的 `epsilon_sharp`；
- 超过该容差时，任务 truth fail closed；Fréchet 外界只保留为求解诊断，不替代正式终局 truth；
- `backadj` 采用标准 back-door criterion，返回 inclusion-minimal 集合；
- collider 条件 do 对比不再是独立 query，只作为 ATE 的诊断切片；
- `mediator_set` 返回所有 X→Y 有向路径上的中间变量与路径边偏序；
- `best_intervention` 只比较显式 `decision_target` 的各状态；该部署变量在实验阶段 readonly，实验 do target 由独立的 manipulability 掩码给出；模型返回所有候选状态的因果价值，最优动作由该向量派生。

`src/cpt_world/task_scoring.py` 现在实现 target_query、decision、discovery 的 terminal parser 和 raw scoring：

```text
target_query:
  parsed categorical effect vector -> truth vector
  component_errors, L1 error, total-variation error, mean squared component error
  parsed individual counterfactual ROI endpoints -> certified endpoint ranges
  lower/upper endpoint error, endpoint MAE/MSE

decision:
  parsed candidate causal-value vector -> exact hard-do value vector
  mean pairwise-gap error; derived action regret and normalized regret

backadj_minimal_sets:
  parsed adjustment-set family -> truth minimal-set family
  precision / recall / F1 / exact_match

mediator_set:
  parsed mediators + order -> truth mediators + path-edge order
  mediator F1；order precision / recall / F1；exact matches
```

`src/cpt_world/rewards.py` 在这些 raw diagnostics 之上实现冻结的
`terminal-quality-v5`，不重新解析模型答案或计算 truth：

```text
ate:                              B_obs / (B_obs + TV_error)
individual counterfactual:       B_obs / (B_obs + endpoint_MAE)
decision:                         B_obs / (B_obs + pairwise_gap_MAE)
backadj_minimal_sets:             maximum-matching soft family F1
mediator_set:                     (mediator_f1 + order_f1) / 2
unfinished / illegal terminal:    0
```

实验样本消耗、query 数、轮数与 token 数保持独立诊断，不进入当前训练奖励。
完整合同见 `docs/terminal-quality-reward-v5.md`。

五个 query type 均由同一个 `iter_sampled_seeds` 主管线发出。ATE、反事实区间与实验决策
使用既有数值稳定 CPT draw；后门调整与中介路径直接复用同一次结构世界采样，
不建立第二个 sampler。

## 4. 仍未实现

```text
- 难度 band 的冻结（当前只有连续难度 profile，无阈值）
- benchmark mixture 的冻结（训练 mixture 已冻结为五类任务各 20%）
- 不做 planner / reference policy / 准入门
```
