# Seed 任务渲染边界（CANDIDATE）

> 对应线性过程第 2 步：渲染任务并发送给 LLM 时，遮蔽什么。
> 本文件只描述当前代码实际实现的边界，不新增 query truth、scorer 或 parser 合同。

## 1. 模块边界

```text
rendering.py 只做两件事：
1. 决定可见字段；
2. 生成发送给 LLM 的 prompt / initial messages。

rendering.py 不做：
- query truth 计算；
- 答案解析；
- 任务评分；
- world 采样或执行；
- 退化检查。
```

`budget` 仍是显式运行参数。主采样 seed 固定写入 `observation_bandwidth`（即 M）
和 `observation_budget_exponent`（在 11、12、13、14 中均匀采样）；默认总标量预算为
`M * 2^observation_budget_exponent`。运行时不能覆盖 M。`measure_max` 只保留给
没有该字段的旧手工 seed；没有指数的旧 seed 继续使用 `2^11`。

## 2. 对 LLM 可见

```text
- 变量：opaque 3 字母 id + state_0 / state_1 / ...
- 查询锚点：按 query/task 约束最少暴露（opaque id）
- 查询目标：effect 公式、最优干预目标、mediator 问题等
- 动作语法：intervene JSON 与 observe JSON；合法 do target 由 manipulability 掩码转换而来
- measure 语法：可读变量列表；intervene 时 target 不可出现在 measure；长度不超过 seed 的 M
- 唯一预算：`max_observations`；一次查询消耗
  `batch_size * len(measure)`，查询次数不另设上限，`batch_size` 可取任何能放入剩余预算的正整数
- 反馈合同：同一隐藏世界下的 IID hard-do 或自然分布样本，只返回所选 measure 的 joint counts
- 终局 JSON 字段：由 task_head 决定
```

## 3. 对 LLM 隐藏

```text
- 内部变量名（Pollution、Smoker、X、Y、V1...）
- DAG 边、CPT 行、参数、world source 文件 / model_id / graph_id / story
- 网络名与上游来源名
- 非锚变量的因果角色、相关变量集合
- manipulability / readable 掩码本身（只表现为动作合法性 / measure 合法性）
- true answer、oracle、难度标签、seed_id
```

## 4. 遮蔽确实受 query 与 task 约束

遮蔽不是独立于 query/task 的统一遮罩。当前实现按以下方式约束可见锚点：

| query_type | 必须暴露的锚点 | task_head | 终局对象 |
|---|---|---|---|
| `ate` | treatment, outcome | `target_query` | 一个连续效应值 |
| `backadj_minimal_sets` | treatment, outcome | `discovery` | one complete adjustment set |
| `mediator_set` | treatment, outcome | `discovery` | mediators + 路径上的连续有向边 |
| `best_intervention` | decision target + outcome + objective + outcome 状态 | `decision` | decision target 的一个部署状态 |

相应 readonly 约束：

```text
outcome / collider 锚点：readonly（不可 do）
ATE treatment 锚点：readonly（只能通过其他合法实验识别）
best-intervention decision target：实验阶段 readonly，只能作为终局部署答案
主采样：锚点 readonly；其余变量中先均匀采样 K，再等概率采一个 K 子集可 do
全部变量：当前均可读
```

## 5. 六种 hiding modes 的当前执行位置

| hiding mode | 当前执行 |
|---|---|
| `mechanism_hidden` | prompt 不序列化 world_source / 图 / CPT |
| `role_hidden` | 只暴露 query 锚点，不暴露其他变量角色 |
| `relevant_set_hidden` | 不标记相关 / 非相关变量 |
| `evidence_by_intervention_only` | prompt 声明无免费初始数据；证据来自显式付费 batch 实验 |
| `no_full_joint` | prompt 声明环境不自动返回 full-joint；仍由模型显式选择 measure 子集 |
| `manipulability_via_action_legality` | 只显示合法 do target，不显示掩码 |

## 6. 已实现与仍未冻结

```text
- 已实现：generic WorldSpec exact hard-do / 自然分布、selected-measure batch runtime
- 已实现：五种 query truth、终局 parser 与 raw task scorer
- 已实现：prompt → intervene/observe → batch feedback → terminal score 闭环
- planner 可解性
- budget 的每 seed 冻结值
- symbol orbit 渲染变体
- 退化检查
- backadj / mediator 的 revealing-quality 指标与 sampler 准入
```

## 7. 回归

```text
- candidate-v1.json 6 个 seed 均可用 render_seed_task_prompt 渲染；
- 渲染结果不出现内部变量名、图、CPT、上游来源名；
- 不同 CPT 但相同结构/seed_id 的 world 生成相同 prompt；
- iter_sampled_seeds 只枚举结构合法的锚点组合（例如 ate 不取无有向路径的变量对）。
- 五种 registered generic 模式均通过真实多轮 runtime 闭环；
- feedback 只含 requested measure，action-keyed tape 对改名、拆 batch、插入其他 arm 不变。
```
