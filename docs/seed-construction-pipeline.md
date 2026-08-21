# CPT-World Seed 构造流水线（CANDIDATE）

> 目标：先做 `(CPT-World × 查询方式 × 任务)` 的匿名化单 seed；
> 再把这些 seed 的构造器泛化，最终得到“任意 CPT-World + 给定可用查询方式 + 特定任务”的多样性任务族。
> 每一步都跑退化检查，不通过就丢弃或降级为诊断。

---

## 1. Seed 的统一定义

一个 seed 是如下 tuple：

```text
seed := (
  world,                 # 有限离散 DAG + rational CPT，来自真实可核查来源
  manipulability,        # 哪些变量可 do，哪些 readonly
  hidden_info,           # 对模型隐藏的图/CPT/变量语义/角色
  visible_schema,        # opaque 变量、取值域、动作 schema、预算、终局 schema
  query_type,            # 来自 query registry
  task_head,             # 答案对象与 scorer
  D0,                    # 初始证据（通常为空）
  truth,                 # owner 精确计算
)
```

前四个字段来自世界 source；query_type 与 task_head 来自注册表；truth 由 owner 重算。

---

## 1.1 Cancer 这样的世界是怎么组织的，与 CPT 的对应关系

以 bnlearn `cancer.bif` 为例。世界由三部分组成：

1. **变量声明**：每个变量有名字和有限取值域。
   例如 `Pollution {low, high}`、`Cancer {True, False}`。
2. **DAG 结构**：由 `probability (child | parents)` 的声明隐式给出。
   例如 `probability (Cancer | Pollution, Smoker)` 表示
   `Pollution→Cancer`、`Smoker→Cancer`。
3. **CPT 表**：每个 `probability` 块就是一张条件概率表。

Cancer 的真实 BIF 结构：

```text
Pollution → Cancer ← Smoker
Cancer → Xray
Cancer → Dyspnoea
```

与 CPT-World 的对应关系：

| CPT-World 概念 | Cancer BIF 中对应物 |
|---|---|
| finite discrete DAG | 5 个变量 + 4 条边 |
| rational CPT | 每个 `probability` 块（如 0.03, 0.97） |
| hidden mechanism | 把变量名换 opaque、隐藏图与 CPT 后即得 |
| hard-do | 干预时删除目标变量的 CPT 行，固定其取值 |
| local readout | 干预后只返回 `measure` 中变量的计数 |

所以 Cancer 不是“故事”，它是**一个可精确重算的有限离散 CPT 世界**。
它比 CLadder motif 更真实，但仍是一个经典参考网络。

## 1.2 Cancer 的定义与任务动机

Cancer 是 BN 教科书/工具库中的标准诊断网络：
- 两个上游原因：Pollution、Smoker；
- 一个共同结果：Cancer；
- 两个下游症状：Xray、Dyspnoea。

它的原始用途是**不确定诊断推理**：
- 给定症状 Xray/Dyspnoea，反推 Cancer；
- 比较 Pollution 与 Smoker 哪个更能解释 Cancer；
- 教学 collider（Pollution→Cancer←Smoker）和 fork（Cancer→Xray/Dyspnoea）。

后续被 CausalBench、LeGIT 等拿来做 causal benchmark，动机不是“猜全图”，而是：
- 在真实结构上检验因果方向识别；
- 检验最优干预目标选择；
- 检验 ATE / backdoor / mediation 等标准查询。

我们从中借用的任务动机是：
- 使 P(Dyspnoea=1) 最小的干预，是一个明确的**决策问题**；
- Pollution/Smoker→Dyspnoea 的 mediator 是一个明确的**局部结构问题**。
这两个问题都不是“输出全图”。

---

## 2. 第一阶段：只做匿名化 seed

### 2.1 可用的真实世界 source

| Source | 直接可用世界 |
|---|---|
| CLadder `graphs/stories/phenomena.py` | chain, fork, collision, confounding, mediation, diamond, diamondcut |
| bnlearn `.bif` | cancer, earthquake, asia, survey |
| CLadder 含 unobserved 的 IV/arrowhead/frontdoor | 等 latent owner 后再启用 |

### 2.2 可用的查询方式

| query_type | 定义 | 当前 head |
|---|---|---|
| `ate` | \(E[Y|do(X=1)]-E[Y|do(X=0)]\) | target query（数值） |
| `backadj` | X→Y 的 backdoor adjustment set | discovery（集合） |
| `best_intervention` | 使 P(Y=y*) 最优的单变量 do(X=v) | decision（regret） |
| `mediator_set` | X→Y 有向路径上的变量及偏序 | discovery |
| `marginal` / `correlation` / `exp_away` | 观测诊断 | diagnostic |

### 2.3 第一批候选 seed（匿名化后）

| Seed ID | world | query | task | 匿名化后模型看到 |
|---|---|---|---|---|
| SEED-CL-CONF-ATE | CLadder confounding：V1→X, V1→Y, X→Y | ate X→Y | target query | 3 opaque vars；可干预 X；返回 ATE |
| SEED-CL-DIAMONDCUT-BACKADJ | CLadder diamondcut：V1→X, V1→V3, X→Y, V3→Y | backadj X→Y | discovery | 4 opaque vars；可干预 X；返回 adjustment set |
| SEED-BN-CANCER-BESTINT | bnlearn Cancer | best_intervention(Y=Dyspnoea) | decision | 5 opaque vars；outcome readonly；返回 (X,v) |
| SEED-BN-ASIA-MEDIATOR | bnlearn Asia | mediator_set(X=Smoke,Y=Dysp) | discovery | 8 opaque vars；X,Y 为锚；返回 mediators+偏序 |
| SEED-BN-SURVEY-BESTINT | bnlearn Survey | best_intervention(Y=Travel=car) | decision | 6 opaque 多值 vars；outcome readonly；返回 (X,v) |

### 2.4 匿名化规则

- 变量 id 使用 opaque token，禁止出现 `X/Y/Z/V1/Asia/Smoke` 等；
- 变量状态在允许时也要做 symbol orbit（多值域尤其重要）；
- 不暴露图、CPT、query 的中间公式、真实世界名；
- manipulability 只出现在动作合法性里，不解释为什么。

### 2.5 每个 seed 必须通过的退化检查

1. **do≠obs**：对每个 treatment 锚点，验证 \(P(Y|do(X=x))\ne P(Y|X=x)\)。
   若相等，该 query 只能做观测诊断，不得当干预主任务。
2. **无 full-joint 自动返回**：环境只返回 `measure` 子集 counts。
3. **只测锚点不够**：最优策略必须至少依赖一个非锚变量或额外干预目标。
4. **无 passive 短路**：D0 不能直接给出 query 答案。
5. **query truth 唯一**：owner 对 hidden world 算出唯一答案。
6. **planner 有解**：预算内可达预注册精度。
7. **非平凡**：随机策略与 planner 有稳定差距；答案不恒等。
8. **symbol orbit**：匿名化重命名不改变答案与分数。
9. **no leak**：prompt 不含 true graph、CPT、oracle action、难度标签。

---

## 3. 第二阶段：泛化为多样性任务族

目标从“单个 seed”升级为：

```text
任意 CPT-World × 给定可用查询方式 × 特定任务 → seed family
```

### 3.1 需要先冻结的注册表

| 注册表 | 内容 |
|---|---|
| World registry | 有限离散 DAG + rational CPT 的合法 world 表示 |
| Query registry | 每个 query 的形式语义、可计算 truth、适用前提 |
| Task-head registry | 答案 schema 与 scorer |
| Compatibility table | 哪个 world 结构支持哪个 query/head |
| Degeneration oracle | 对每个 `(world, query, head)` 计算退化分数 |

### 3.2 泛化生成器

```text
world ∈ WorldRegistry
  → query ∈ CompatibleQueries(world)
    → head ∈ CompatibleHeads(query)
      → anonymize(world, query, head)
        → compute truth
          → run degeneration gates
            → emit seed certificate 或 REJECT
```

### 3.3 多样性门

固定 `(query, head)` 后，采样多个 world 时：
- 图的非同构类不能坍缩；
- query 答案不能全部相同；
- 最优首动作不能全部相同；
- 策略树不能全部同构；
- 每条边/每个变量必须在至少一个 seed 中真实影响答案或最优动作。

---

## 4. 可能的退化清单（每扩一层都重跑）

| 退化 | 检测 |
|---|---|
| 观测条件=干预分布 | 对每个 treatment 锚点比较 \(P(Y|do(X))\) 与 \(P(Y|X)\) |
| full joint 捷径 | 检查是否存在一个 measure 集合让所有 seed 一次查表 |
| 单变量查询 | 检查 query truth 是否只依赖固定少数变量 |
| passive 观测短路 | 用 D0 计算 query，若可识别则降级 |
| 干预无关 | 检查所有 do 动作是否不改变 query 答案 |
| 世界同构 | 图同构检测 |
| 答案坍缩 | 不同 world 的 query 答案熵过低 |
| 策略坍缩 | 最优首动作分布熵过低 |
| 语义泄漏 | prompt/标签中检测原始变量名、图结构字符串 |
| 成本坍缩 | 检查 cheap 动作是否总是支配昂贵动作 |

---

## 5. 当前动作

```text
DONE:  data/seeds/candidate-v1.json 已生成 6 个匿名化候选 seed
       src/cpt_world/seeds.py 已固定 CandidateSeedSpec 与 manifest loader
       scripts/create_seed_manifest_v1.py 可复现生成
THEN:  对每个 seed 跑 2.5 退化检查与 planner 可解性
NEXT:  只通过检查的 seed 进入 owner/采样实现
LATER: 冻结三个注册表，实现 3.2 的泛化生成器
```
