# CPT-World 任务设计 v5.2（CANDIDATE）：有限离散真实因果网络上的局部因果任务

> 取代 v5.1。约束更正：**世界变量不是必须二元**；采用有限离散 DAG + 有理 CPT。
> 主任务仍是真实网络上的局部因果问题，不是全图。
> 状态：CANDIDATE；隐藏世界、planner、scorer owner 闭合前不得采样。
> 可直接复用的上游世界/任务/查询清单见 `docs/directly-reusable-task-sources.md`。

---

## 1. 对两个问题的直接回答

### 简单路径假设是否太强？

**是。** 它为了“策略树可枚举”强行规定：
- 恰好一条路径；
- 中间变量互不相连；
- 所有边同参数、所有非路径变量严格 isolated。

现实因果系统几乎从不长这样：会有 fork、collider、多路径、直接边与间接边并存。
所以 v5.0 的 PATHWAY-RECON 不是“模拟现实”，而是“把现实削成可解玩具”。撤回。

### 全图发现为什么也不行

全图发现把任务变成“图编辑距离优化”，模型被要求输出许多与当前问题无关的边；
这既不是真实科学家/决策者的目标，也不是你最初的双向效应任务。
真实因果工作的主流也不是全图：它们问的是**一个具体 query 需要的那部分结构**。

---

## 2. 当前因果 benchmark / 环境实际教给我们什么

以下是本轮重新核验的构造机制，按“他们实际做了什么”而不是原则转述：

| 工作 | 实际构造 | 对 CPT-World 的真正启发 |
|---|---|---|
| SciGym（本地 repo + PDF） | 用真实 SBML 生物模型，给 partial model，允许 `change_initial_concentration` / `nullify_species`，truth 是完整模型 | 隐藏世界应是**真实机制模型**，任务来自“部分被遮住”的真实网络，而不是人工拼路径 |
| CausalBench（arXiv:2404.06349） | 15 个 bnlearn 真实贝叶斯网（Asia 8 节点、Cancer 5、Earthquake 5、Survey 6、Sachs 11、Child 20...），做 correlation/skeleton/causality 三层任务 | 用真实 DAG 拓扑和真实 CPT 做结构任务；难度来自网络本身规模/密度 |
| LeGIT（arXiv:2503.01139） | 在 Asia/Child/Insurance/Alarm 真实 BN 上做多轮干预目标选择，SHD/SID 评价 | “给定真实网络，选择最优干预”是现实任务，可直接继承其离散 BN 任务形态 |
| CausalPitfalls（arXiv:2505.13770） | 每个任务围绕一个真实统计陷阱（Simpson、Berkson、confounding、mediation...），SCM 生成数据，唯一正确实验/分析策略 | 任务族应由**真实因果问题**驱动，不是抽象图族 |
| CausalGame（arXiv:2607.04293，repo 核验） | 每个场景一个具体 SCM（如 antenna_trap），有 latent/selection/measurement error；agent 在预算内设计实验，analytic optimum 校准 | 真实任务 = “隐藏机制 + 明确效用 + 预算实验”；但其 latent/selection 超出当前 CPT-World owner |
| CausaLab（arXiv:2605.26029，repo 核验） | 隐藏 SCM 共享于 prior records / manipulator / reactor；任务分“预测准确”与“机制恢复”两项 | 任务分与机制恢复分应分离；模型必须从干预中恢复机制，而不是猜答案 |
| CausalDS（arXiv:2607.08093） | 概念 SCM 与观测层分离；identifiability gating；不可识别就 abstain | 隐藏世界与观测难度必须分离；不可识别是一等结果 |
| CauGym（arXiv:2602.06337） | 采样 10 节点 DAG、语义化节点、DoWhy 求 ATE/CDE/ETT/NDE/NIE/PN/PS 等标准因果 query | 任务头应是标准因果估计量，而不是“找全图” |
| NewtonBench（arXiv:2510.07172） | 对 canonical 物理律做 counterfactual shift，得到可扩展且抗记忆的隐藏机制；solvability proof | 真实结构 + 机制扰动 = 既不靠记忆、又保持现实语义 |
| CLadder / CausalWorld | motif × query × truth owner；参数化 generator + 干预 API | 只继承 owner 与干预 API，不继承故事/机器人 |

结论：**前沿工作没有一个靠“单一路径”当隐藏世界；它们靠真实网络、真实 query、真实实验 API、truth owner。**

---

## 3. 修订后的真实需求

1. 隐藏世界优先采用**真实小规模因果网络**（bnlearn 小 BN、Boolean GRN），不是手写 path。
2. 任务是一个**局部因果问题**，答案不是全图。
3. 锚点最少：决策/干预目标问题只锚 outcome；效应/机制问题只锚 treatment-outcome 对。
4. 模型自主选择干预与测量，探索 query 所需的局部结构。
5. 难度由真实网络规模、query 相关结构量、机制强度决定。
6. 有解：planner 在统一预算内可解；过难就把网络/query 换小，不加结构假设。
7. 非平凡：只测锚点不能完成 query。
8. 可精确验证：truth 由完整网络 owner 计算。
9. 抗记忆：真实网络变量名替换为 opaque id，CPT 做有理化扰动（借鉴 CausalBench/Caliper/NewtonBench）。

---

## 3.1 原始 QN seed 的致命退化（必须写清，不能再用）

原始 QN 世界里：

\[
P(X,Y)=
\begin{cases}
\frac14+\frac e2, & (0,0),(1,1)\\
\frac14-\frac e2, & (0,1),(1,0)
\end{cases}
\]

因此观测律已经同时给出：

\[
P(Y=1|X=1)=P(X=1|Y=1)=\frac12+e,\quad
P(Y=1|X=0)=P(X=1|Y=0)=\frac12-e.
\]

而 FORWARD 世界下：

\[
P(Y=1|do(X=x))=P(Y=1|X=x),
\quad
P(X=1|do(Y=y))=\frac12.
\]

REVERSE 世界对称。也就是说：

1. **FORWARD 与 REVERSE 的观测联合完全相同**；
2. **主动方向的 do 分布等于该方向的观测条件分布**；
3. **非主动方向的 do 分布恒为 Bern(1/2)**；
4. **孤立变量的 do 分布就是观测联合本身**。

最致命的退化是第 2、3 条合起来：观测数据已经包含全部效应大小；
干预只提供“哪一侧是主动方向”这一个 bit，而且这个 bit 只是比较
“do 结果是否等于观测条件表”。这不是在做因果实验，而是在查表。
再叠加 full joint counts 自动返回，任务完全坍缩。

### 由此得到的 seed 选择红线

- 不接受任何“主动边 do 分布 = 观测条件分布”的世界；
- 对每个 treatment 锚点，必须预注册验证 \(P(Y|do(X))\ne P(Y|X)\)；
- 优先选择 treatment 节点有 parent/confounder 的真实网络；
  根节点 treatment 通常退化为可观测推断（例如 Cancer 的 Pollution/Smoker 作为 exposure 就有此风险）；
- 查询必须不能由观测条件表直接读出。

---

## 4. 提议的隐藏世界：真实小规模离散贝叶斯网

变量域为有限离散集合，CPT 为有理数；二元网络只是特例，Survey 等多值网络直接使用。

### 4.1 实际网络与结构（来自 bnlearn 公开 .bif，已下载核验）

| 网络 | 类型 | 节点/边 | 实际 DAG | 说明 |
|---|---:|---:|---|---|
| Cancer | 全二元 | 5 / 4 | Pollution→Cancer←Smoker；Cancer→Xray；Cancer→Dyspnoea | 一个 collider + 一个 fork；标准医学诊断网 |
| Earthquake | 全二元 | 5 / 4 | Burglary→Alarm←Earthquake；Alarm→JohnCalls；Alarm→MaryCalls | 经典 collider + fork；explaining away 标准例 |
| Asia | 全二元 | 8 / 8 | Asia→Tub；Smoke→Lung；Smoke→Bronc；Tub,Lung→Either；Either→Xray；Bronc,Either→Dysp | 多路径 + collider + deterministic OR；经典诊断网 |
| Survey | 有限离散 | 6 / 6 | A→E,S→E,E→O,E→R,O,R→T | Age(3)、Travel(3)，其余二元；直接进入离散 seed，不必二值化 |

来源：`https://www.bnlearn.com/bnrepository/discrete-small.html` 的 `.bif` 文件。
这些结构不是我们编的，是 CausalBench / LeGIT / bnlearn 社区使用的标准网络。

### 4.2 这些网络上文献中已有的任务类型

对这类网络，已有 benchmark 实际使用的任务不是“输出全图”，而是：

1. **观测查询 / 诊断**：P(cause | symptom)、explaining away（Earthquake 的 Alarm 场景是标准例）。
2. **干预效应查询**：ATE / P(Y | do(X))。CLadder Rung 2、CauGym 的 ATE。
3. **中介查询**：X→Y 的 mediator set / path-specific effect。CLadder mediation、CauGym CDE/NDE。
4. **干预目标选择**：给定 outcome，找最优单变量干预。LeGIT 的 online intervention targeting。
5. **结构恢复**：correlation → skeleton → orientation。CausalBench 三层任务；但结构恢复只作诊断，不作主评分。
6. **反事实转变范围**：不选择隐藏 SCM；在与完整 CPT-World 相容的全部机制完成上计算跨世界转变概率的精确上下界。两条干预边缘给出的 Fréchet 区间只作外界或可另证 sharp 的特例。要求唯一 PN/PS 点值仍排除。

### 4.3 可直接做 seed 的实例（用 BIF 真实 CPT 的 exact 值）

| Seed | 网络 | 任务 | 答案 | 为什么不是 toy |
|---|---|---|---|---|
| S1 | Cancer | R1：使 P(Dyspnoea=1) 最小的单变量干预 | `do(Cancer=0)`：0.3000；`do(Smoker=0)`：0.3010；`do(Pollution=0)`：0.3034 | 必须发现 Cancer 是直接原因；两个上游原因效应相近但都不是最优 |
| S2 | Earthquake | R1：使 P(MaryCalls=1) 最小的单变量干预 | `do(Alarm=0)`：0.0100；`do(Burglary=0)`：0.0147；`do(Earthquake=0)`：0.0172 | 必须发现 Alarm 是直接开关；有 collider |
| S3 | Asia | R1：使 P(Dysp=1) 最小的单变量干预 | `do(Bronc=0)`：0.1389；`do(Smoke=0)`：0.3191；`do(Lung=0)`：0.4189；`do(Either=0)`：0.4150 | 8 节点多路径；必须区分直接原因 Bronc 与远端原因 |
| S4 | Cancer | R2：Pollution/Smoker 到 Dyspnoea 的 mediators | `{Cancer}`，偏序 `Cause→Cancer→Dyspnoea` | 中介唯一但存在两个上游 collider 父节点 |
| S5 | Asia | R2：Smoke 到 Dysp 的 mediators | `{Lung, Either, Bronc}`；偏序 Lung→Either→Dysp，Bronc→Dysp | 多路径 + collider + 公共 fork；不是简单路径 |
| S6 | Survey | R1：使 P(Travel=car) 最大的单变量干预 | `do(Occupation=self)`：0.6668；`do(Residence=big)`：0.5860；`do(Education=uni)`：0.5690 | 多值域；必须发现 Occupation 是 Travel 的直接父节点 |
| S7 | Survey | R2：Age/Sex 到 Travel 的 mediators | `{Education, Occupation, Residence}`；偏序 E→O→T，E→R→T | 多值域 + fork + collider at Travel |

以上数值由下载的 BIF 真值经 exact hard-do 计算得到；planner 与渲染 owner 仍需按第 9 节闭合。

**Manipulability mask 必须按真实语义预注册**：真实 BN 不是所有节点都可干预。
例如 S1 的 exposure 版只允许 `do(Pollution)`、`do(Smoker)`，此时最优是 `do(Smoker=0)`；
若把 Cancer 也声明为可干预的“治疗靶点”，答案才是 `do(Cancer=0)`。
每个 seed 必须同时冻结 outcome readonly 与 manipulable set，否则会制造“干预症状”这种伪任务。

### 4.4 隐藏与扰动

- 变量名替换为 opaque token，模型不知道语义。
- 先直接用 BIF 真实 CPT 做第一批 owner 测试；再施加预注册 rational perturbation 生成抗记忆变体。
- planner 侧有限世界集 = 真实拓扑 × 若干 CPT 扰动档。
- 模型只看到 N 个有限离散变量、各自取值域、动作 schema、终局问题。

---

## 4.5 Seed 任务清单：网络 motif × 查询方式 × 任务定义

| 网络 | 主要因果 motif | 主要查询方式 | Seed 任务定义 |
|---|---|---|---|
| Cancer | collider：Pollution→Cancer←Smoker；fork：Cancer→Xray、Cancer→Dyspnoea | ATE / do-query；mediator set；best intervention | S1：给定 outcome Dyspnoea，在预注册 manipulable set 中返回使 P(Dyspnoea=1) 最小的 (X,v)；S4：给定 Cause∈{Pollution,Smoker} 与 Dyspnoea，返回 Cause→Dyspnoea 的 mediator set 与偏序 |
| Earthquake | collider：Burglary→Alarm←Earthquake；fork：Alarm→JohnCalls、Alarm→MaryCalls | explaining away；ATE / do-query；best intervention；mediator set | S2：给定 outcome MaryCalls，返回使 P(MaryCalls=1) 最小的 (X,v)；备选：Burglary→MaryCalls 的 mediator set |
| Asia | fork：Smoke→Lung、Smoke→Bronc；collider：Lung→Either←Tub；多路径到 Dysp | mediator set + order；best intervention；诊断/explaining away | S3：给定 outcome Dysp，返回使 P(Dysp=1) 最小的 (X,v)；S5：给定 Smoke 与 Dysp，返回 mediators `{Lung,Either,Bronc}` 及偏序 |
| Survey | fork：E→O、E→R；collider：O→T←R；链 A→E、S→E；多值域 | best intervention；mediator set；subgroup policy | S6：给定 outcome Travel=car，返回使 P(Travel=car) 最大的 (X,v)；S7：给定 A/S 与 Travel，返回 mediators `{E,O,R}` 及偏序 |

查询方式定义：
- **R0 效应查询**：返回 \(P(Y=y^*\mid do(X=x))\) 或 ATE；只作诊断，不作主评分。
- **R1 最优干预**：返回 \(\arg\min/\max_{(X,v)\in\mathcal A} P(Y=y^*\mid do(X=v))\)；评分 regret。
- **R2 中介集**：返回 X→Y 所有有向路径上的观测变量集合 M 及 M∪{X,Y} 上的偏序；评分 mediator F1 + order correctness。
- **R3 分层决策**：返回按协变量分层的 treatment rule；评分 regret（DRAFT）。
- **R4 反事实转变范围**：返回目标状态在 treatment 下发生、在 baseline 下不发生的 sharp `[lower, upper]`；评分连续端点误差。

---

## 5. 提议的任务族：真实网络上的局部因果问题

### 5.1 R1：Best-intervention / Root-cause targeting（1 锚点）

- 只公开 outcome `Y`。
- 问题：在可干预变量中，找出单变量干预 `do(X=v)`，使 `P(Y=1)` 最大（或最小）。
- 终局：
  ```json
  {"type":"answer","intervention":{"target":"KJM","value":1}}
  ```
- 评分：regret
  \[
  \max_{a} u(a;W)-u(\hat a;W),\quad u(a;W)=P_W(Y=1\mid do(a)).
  \]
- 现实对应：政策干预选择、故障定位、LeGIT 的干预目标问题。
- 模型必须发现哪些变量是 Y 的祖先、哪些干预有效；不需要恢复全图。

### 5.2 R2：Mediator-set / pathway query（2 锚点）

- 公开 treatment `X` 与 outcome `Y`。
- 问题：找出所有位于 X→Y 有向路径上的观测变量，并给出它们相对 X,Y 的偏序。
- 终局：
  ```json
  {"type":"answer","mediators":["V3","V1"],"order":[["X","V3"],["V3","Y"],["X","V1"],["V1","Y"]]}
  ```
- 评分：mediator set 的 F1 + order 的 pairwise correctness；不要求输出全图。
- 现实对应：mediation/pathway 分析中“哪些变量解释 treatment 对 outcome 的作用”。
- 在有 fork/collider/多路径的真实 DAG 上，这个答案不再退化成简单路径。

### 5.3 R3：Subgroup decision（2 锚点，DRAFT）

- 公开 treatment `T` 与 outcome `Y`，以及若干 opaque 基线协变量。
- 问题：输出按协变量分层的 treatment rule，使期望 Y 最大。
- 现实对应：heterogeneous treatment effect / individualized treatment rule。
- 该族等 decision rule 合同冻结后再做。

### 5.4 为什么不把全图作为主任务

全图只在诊断中保留（例如 SHD 报告 planner 的恢复程度）。
当前 RL milestone 的训练奖励只来自 query 终局答案的连续质量；regret 与结构正确性
按 `terminal-quality-v7` 标量化，样本成本单独报告且不进入奖励。

---

## 6. 难度分级（由真实网络与真实 query 给出）

| 档 | 网络 | 任务 | 需要探索利用的结构 |
|---|---|---|---|
| Q0 | Cancer (5) | S1：Dyspnoea 的最优单变量干预 | Cancer 的直接原因/上游 collider |
| Q1 | Earthquake (5) | S2：MaryCalls 的最优单变量干预 | Alarm 的直接原因/collider |
| Q2 | Asia (8) | S3：Dysp 的最优单变量干预 | 多路径祖先集 |
| Q3 | Cancer (5) | S4：Pollution/Smoker → Dyspnoea 的 mediator set | 单一中介 Cancer |
| Q4 | Earthquake (5) | Burglary → MaryCalls 的 mediator set | 单一中介 Alarm |
| Q5 | Asia (8) | S5：Smoke → Dysp 的 mediator set | 多路径 + collider + fork |
| Q6 | Survey (6，多值) | S6：Travel=car 的最优单变量干预 | 多值域 + Travel 的直接父节点 |
| Q7 | Survey (6，多值) | S7：Age/Sex → Travel 的 mediator set | 多值域 + fork + collider |

Survey 的多值变量直接进入离散 seed，不需要二值化。
统计难度仍用机制强度/CPT 扰动档（易、中、难）缩放，不发明路径长度。
难度来源是真实网络的规模和 query 相关结构量，不是“恰好一条路径”。

---

## 7. 交互与观测合同（不变）

```json
{"type":"intervene","target":"KJM","value":1,"measure":["NGR","LWH"],"batch_size":8}
```

- `value` 是 target 取值域中的任意一个状态，不再限于 0/1；
- 只返回 `measure` 的 joint counts，count table 的维数是 `measure` 变量取值域的笛卡尔积；
- 成本 = `batch_size × |measure|`；
- 所有变量默认 manipulable/readable（R3 协变量除外）；
- 题面只给锚点与问题，不给网络拓扑、CPT、相关变量提示。

---

## 8. 准入 gate

1. 隐藏世界来自公开可核查的真实网络来源（bnlearn/Boolean GRN 等），不接受手写 path。
2. 主答案不是全图；全图只作诊断。
3. 只测锚点不能拿满分。
4. 没有 full joint 自动返回。
5. planner 在统一预算内可解。
6. 无 passive 短路：给定观测不能直接算出 query。
7. 无泄漏：opaque 变量名、CPT 扰动、不出现真实网络名。
8. symbol orbit 不变性。
9. 每个任务能指出真实实验逻辑来源。
10. 过难就降网络/降 query，不加结构假设。

---

## 9. 下一步

1. 获取 Cancer / Earthquake / Asia / Survey 的 DAG 与 CPT（bnlearn 公开 .bif，本轮已下载核验）。
2. 写 owner：真实离散 DAG + rational CPT + hard-do + 任意 measure 的 exact marginal；owner 必须支持多值域。
3. 先做 Q0：Cancer 上的 S1 best-intervention（Dyspnoea）。
   - planner 穷举验证预算内可解；
   - random / endpoint-only / greedy / planner 基线；
   - 检查只测 outcome 的策略不能找到 best intervention。
4. 再做 Q1-Q5；之后决定是否用 CausalGame/CausalDS 方式扩展 latent/selection。
5. 所有 gate 通过后才进入采样分布。

---

## 10. 当前裁决

```text
SIMPLE-PATH WORLD: REJECTED
FULL-GRAPH AS PRIMARY: REJECTED
REAL BN + LOCAL QUERY: CANDIDATE
Q0 (Cancer S1: Dyspnoea best-intervention): FIRST IMPLEMENTATION TARGET
CD-POLICY: DRAFT
LATENT/SELECTION EXTENSION: FUTURE, REQUIRES OWNER CHANGE
```
