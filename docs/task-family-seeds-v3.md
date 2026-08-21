# CPT-World 任务族 Seed 设计 v3（CANDIDATE）

> **主 benchmark 设计方向已由 `docs/task-family-seeds-v5.md` 取代；v5 已放弃 A/P/N。**
> v3 仍保留为 local-readout 合同、TQ-CB/TQ-S/CD-S 精确 CPT 与防坍缩 gate 的历史/诊断来源；
> 其显式角色指定族不再作为“自主结构探索”主任务。

> 本文件不再改变研究自动机的状态。
> `CANDIDATE` 表示：只能用于反推 hard-do owner、观测 owner、oracle 和 scorer 合同；
> 在 `oracle.py` 复算通过之前，任何 seed 不得进入采样分布、不得标为 `VERIFIED`。
>
> 约束来源（不得违反）：
> `CPT_WORLD_BENCHMARK_RESEARCH_AUTOMATON.md`（S1 状态、A/P/N 轴、三终局头、全局非评分 cap）、
> `CPT_WORLD_LEARNING_OBJECTIVE_CONTRACT_V2.md`（终局二维连续奖励、risk-cost/oracle-regret 评测）、
> `CPT_WORLD_TASK_PROPERTY_CONTRACT.md`（owner 盘点与 Tier-0/1/2 性质）。

---

## 0. 裁决摘要

```text
QN 无条件双向效应任务: REJECT_FOR_MAIN_BENCHMARK
  - 只能作为 parser/tape/N 轴标尺诊断，与 v2 结论一致。
  - 原因不是 renderer，而是 query 的数学性质：minimal sufficient readout set = {X,Y}。

V3 主候选:
  TQ-CB  A2/A1-P1-N-*   条件双向 target query（QN 的合法升级，P1 标尺）
  TQ-S   A2-P2-N-*      subgroup target query（保留 v2 的 HT，P2 标尺）
  AD-MEC3 A=log2 3-P2-N-* 三节点 Markov 等价类主动发现
  CD-S   A2-P2-N-*      协变量条件下的因果决策（regret）
  AD-PATH4 A2-P3-N-*    四节点等价类主动发现（DRAFT，待 oracle 复算）

不进入 V1: 多目标 do、latent confounder、counterfactual、story rendering、self-play/training。
```

---

## 1. 为什么 QN 坍缩了：回答“只测两个变量是否足够”

### 1.1 无条件双向效应：确实只测两个变量就够

设世界是有限 DAG，无 latent confounding，干预是 Pearl hard-do。终局对象是

\[
\tau_{X\to Y}=P(Y=1\mid do(X=1))-P(Y=1\mid do(X=0)),
\qquad
\tau_{Y\to X}=P(X=1\mid do(Y=1))-P(X=1\mid do(Y=0)).
\]

在 hard-do 下，`do(X=x)` 截断所有指向 X 的边；Y 的条件边际只由
`do(X=x)` 后 Y 的分布决定。因此：

- 估计 \(\tau_{X\to Y}\) 只需要对 `do(X=0/1)` 测量 **Y**；
- 估计 \(\tau_{Y\to X}\) 只需要对 `do(Y=0/1)` 测量 **X**；
- 其他变量既不改变可识别性，也不改变无偏性；在 QN 中 Z 与 (X,Y) 独立，连方差效率增益也没有。

所以“是否需要测量第三个变量”的答案是：

> 对**无条件**双变量总效应：不需要，X、Y 的 outcome readout 已经是最小充分集。
> 任何“强制多测一个变量”的观测约束都只是增加噪声和成本，不改变 Bayes risk envelope。
> 这是 query 的性质，v2 的 local readout 合同救不了它。

### 1.2 什么情况下必须测第三个变量

| 终局对象 | 最小充分 readout（不含被干预目标） |
|---|---|
| 无条件 \(\tau_{X\to Y},\tau_{Y\to X}\) | {Y}, {X}，即 {X,Y} |
| **条件**效应 \(\tau_{X\to Y\mid S=s}\)、\(\tau_{Y\to X\mid S=s}\) | {S,Y}, {S,X}；S 必须测 |
| 决策规则 \(a(S)\in\{0,1\}\)（按基线协变量 S 选择 treatment） | {S,Y}；S 必须测 |
| 图/边发现（≥3 节点 Markov 等价类） | 至少需要能区分等价类的节点/边组合 |
| controlled direct effect（若引入多目标 do(T,M)） | {M,Y}；M 必须测 |

因此 V3 的构造原则是：**先选 query，再求 \(S_Q\)，拒绝 \(|S_Q|\le2\) 的 query 进入主 benchmark**；
不要试图给一个只需要两个变量的 query 硬加第三个变量。

### 1.3 另一个必须写明的坑：公开候选世界会缩短策略树

若候选世界集合 \(\mathcal W\) 与 prior 公开，且某个 `do(X=1), measure={S,Y}` 的模式在
\(\mathcal W\) 内唯一决定 answer class，那么最优策略可能只需要一个干预臂。这不是 bug，
而是 Bayesian hypothesis-testing 任务的标准形态（见 Wang et al. 的 bivariate active learning
和 A-CBO）。设计时必须用 oracle 标出**最小充分动作集**；若难度声明依赖“两个方向都要做实验”，
但 oracle 显示一个方向就够，则该 seed 的 P 标签必须降级，不能照抄直觉。

V3 中：
- TQ-CB 公开候选世界后是 **P1 单臂筛查**（但必须买 {S,Y} joint readout），只用于 A/N 标尺；
- 真正需要自适应切换干预目标的 P2/P3 任务由 TQ-S、AD-MEC3、CD-S、AD-PATH4 承担。

---

## 2. 前沿工作“实际怎么构造任务”的核验表

以下条目来自本轮重新检查的 PDF/摘要，不是原则性转述。`PDF` 表示本轮已下载并检查全文；
`LOCAL-PDF` 表示已有本地全文；`METADATA` 表示只核验到书目信息，细节不得引用。

| 工作 | 它实际怎么构造任务 | CPT-World 继承什么 | 不继承什么 |
|---|---|---|---|
| CLadder (Jin et al., NeurIPS 2023; arXiv:2312.04350, PDF) | 采样 3-4 节点 graph motif（confounding/mediation/chain/fork/collider/IV 等） × Pearl rung × query 类型；CI engine 符号求 truth；模板转自然语言 | 任务 = 结构族 × 终局头 × answer 等价类；truth 由唯一 owner 计算；结构族做 split | 不继承故事、rung 当难度标签、counterfactual |
| CausalWorld (Ahmed et al., ICLR 2021; arXiv:2010.04296, PDF) | 参数化 task generator 共享因果结构；对所有暴露变量可做 do；用 intervention actors 定义训练/评估空间和 curriculum | generator-as-owner；干预是一等动作；结构族/参数轴做 split 与难度梯度 | 不继承机器人、视觉、连续状态 |
| Corr2Cause / A-CBO (arXiv:2605.27567, PDF) | 用 chain/fork 等 near-miss 图对：观测前提几乎或完全相同、Markov 等价类不同；把图选择移出 LLM，LLM 只回答“do(X) 是否改变 Y”，外部 Bayesian loop 按 EIG/后验收敛 | near-miss MEC 任务族；干预打破观测等价；query-discrimination 是任务核心；难度与图深度/motif 绑定 | 不继承“LLM 只做二元 oracle”的架构；CPT-World 中模型仍必须自己选干预和终局输出 |
| Wang et al., Bayesian Active Learning for Bivariate Causal Discovery (LOCAL-PDF `cpt_lit_review/wang25bp.txt`) | 二元 SCM；weak-causation 强度 ε；以 decisive-and-correct evidence 为决策目标（非单纯 EIG）；离散情形枚举干预值，DP 做多步最优 | 二元 CPT world、decision-focused oracle、精确 DP、弱效应做 N 轴 | 不继承连续优化；其 BN/ANM 扩展不进入第一批 seed |
| Tigas et al. (arXiv:2203.02016, PDF) | Bayesian causal discovery 下用 EIG 同时选干预 target 和 value；合成 ER/SF 图 | 世界后验 + 动作价值 + risk-cost envelope 的 oracle 形态 | 不继承非线性连续 SCM、可微 BO |
| Toth et al., ABCI (arXiv:2206.02063, PDF) | 不先恢复全图，直接在因果 query 的后验上设计干预 | query-focused 候选世界；P 轴策略树；oracle 只对 Q(W) 求价值 | 不继承 GP/加性噪声模型 |
| Sussex et al. (arXiv:2105.14024, PDF) | 设计**批量多变量联合干预**，用次模性给近似保证 | 组合动作空间与预算约束的意识；为未来 multi-target do 留 seam | 第一批 seed 仍只用单 target do，避免未冻结语义 |
| CausalBench (arXiv:2404.06349, PDF) | 真实 BN（2-109 节点）上构造 correlation/skeleton/causality 三个静态任务，控制图规模和 prompt 格式 | 任务头按“关联→骨架→方向”分层；符号/格式稳健性审计 | 不继承静态表格、真实语义变量名、大图 |
| CausalDS (arXiv:2607.08093, PDF) | 每场景采样隐藏 SCM；**观测层与概念 SCM 分离**（measurement model 只改估计难度，不改 identifiability）；按 motif/graft/identifiability 配置；不可识别时 abstain 是一等结果 | conceptual CPT-world 与 visible measurement 分离；identifiability gate；难度由合成配置轴控制 | 不继承故事、latent confounder、counterfactual、coding sandbox |
| CausaLab (arXiv:2605.26029, PDF) | 隐藏 SCM 生成 prior records/manipulator/reactor；模型预算内干预可干预子集；每步 DSL hypothesis；双评分：任务预测 vs 机制恢复 | 隐藏机制跨 evidence stream 共享；干预预算；终局任务分与机制恢复诊断分离 | 不继承连续 SCM、shift intervention、hidden disturbance、晶体故事 |
| CausalPitfalls (arXiv:2505.13770, PDF) | 6 类统计陷阱 × 15 挑战 × 5 个难度（提示逐渐减少）；SCM 生成数据；rubric 评分 | 每个任务族绑定一个明确陷阱和唯一正确实验策略；难度通过减少提示/要求显式调整变量实现 | 不继承 latent confounding/selection/mediation 等超出 V1 scope 的陷阱、开放故事 |
| NewtonBench (arXiv:2510.07172, LOCAL-PDF) | 对 canonical law 做 counterfactual shift 生成可扩展、抗记忆的隐藏机制；两条独立难度轴：intrinsic law 突变深度、extrinsic 系统复杂度；给出 solvability proof | 机制突变生成新颖隐藏世界；intrinsic/extrinsic 复杂度分离；证书必须含可解性证明 | 不继承连续方程、物理语义 |
| CausalGame (arXiv:2607.04293, PDF) | SCM 作为游戏引擎；14 个场景注入 selection/measurement error/hidden confounder；探索-部署两阶段预算；analytic optimum 校准难度 | 隐藏 SCM + 预算化主动实验 + analytic optimum 做可解性标尺 | 不继承 latent confounding/selection/story/报告 rubric |
| CauGym (arXiv:2602.06337, PDF) | 采样 10 节点 DAG → 语义化节点 → SCM 方程 → 七类因果任务 → DoWhy 求 truth；再构造 omitted/redundant/insufficient 变体 | 一个 SCM 派生多任务、truth owner 独立、模板化 DSL | 不继承训练/post-training、连续 SCM、故事、counterfactual 任务 |
| Optimal stopping for sequential BED (arXiv:2509.21734, PDF) | 把设计和停时作为耦合决策；证明固定设计下最优停时是 continuation value 阈值 | risk-cost envelope 的 optimal stopping oracle；终局 STOP 的价值比较 | 不继承其特定连续设计空间 |
| Butkus & Kriegeskorte, Causal Discovery and Inference through Next-Token Prediction (NeurIPS 2025, DOI 10.52202/085713-2292, METADATA) | 书目信息已核验；v2 中 DO/OBS/DATA/INFERENCE token 细节尚无全文，**不得在本文件或论文中继续引用其细节**，PDF 到手后再补 | 只保留“结构化 token 序列可作为实验 transcript”的候选方向 | 在核验前不继承任何具体 token 语义 |

核验结论：没有发现一篇工作同时满足 “hidden finite binary CPT-world + explicit local readout cost + A/P/N exact oracle”。
这是 CPT-World 的空白；但以上每一条继承都必须能指到上表的具体机制，不能只写原则。

---

## 3. V3 观测与动作合同（CANDIDATE，比 v2 更窄）

### 3.1 变量类型

每个变量显式声明：

```yaml
variables:
  - id: T
    domain: [0, 1]
    manipulable: true
    readable: true
  - id: S
    domain: [0, 1]
    manipulable: false
    readable: true
  - id: Y
    domain: [0, 1]
    manipulable: false
    readable: true
```

### 3.2 动作

```json
{
  "type": "intervene",
  "target": "X",
  "value": 1,
  "measure": ["S", "Y"],
  "batch_size": 8
}
```

- `target` 必须是 `manipulable=true` 的变量；`value ∈ domain`。
- `measure` 是可读变量的非空子集，且 `target ∉ measure`。
- 任务声明 \(k_{\max}\)，要求 \(1\le |measure|\le k_{\max}\)。
- 环境只返回 `measure` 中变量的 joint count table；未选择变量不返回。
- 原子样本成本 = `|measure|`；batch 成本 = `batch_size × |measure|`；episode 成本 \(C=\sum_t b_t|M_t|\)。
- 终局仍可任意早期 STOP；STOP 前成本为 0 合法。
- 第一批 seed **不引入多目标 do**；CDE/多目标联合干预只保留为 DRAFT seam，语义 owner 未冻结前禁止使用。

### 3.3 初始证据 D0

每个任务显式声明 D0：
- `D0=EMPTY`：没有 observational sample；
- `D0=EXACT_OBS_JOINT`：公开精确观测联合（按任务符号域渲染成频率/有理数表）；
- 其它 D0 类型必须先通过“D0 不泄漏 true world index”门。

D0 用于 A 轴控制，不用于成本计算。

---

## 4. A/P/N 的精确定义（沿用 v2，补 D0 与局部观测）

公开 \((\mathcal W,\pi,D_0)\)；只隐藏 true world index 与未来随机观测。
oracle 只能使用公开 prior、公开 D0 和模型可见 history；不得使用 hidden world。

### 4.1 A

\[
A=-\sum_c\pi(c\mid D_0)\log_2\pi(c\mid D_0)\quad\text{bits},
\]

其中 \(c\) 是终局答案等价类（TQ 为连续 query 时，A 用公开先验下 answer 值的 Bayes risk 或离散化等价类；
本文件所有主 seed 均为有限 answer class，先用上式）。

### 4.2 P

\[
P2 \iff \exists c,\ \mathcal R^*_{\rm fixed}(c)>\mathcal R^*_{\rm adaptive}(c),
\]
且最优自适应策略树深度为 2。
\[
P3 \iff \exists c,\ \mathcal R^*_{\rm fixed}(c)>\mathcal R^*_{\rm adaptive}(c),
\]
且最优自适应策略树深度至少 3。
否则 P1。

固定策略的干预/测量序列不依赖观测；自适应策略的每一步可依赖 history。
成本包含 \(|M|\)，所以“先买便宜边际、再买联合”的策略价值会被 envelope 显式算出来。

### 4.3 N

对每对不同 answer class \((W,W')\)，在其最优策略树上的首个分离节点处：

\[
N=\min_{W\ne W'} d_{\rm TV}\bigl(P(\mathrm{obs}\mid do(a),M,W),P(\mathrm{obs}\mid do(a),M,W')\bigr).
\]

`obs` 是 `measure` 返回的完整 joint count 的单样本分布；N 是逐样本分辨力。
oracle 再由 N 转 \(n_{\min}(\delta)\)，不把 N 当预算。

---

## 5. 防坍缩任务准入门（V3）

每个 seed 必须通过：

1. **Query minimal sufficient readout gate**：oracle 求 \(Q(W)\) 的最小充分 readout 变量集 \(S_Q\)。
   若 \(|S_Q|\le2\) 且任务声称第三个变量提供难度，REJECT。
2. **Universal full-table 非支配**：\(\pi_{\rm table}\) 若可行，不得在 risk-cost envelope 上处处最优。
3. **测量选择价值**：存在 history，使不同 `measure` 子集有不同的 continuation value。
4. **可干预性/可读性约束真实生效**：至少一个 query-relevant 变量 readonly，或
   \(k_{\max}<\) 某条可行路径需要的 readout 数，从而不存在零成本的 full joint 捷径。
5. **答案非平凡**：answer class ≥2，A>0。
6. **Oracle 有解**：有限 belief MDP 可精确求出 \(Q^*,V^*\)，统一 cap 内可达到预注册风险。
7. **无捷径泄漏**：prompt 不含 true world、最优 measure、q/e、难度标签、oracle action。
8. **D0 不短路**：若 D0 已点识别 Q(W)，该任务只能作为 observational 标尺，不得宣称为干预任务。
9. **Symbol orbit**：role 置换、target/measure/answer 顺序全交叉。
10. **基线阶梯**：random < correlation-only < fixed < greedy < adaptive ≤ oracle。

---

## 6. Seed 族

记号：`low = 1/2 - e`，`high = 1/2 + e`。所有 CPT 都是精确有理数；所有边际都是 exact Fraction。

### 6.1 TQ-CB：条件双向 target query（P1；A/N 标尺）

变量：`X,Y` manipulable；`S` readonly，\(P(S=1)=1/2\)。
终局输出：

\[
(\hat\tau_{X\to Y\mid S=1},\ \hat\tau_{Y\to X\mid S=1}),
\qquad
\tau_{A\to B\mid S=1}=P(B=1\mid do(A=1),S=1)-P(B=1\mid do(A=0),S=1).
\]

四个候选世界（\(\sigma\in\{+1,-1\}\)）：

`Fσ`（DAG: X→Y, S→Y）：

\[
P(Y=1\mid X=0,S=1)=\tfrac12-\sigma e,\quad
P(Y=1\mid X=0,S=0)=\tfrac12+\sigma e,
\]
\[
P(Y=1\mid X=1,S=1)=\tfrac12+\sigma e,\quad
P(Y=1\mid X=1,S=0)=\tfrac12-\sigma e.
\]

`Rσ`（DAG: Y→X, S→X），对称：

\[
P(X=1\mid Y=0,S=1)=\tfrac12-\sigma e,\quad
P(X=1\mid Y=0,S=0)=\tfrac12+\sigma e,
\]
\[
P(X=1\mid Y=1,S=1)=\tfrac12+\sigma e,\quad
P(X=1\mid Y=1,S=0)=\tfrac12-\sigma e.
\]

| World | \((\tau_{X\to Y|S=1},\tau_{Y\to X|S=1})\) |
|---|---|
| F+ | \((+2e,0)\) |
| F- | \((-2e,0)\) |
| R+ | \((0,+2e)\) |
| R- | \((0,-2e)\) |

关键性质（已用 exact Fraction 核验）：
- `F+` 与 `R+` 的观测联合**完全相同**；`F-` 与 `R-` 的观测联合完全相同。
  因此 D0=EXACT_OBS_JOINT 只把后验压到 {F+,R+} 或 {F-,R-}，不能定方向。
- `do(X=1), measure={Y}` 在四个世界都是 Bern(1/2)，**零信息**；`measure={S}` 也零信息。
- `do(X=1), measure={S,Y}` 分离全部四类：TV(F+,F-)=2e，TV(Fσ,R·)=e。
  同理 `do(Y=1), measure={S,X}` 分离全部四类。
- 最小充分 readout 集包含 S 与 outcome，因此 QN 的“只测两端点”捷径不存在。

| Seed | e | 答案效应 | N | A(D0=EMPTY) | A(D0=EXACT_OBS_JOINT) | P |
|---|---:|---|---:|---:|---:|---|
| TQ-CB-A2-P1-N-EASY | 2/5 | 4/5 | 2/5 | 2 bits | 1 bit | P1 |
| TQ-CB-A2-P1-N-MEDIUM | 1/5 | 2/5 | 1/5 | 2 bits | 1 bit | P1 |
| TQ-CB-A2-P1-N-HARD | 1/20 | 1/10 | 1/20 | 2 bits | 1 bit | P1 |

`k_max=2`，合法 measure 为 `{Y}`、`{S}`、`{S,Y}`。`{Y}`/`{S}` 是零信息 distractor。
最优策略是固定设计：`do(X=1), measure={S,Y}`（或对称地 `do(Y=1), measure={S,X}`）。
**Collapse audit**：公开候选世界使一个干预臂足以定类，因此它只是 P1 筛查题；
它测试的是“能否拒绝便宜但无信息的 Y-only，购买 S,Y joint”，不是双向多臂设计。
主 benchmark 不得把它的 P 标成 P2。

### 6.2 TQ-S：subgroup target query（P2；保留并收紧 v2 HT）

变量：`T` manipulable；`S,Y` readonly；\(P(S=1)=1/2\)。
终局输出两个 subgroup effects：

\[
(\hat\tau_0,\hat\tau_1),\quad
\tau_s=P(Y=1\mid do(T=1),S=s)-P(Y=1\mid do(T=0),S=s).
\]

四个世界：

| World | 机制 | \((\tau_0,\tau_1)\) |
|---|---|---|
| W0 | T→Y，S 独立，效应 +q | \((+q,+q)\) |
| W1 | T→Y，S 独立，效应 -q | \((-q,-q)\) |
| W2 | T=1 时两 S 层 Y 均值均为 1/2；T=0 时 S=0 为 1/2-q、S=1 为 1/2+q | \((+q,-q)\) |
| W3 | T=1 时两 S 层均为 1/2；T=0 时 S=0 为 1/2+q、S=1 为 1/2-q | \((-q,+q)\) |

W0/W1 的 `do(T=1)` Y 边际为 \(1/2\pm q/2\)；W2/W3 为 1/2。
因此最优策略树：

1. `do(T=1), measure={Y}`，成本 1/sample：
   - 支持 W0 → STOP，回答 \((+q,+q)\)；
   - 支持 W1 → STOP，回答 \((-q,-q)\)；
   - 支持 {W2,W3} → 转第 2 步。
2. `do(T=0), measure={S,Y}`，成本 2/sample，分离 W2/W3。

- A=2 bits；P=P2；stage-1 TV=q/2，N=q/2。
- 最小充分 readout 集包含 S；`{Y}` 便宜但不充分，`{S,Y}` 必须显式购买。
- full-table policy（两臂都买 {S,Y}）不是处处最优：W0/W1 时 cheap arm 提前 STOP 更优。

| Seed | q | N | A | P |
|---|---:|---:|---:|---|
| TQ-S-A2-P2-N-EASY | 1/2 | 1/4 | 2 bits | P2 |
| TQ-S-A2-P2-N-MEDIUM | 1/4 | 1/8 | 2 bits | P2 |
| TQ-S-A2-P2-N-HARD | 1/20 | 1/40 | 2 bits | P2 |

### 6.3 AD-MEC3：三节点 Markov 等价类主动发现（P2）

变量 `X,M,Y` 均可干预；`k_max=1`（每轮只能测一个非目标变量）。
候选世界是同一 Markov 等价类中的三个 DAG：

```text
C:  X -> M -> Y
F:  X <- M -> Y
C': Y -> M -> X
```

CPT 规则：根节点 Bern(1/2)；每条有向边 \(U\to V\) 的
\(P(V=1\mid U=0)=1/2-e,\ P(V=1\mid U=1)=1/2+e\)。

性质（已核验）：
- 三个世界的观测联合**完全相同**，所以 D0（无论 EMPTY 还是观测联合）都不能定方向。
- `do(X=1), measure={M}`：C 的 M 均值为 \(1/2+e\)；F、C' 为 \(1/2\)。TV=e。
- `do(Y=1), measure={M}`：C' 的 M 均值为 \(1/2+e\)；F（以及 C）为 \(1/2\)。TV=e。
- 因此最优策略树：
  1. `do(X=1), measure={M}`：高 → C，STOP；中 → {F,C'}。
  2. `do(Y=1), measure={M}`：高 → C'；中 → F。
- 第一轮结果改变第二轮干预 target，P=P2。
- `k_max=1` 使 full joint readout 不可行；测 `Y` 而不测 `M` 的第一轮动作价值更低。

| Seed | e | A | P | N |
|---|---:|---:|---:|---:|
| AD-MEC3-A2.585-P2-N-EASY | 2/5 | log2 3 ≈ 2.585 bits | P2 | 2/5 |
| AD-MEC3-A2.585-P2-N-MEDIUM | 1/5 | log2 3 ≈ 2.585 bits | P2 | 1/5 |
| AD-MEC3-A2.585-P2-N-HARD | 1/20 | log2 3 ≈ 2.585 bits | P2 | 1/20 |

### 6.4 CD-S：协变量条件下的因果决策（P2；decision head 的候选合同）

变量 `T` manipulable；`S,Y` readonly。终局输出一个决策规则：

```json
{"type": "answer", "policy": [0, 1]}
```

语义：`policy[s]` 是对 S=s 的受试者选择的 `do(T=t)`。
效用 \(u_s(t;W)=P(Y=1\mid do(T=t),S=s)\)；
regret：

\[
R(W,a)=\sum_{s\in\{0,1\}}P(S=s)\left[\max_t u_s(t;W)-u_s(a_s;W)\right].
\]

四世界与 6.2 同构（但终局是离散 policy）：

| World | 最优 policy | A | P | N |
|---|---|---:|---:|---|
| W0 | `[1,1]` | 2 bits | P2 | q/2 |
| W1 | `[0,0]` | 2 bits | P2 | q/2 |
| W2 | `[1,0]` | 2 bits | P2 | q/2 |
| W3 | `[0,1]` | 2 bits | P2 | q/2 |

最优策略树同 6.2：先 `do(T=1), measure={Y}`；若支持 W0/W1 直接定策；
若支持 {W2,W3} 再 `do(T=0), measure={S,Y}` 分离 subgroup。
N 轴取 q=1/2,1/4,1/20 三档。
该族把 “RCT + subgroup + 决策” 变成可精确判分的有限问题，并且不引入 latent confounder。

### 6.5 AD-PATH4：四节点等价类，P3（DRAFT，待 oracle.py）

变量 `V1,V2,V3,V4` 均可干预；`k_max=1`。
候选世界是 path skeleton `V1-V2-V3-V4` 的四个无 collider 定向：

```text
O1: V1 -> V2 -> V3 -> V4
O2: V1 <- V2 -> V3 -> V4
O3: V1 <- V2 <- V3 -> V4
O4: V1 <- V2 <- V3 <- V4
```

CPT 同 6.3。四个世界观测联合完全相同；A=2 bits。
核验到的策略树深度为 3：

1. `do(V1=1), measure={V2}`：O1 高；{O2,O3,O4} 中。
2. `do(V2=1), measure={V3}`：O2 高；{O3,O4} 中。
3. `do(V4=1), measure={V3}`：O4 高；O3 中。

每层 TV=e，因此 N=e。
这是 P3 候选，用于检验 adaptive depth 上界；在 exact oracle 给出
risk-cost envelope 与 fixed-design gap 前保持 DRAFT，不得进入采样。

---

## 7. 难度格（当前候选）

| Cell | A | P | N | 用途 |
|---|---|---|---|---|
| TQ-CB-A2-P1-N-EASY/MEDIUM/HARD | 2 | P1 | e | N 轴标尺；parser/tape/测量选择 |
| TQ-CB-A1-P1-N-MEDIUM | 1 | P1 | e | A 轴（D0 改变先验歧义） |
| TQ-S-A2-P2-N-EASY/MEDIUM/HARD | 2 | P2 | q/2 | target-query 主任务 |
| AD-MEC3-A2.585-P2-N-EASY/MEDIUM/HARD | log2 3 | P2 | e | discovery 主任务 |
| CD-S-A2-P2-N-EASY/MEDIUM/HARD | 2 | P2 | q/2 | decision 主任务 |
| AD-PATH4-A2-P3-N-MEDIUM | 2 | P3 | e | P3 DRAFT |

尚未满足 S4 的单因素配对：
- P1→P2 配对：需要 oracle 在**同 A、同 N** 下找到 fixed 等价变体；
  候选做法是改变动作空间/候选世界集，而不是改 q。
- P2→P3 配对：用 AD-MEC3 与 AD-PATH4 的 envelope 比较，但 A 不同（log2 3 vs 2），
  必须先证明旁轴漂移受控，否则只能作为 family scaling，不能作为 P 轴单因素证据。
- 在 `oracle.py` 给出 envelope 前，不得宣称 A/P/N 已条件可控。

---

## 8. 实现顺序与 owner 门

1. `world_space.py` / `query_truth.py`：实现通用 `WorldSpec`（DAG + rational CPT + single-target hard-do），
   唯一 owner 负责任意 `measure` 子集的 exact marginal/joint counts。
2. `rendering.py` / `world_runtime.py`：`manipulable/readable` 类型、`measure` 字段、`k_max`、D0 渲染、strict JSON 解码。
3. `oracle.py`：对候选世界与 local readout 观测求 posterior、Q-value、最优策略树、
   risk-cost envelope、A/P/N 证书；全部用 Fraction/整数计数。
4. `seeds.py`：加载 TQ-CB、TQ-S、AD-MEC3、CD-S；AD-PATH4 标记 `DRAFT`。
5. 对每个 seed 自动跑第 5 节 10 个 gate；REJECT 的输出必须可复现。
6. QN 从主 benchmark 移除，只保留为 interface/tape/N 轴诊断。

分布性质的初步 exact 核验在 `scripts/verify_task_family_seed_math.py`；
它只核验本文中的 TV/观测等价/零信息边际，不替代 future hard-do owner 或 oracle。

未冻结仍阻塞 VERIFIED：
- hard-do/interventional-law 唯一 owner；
- 观测返回模式的唯一 owner；
- terminal head 的语义（TQ-CB/TQ-S 是否作为 D11 二维数值 query 的新子合同；AD/CD 的 proper score/regret 合同）；
- public prior Bayes risk、同成本 oracle risk 与 action-keyed tape。

---

## 9. 当前裁决

```text
V2 LOCAL-READOUT CONTRACT: KEPT_AND_TIGHTENED
TQ-CB-A2/A1-P1-N-*: CANDIDATE
TQ-S-A2-P2-N-*: CANDIDATE
AD-MEC3-A2.585-P2-N-*: CANDIDATE
CD-S-A2-P2-N-*: CANDIDATE
AD-PATH4-A2-P3-N-*: DRAFT
VERIFIED-SEED: BLOCKED_BY_OWNER_AND_ORACLE
```

在 `oracle.py` 复算通过前，任何 seed 不得进入采样分布。
