> **已由 `docs/task-family-seeds-v3.md` 取代作为当前设计稿；本文保留为 v2 历史与 HT/TQ-S 族来源。**
# CPT-World 任务族 Seed 设计 v2（CANDIDATE）

> 本文件取代 `task-family-seeds.md` v0.3。
> v0.3 的坍缩原因是：`INTERVENE` 默认返回所有变量的完整 joint counts。
> 在 3 变量、双方可干预的设置下，模型只要干预 X 和 Y 并读取 joint counts，就获得了完整 interventional law；之后任何 query/discovery/decision 都退化成查表。
> v2 的修复不是“把题变难”，而是改变观测合同：**读什么变量是显式、受限、有成本的实验选择**。

---

## 1. 坍缩的形式化诊断

设世界 \(W\)，动作 \(a\)，观测 \(O(a,W)\)。若对所有 \(a\) 都返回全部变量的 joint distribution，则存在一个与查询无关的 universal policy：

\[
\pi_{\rm table}:
\quad
\text{对每个合法 do-arm 收集 full joint，然后读表回答}.
\]

对任何 task head，只要预算足够，\(\pi_{\rm table}\) 都能达到接近零风险。此时 benchmark 测的不是因果实验设计，而是“是否愿意枚举全表”。

因此 v2 的任务准入增加一条：

> 如果 universal full-table policy 在 risk-cost envelope 上对每个成本都达到最优或 \(\varepsilon\)-最优，该任务 REJECT。

---

## 2. v2 观测合同：explicit local readout

### 2.1 动作

```json
{
  "type": "intervene",
  "target": "T",
  "value": 1,
  "measure": ["Y"],
  "batch_size": 8
}
```

- `target` 必须是本任务声明的 **manipulable** 变量。
- `measure` 是 **readable** 变量的非空子集。
- 每个任务声明测量容量 \(k_{\max}\)，且 \(1\le|\text{measure}|\le k_{\max}\)。
- 环境只返回 `measure` 中变量的 joint counts；未选择的变量不返回。
- 一个 atomic sample 的成本为 \(|\text{measure}|\)；一个 batch 成本为 \(b\cdot|\text{measure}|\)。
- 对某些变量不可做 `do`：例如 outcome \(Y\) 和基线协变量 \(S\) 只读，不可干预。

### 2.2 直接效果

- 完整 joint counts 不再自动出现。
- 对 \(d\) 个非目标可读变量，若 \(k_{\max}<d\)，full joint 根本不可行；若可行，其成本也最高。
- 模型必须显式选择 “这一轮测哪个变量/哪些变量”。

### 2.3 与前沿工作的对应

- 现代 Bayesian experimental design：观测模型是设计的一部分，readout 子集就是 design。
- 主动因果发现 / targeted cause discovery：只测与查询相关的变量，不为全图收集全表。
- RCT / best-arm identification：干预哪只 arm、测哪个 endpoint、是否测协变量，是独立的成本决策。

---

## 3. A / P / N 定义

设候选世界集 \(\mathcal W\)、uniform prior、初始证据 \(D_0\) 为空；本 seed suite 公开候选世界类和 prior，只隐藏 `true_world_index`。

### 3.1 A

\[
A=-\sum_c\pi(c)\log_2\pi(c)\quad\text{bits},
\]

其中 \(c\) 是终局答案等价类。连续 query 的 v2 变体可改用公开先验 Bayes risk \(R_{\rm ref}\)；本文件先只放有限答案类 seed。

### 3.2 P

\[
P2 \iff \exists c,\quad \mathcal R^*_{\rm fixed}(c)>\mathcal R^*_{\rm adaptive}(c).
\]

其中 \(\mathcal R^*_{\rm fixed}(c)\) 是期望成本 \(c\) 下最优固定设计（干预序列不依赖观测）的风险，\(\mathcal R^*_{\rm adaptive}(c)\) 是最优自适应设计的风险。P1 表示对所有 \(c\) 两者相等。

### 3.3 N

对每对不同答案类，取它们**首次被区分**的 oracle 决策节点上的逐样本 TV 距离：

\[
N=\min_{W,W':\mathrm{answer}(W)\ne\mathrm{answer}(W')}d_{\rm TV}
\bigl(P(\mathrm{obs}\mid do(a),M,W),P(\mathrm{obs}\mid do(a),M,W')\bigr).
\]

oracle 再把 N 转换成 \(n_{\min}(\delta)\)。


---

## 3.5 与已有工作的构造机制对齐

以下映射用于防止“设计原则无出处”。`LOCAL` 表示本地文献笔记已核验到论文页/摘要；`MEMORY` 表示需你本地 PDF 再核验细节。

| 工作 | 它实际怎么构造任务 | CPT-World 继承什么 | 不继承什么 |
|---|---|---|---|
| CLadder, Jin et al., NeurIPS 2023（MEMORY） | 采样图 motif（chain/fork/collider/confounded/mediation 等）× Pearl rung × query 类型，用 SCM 符号计算答案，再套故事模板和 distractor | 任务 = 结构族 × terminal head × 答案等价类；truth 由 owner 计算 | 不用自然语言故事；不把 rung 直接当难度标签 |
| CausalWorld, Ahmed et al., ICLR 2021（MEMORY） | 环境生成器持有 SCM；动作是显式 do；按结构/可观测性变化生成任务族；用下游成功和迁移区分能力 | generator-as-owner；干预是一等动作；按结构族做 split | 不继承机器人域和视觉观测 |
| CausalPitfalls, ICLR 2026（LOCAL，见 AI 研究文献笔记 [49]） | 按混杂、selection、mediation、collider、counterfactual 等统计陷阱生成带 ground-truth 的案例 | 每个任务族必须对应一个明确陷阱；seed 必须有唯一正确实验策略 | 不生成开放故事，仍用 finite CPT-world |
| Butkus & Kriegeskorte, NeurIPS 2025（LOCAL） | 合成 SCM 编码为 DO/OBS/DATA/INFERENCE token 序列，训练 next-token predictor 做发现/推断 | 结构化 DSL；实验 transcript 本身就是任务表示 | 不做模型训练；不用其 linear-Gaussian 连续 SCM |
| COAT, NeurIPS 2024（LOCAL） | LLM 提变量 → 离线标注 → FCI 建图 → 反馈循环 | 只说明变量发现是独立缺口 | LLM 和 FCI 都不进入 truth generator |
| Tigas et al., NeurIPS 2022；Sussex et al., NeurIPS 2023；Toth et al., NeurIPS 2022（MEMORY） | 用 EIG/后验/预算优化选择下一轮干预目标；部分工作在 POMDP/belief 上求策略 | A/P/N 的 oracle 定义、P 轴策略树、risk-cost envelope | 不让可微发现器替代 exact finite-DP |
| Causal bandits / best-arm identification（Lattimore et al. 2016 等，MEMORY） | 有限臂、Bernoulli outcome、Bayesian posterior、regret 作为目标 | CD/RCT 任务族的终局 decision、regret、N 轴 | 不把因果图简化为互不相关 arms |

**本地文献笔记没有覆盖到的缺口**：我尚未找到一篇工作同时满足 “hidden finite CPT-world + 显式 readout 成本 + 连续 effect query + 有 exact oracle 的难度格”。这是 CPT-World 论文需要声明的空白；但这个空白本身必须由上述工作的构造机制逐条支撑，不能只写原则。

---

## 4. 防坍缩任务准入门

每个 seed 必须通过：

1. **局部性**：每个动作只返回 `measure` 指定变量；没有隐式 full joint。
2. **可干预性约束**：至少一个与答案有关的变量不可干预（如 Y 只读）。
3. **query 最小充分变量集**：oracle 必须计算 query \(Q(W)\) 的最小充分 readout 集 \(S_Q\)。若 \(|S_Q|\le2\) 且难度声明依赖 \(S_Q\) 以外的变量，REJECT。
4. **full-table 非支配**：\(\pi_{\rm table}\) 若可行，不得在 risk-cost envelope 上处处最优。
5. **测量选择价值**：存在一个 history，使不同 `measure` 子集有不同的 continuation value。
6. **答案非平凡**：答案类至少 2 个；A > 0。
7. **oracle 有解**：有限信念 MDP 可精确求出 \(Q^*,V^*\)，且在统一 cap 下可达到预注册风险。
8. **无捷径泄漏**：prompt 不包含 true world、最优 measure、q 或难度标签。
9. **symbol orbit**：所有 role、target order、readout order 全交叉。
10. **基线阶梯**：random < fixed < adaptive ≤ oracle。

---

## 5. Seed family 1：RCT / heterogeneous treatment effect query（HT）

### 5.1 任务定义

变量：

- `T`：manipulable，treatment。
- `S`：readable only，baseline covariate。
- `Y`：readable only，outcome。

测量容量 \(k_{\max}=2\)。合法测量子集为 `{Y}` 或 `{S,Y}`；`{S}` 允许但无终局价值。

终局答案：

\[
(\hat\tau_0,\hat\tau_1),\qquad
\tau_s=P(Y=1\mid do(T=1),S=s)-P(Y=1\mid do(T=0),S=s).
\]

这是 RCT 的 subgroup treatment effect，不是故事题，也不允许干预 outcome。

### 5.2 HT-A2-P2-N-MEDIUM：先读廉价边际，再按需读联合

令 \(q=1/2\)。四个候选世界：

| World | 机制 | \((\tau_0,\tau_1)\) |
|---|---|---|
| W0 | T→Y，效应 +q；S 独立 | \((+q,+q)\) |
| W1 | T→Y，效应 -q；S 独立 | \((-q,-q)\) |
| W2 | T=1 时两个 S 层的 Y 均值均为 \(1/2\)；T=0 时 S=0 为 \(1/2-q\)，S=1 为 \(1/2+q\) | \((+q,-q)\) |
| W3 | T=1 时两个 S 层均为 \(1/2\)；T=0 时 S=0 为 \(1/2+q\)，S=1 为 \(1/2-q\) | \((-q,+q)\) |

`do(T=1), measure={Y}` 下：

- W0 的 Y 均值为 \(1/2+q/2\)；
- W1 为 \(1/2-q/2\)；
- W2/W3 均为 \(1/2\)。

因此：

- A = 2 bits。
- P = P2：
  1. 先做 `do(T=1), measure={Y}`，成本 1/sample。
  2. 若后验支持 W0 或 W1，STOP。
  3. 若后验支持 {W2,W3}，改做 `do(T=0), measure={S,Y}`，成本 2/sample，直到区分 W2/W3。
- N = 1/4：stage 1 区分 W0 与 {W2,W3} 的 TV 为 \(q/2=1/4\)；stage 2 区分 W2/W3 的 TV 为 1。

**为什么不再坍缩**：最便宜的 Y-only 测量无法识别 W2/W3；模型必须先买 treatment 臂的廉价边际，再按结果决定是否购买控制臂的 `{S,Y}` joint readout。没有任何一个干预会自动带回整张表。

### 5.3 HT-A2-P2-N-EASY / N-HARD

| Seed | q | stage-1 TV | A | P | N |
|---|---:|---:|---:|---:|---:|
| HT-A2-P2-N-EASY | 1/2 | 1/4 | 2 bits | P2 | 1/4 |
| HT-A2-P2-N-MEDIUM | 1/4 | 1/8 | 2 bits | P2 | 1/8 |
| HT-A2-P2-N-HARD | 1/20 | 1/40 | 2 bits | P2 | 1/40 |

W0/W1 按 `Edge(T -> Y; ±q)` 替换；W2/W3 保持 T=1 两层的 Y 均值为 \(1/2\)，并把 T=0 的两层均值替换为 \(1/2\mp q\) 与 \(1/2\pm q\)。所有 CPT 仍为精确有理数。

### 5.4 HT-A2-P1-N-MEDIUM：固定设计，但联合测量仍不可避免

令 \(q=1/2\)。四个世界都使 `do(T=0/1), measure={Y}` 的 Y marginal 恒为 1/2，因此廉价边际零信息：

| World | T=0: P(Y=1|S=0,S=1) | T=1: P(Y=1|S=0,S=1) | \((\tau_0,\tau_1)\) |
|---|---|---|---|
| W0 | 1/4, 3/4 | 3/4, 1/4 | \((+1/2,-1/2)\) |
| W1 | 3/4, 1/4 | 1/4, 3/4 | \((-1/2,+1/2)\) |
| W2 | 3/8, 5/8 | 5/8, 3/8 | \((+1/4,-1/4)\) |
| W3 | 5/8, 3/8 | 3/8, 5/8 | \((-1/4,+1/4)\) |

- A = 2 bits。
- 最优设计：固定执行 `do(T=0), measure={S,Y}` 与 `do(T=1), measure={S,Y}`；P1。
- N 由 oracle 取首分离节点最小 TV（候选值 1/8，待复算）。

这个 seed 证明：即使 P1，任务也不是查表，因为唯一可用的信息必须通过 2-cost joint readout 购买。

---

## 6. Seed family 2：Active discovery with local readouts（AD）

### 6.1 任务定义

变量 `X,Y,Z,W`；W 是合法但无信息的 distractor。
测量容量 \(k_{\max}=2\)，且所有变量均可干预；但每个动作必须显式选择最多两个 readout。
候选世界：X,Y,Z 之间恰好一条正边，q=+1/2；W isolated。
答案：输出这条有向边。
候选数 6，A = \(\log_2 6\approx 2.585\) bits。

### 6.2 AD-A2-P2-N-MEDIUM

最优策略树：

1. `do(X=1), measure={Y,Z}`，成本 2/sample。
   - Y 均值 3/4 且 Y⊥Z → `X->Y`，STOP。
   - Z 均值 3/4 且 Y⊥Z → `X->Z`，STOP。
   - Y,Z 都像 Bern(1/2) 且独立 → 真边在 `{Y->X, Z->X}`。
   - Y 与 Z 相关且边际均为 1/2 → 真边在 `{Y->Z, Z->Y}`。
2. 第二动作由第一轮结果决定：
   - `{Y->X, Z->X}`：`do(Y=1), measure={X}`。
     X 均值 3/4 → `Y->X`；X 均值 1/2 → `Z->X`。
   - `{Y->Z, Z->Y}`：`do(Y=1), measure={Z}`。
     Z 均值 3/4 → `Y->Z`；Z 均值 1/2 → `Z->Y`。

- P = P2：第二轮 measure 目标依赖第一轮 joint pattern。
- N = 1/4。
- 若模型选择 `do(X=1), measure={Y}`（成本 1），它只能检测 `X->Y`；若选择 `measure={W}`，信息为零。不同 readout 的 continuation value 不同。

**为什么不再坍缩**：没有任何动作自动返回 X,Y,Z 的完整 joint；一次只能读两个非目标变量，且必须为 readout 支付成本。

---

## 7. 现有 QN 降级：pairwise marginal query 不得进入主 benchmark

你指出的问题必须写死：若终局对象只是

\[
(\tau_{X\to Y},\tau_{Y\to X}),
\]

那么它的最小充分实验集合确实只包含 X,Y 两个变量；测量任何第三个变量都不会降低 Bayes risk。这是 query 本身的数学性质，不能用任何观测约束修复。因此：

- 现有 QN 只保留为 **interface/parser/tape 诊断**，不进入主 benchmark 任务族；
- 任何 query 在准入前必须通过 **最小充分变量集 gate**：
  1. 由 oracle 求出 query \(Q(W)\) 在所有 legal action 下的最小充分 readout 变量集 \(S_Q\)；
  2. 若 \(|S_Q|\le 2\) 且任务声称难度来自 \(S_Q\) 以外的变量，REJECT；
  3. 若存在一个只测 \(S_Q\) 的 oracle 与使用全部变量的 oracle 达到相同 risk-cost envelope，REJECT。

现有 QN 在 v2 合同下变成：

```text
合法动作：do(X=0/1) 或 do(Y=0/1)；
每轮只能 measure 另一个 focal 变量，cost = 1/sample。
```

| Seed | q | 区分动作 | N | A | P |
|---|---:|---|---:|---:|---:|
| TQ-LOC-N-EASY | 4/5 | do(X=1), measure Y | 2/5 | 1 bit | P1 |
| TQ-LOC-N-MEDIUM | 2/5 | 同上 | 1/5 | 1 bit | P1 |
| TQ-LOC-N-HARD | 1/10 | 同上 | 1/20 | 1 bit | P1 |

这些 seed 只用于验证局部观测 parser 和 N 轴标尺，不用于能力结论。

---

## 8. 下一族：RCT with passive/active cost（DRAFT）

为逼近真实 RCT 构建，下一候选族在 v2 合同上增加：

```text
PASSIVE(measure, batch)：
  不干预，从 observational law 采样；
  成本 = |measure|/sample；
  在存在 measured confounding 时，passive 数据不能替代 hard-do。
```

候选世界：C→T、C→Y、T→Y 的有无与符号组合；T 是唯一 manipulable 变量；C,Y 只读。
终局：估计 ATE 或选择 treatment。
该族必须通过 “passive shortcut 非充分” 门：若 passive 联合分布已经能点识别 query，REJECT。

本文件暂不把该族列为 CANDIDATE seed，等观测合同与 passive 语义 owner 冻结后再补。

---

## 9. 实现顺序

1. `query_truth.py`：通用 `WorldSpec` + `do` 后对任意 `measure` 子集求 exact marginal/joint counts。
2. `world_runtime.py`：`measure` 字段、readable/manipulable 变量类型、\(k_{\max}\)、按 measure 渲染 counts。
3. `oracle.py`：对候选世界和 local readout 观测求 posterior、Q-value、risk-cost envelope、P/N 证书。
4. `seeds.py`：加载本文件 seed，先实现 `TQ-LOC-N-*` 和 `HT-A2-P2-N-*`。
5. 对每个 seed 跑 full-table dominance 测试和 baseline ladder；未通过的 REJECT。

## 10. 当前裁决

```text
V0.3 FULL-JOINT CONTRACT: REJECTED (collapse)
V2 LOCAL-READOUT CONTRACT: CANDIDATE
TQ-LOC-N-EASY/MEDIUM/HARD: DIAGNOSTIC-ONLY (pairwise query)
HT-A2-P2-N-EASY/MEDIUM/HARD: CANDIDATE
HT-A2-P1-N-MEDIUM: CANDIDATE
AD-A2-P2-N-MEDIUM: CANDIDATE
RCT-PASSIVE/ACTIVE: DRAFT
CDE/mediation query requiring joint do(T,M): DRAFT
```

在 `oracle.py` 复算通过前，任何 seed 不得进入采样分布。
