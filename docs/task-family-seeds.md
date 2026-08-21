# CPT-World 任务族 Seed 设计 v0.3（SUPERSEDED）

> **本文件已被 `docs/task-family-seeds-v2.md` 取代。**
> 原因：v0.3 沿用完整 joint counts 观测；该合同下只要干预两个端点即可获得整张 interventional law，任务坍缩为查表。
> 新设计采用“显式测量子集 + 可干预性约束 + 测量成本”的局部观测合同，见 v2。
>
> 以下内容只保留历史，不得再用于任务构造。

---

## 1. 统一 world 语法

### 1.1 有限二元 CPT world

每个 world 是有限 DAG，节点为二元变量，CPT 为精确有理数。V1 无 latent、无 confounding、无 counterfactual。

### 1.2 有符号二元边

记号 `Edge(U -> V; q)`，其中 \(q\in[-1,1]\setminus\{0\}\) 是方向效应：

\[
P(U=1)=\frac12,
\qquad
P(V=1\mid U=0)=\frac12-\frac q2,
\qquad
P(V=1\mid U=1)=\frac12+\frac q2.
\]

因此

\[
P(V=1\mid do(U=1))-P(V=1\mid do(U=0))=q.
\]

未被边连接或未被干预的二元变量均为独立 \(\mathrm{Bernoulli}(1/2)\)。

### 1.3 动作与观测

- 动作：`INTERVENE(variable, value, batch_size)`，hard-do 语义。
- 观测：TQ 与 AD 返回该 batch 的完整 joint counts，顺序由 symbol orbit 决定；CD 返回 outcome \(Y\) 的 0/1 计数（RCT 的边际结果观测），非 outcome 变量不进入 batch result。
- 成本：原子干预样本数；预算不进入难度标签。
- 每个 seed 都包含至少一个合法但无信息的 distractor 动作，用于检验模型是否会拒绝无用干预。

---

## 2. A / P / N 的可计算定义

设候选世界集为 \(\mathcal W\)，公开先验 \(\pi\) 为 uniform，初始证据 \(D_0\) 为空。
本 seed suite 采用**公开候选世界集 + uniform prior，只隐藏 true_world_index**；这与候选 seed 合同一致。若未来改为隐藏 prior，A/P/N 必须重新计算。

### 2.1 A：初始 query-relevant 歧义

把候选世界按**终局答案对象**合并：

\[
W\sim W' \iff \mathrm{answer}(W)=\mathrm{answer}(W').
\]

\[
A = -\sum_c \pi(c)\log_2\pi(c)\quad\text{bits}.
\]

### 2.2 P：自适应规划深度

P 不依赖未冻结的 accuracy-cost 标量化，而用 risk-cost envelope 定义：

- 令 \(\mathcal R^*_{\rm adaptive}(c)\) 为期望成本 \(c\) 下所有自适应策略的最低终端风险；
- 令 \(\mathcal R^*_{\rm fixed}(c)\) 为同一成本下、干预序列不依赖观测历史的策略的最低风险。
- \(P1\)：对所有可达成本 \(c\)，\(\mathcal R^*_{\rm fixed}(c)=\mathcal R^*_{\rm adaptive}(c)\)。
- \(P2\)：存在成本 \(c\)，使 \(\mathcal R^*_{\rm fixed}(c)>\mathcal R^*_{\rm adaptive}(c)\)，且最优自适应策略树深度为 2。
- \(P3\)：存在成本 \(c\)，使 fixed < adaptive，且最优自适应策略树深度至少 3（留待观测模式冻结后启用）。

### 2.3 N：统计分辨度

对最优风险-cost envelope 上的策略树，令每对答案类 \((W,W')\) 的
**首个分离节点**为它们首次产生不同观测律的决策节点。定义

\[
d_{W,W'}=d_{\rm TV}\bigl(P(\mathrm{obs}\mid do(a),W),P(\mathrm{obs}\mid do(a),W')\bigr)
\]

在该节点上计算。于是

\[
N=\min_{W,W':\mathrm{answer}(W)\ne\mathrm{answer}(W')} d_{W,W'}.
\]

\(N\) 是每样本分辨能力；oracle 再把 \(N\) 转成达到目标风险的 \(n_{\min}(\delta)\)。
这不是预算，也不是效应大小本身。

---

## 3. Task family 1：Target causal query（TQ）

终局输出：两个方向效应

\[
\hat\tau=(\hat\tau_{X\to Y},\hat\tau_{Y\to X}).
\]

评分：连续 MSE 向量误差 + 负原子样本成本。

### 3.1 保留现有 QN 作为 N 轴参考

候选世界两个：

- `FWD`: `Edge(X -> Y; q)`，Z isolated。
- `REV`: `Edge(Y -> X; q)`，Z isolated。

| Seed | q | do(X=1) 下 Y 的均值 | N（TV） | A | P |
|---|---:|---:|---:|---:|---:|
| TQ-N-EASY | 4/5 | 9/10 vs 1/2 | 2/5 | 1 bit | P1 |
| TQ-N-MEDIUM | 2/5 | 7/10 vs 1/2 | 1/5 | 1 bit | P1 |
| TQ-N-HARD | 1/10 | 11/20 vs 1/2 | 1/20 | 1 bit | P1 |

这些 seed 不退化，因为 hidden truth 仍有两个方向；但它们只允许变化 N。

### 3.2 A 轴：公开方向，隐藏符号效应

公共 schema：`X -> Y` 方向已知，Z isolated；效应大小与符号隐藏。
终局仍需返回两个坐标，但 \(\tau_{Y\to X}=0\) 是公开语义。

**TQ-A1-P1-N-MEDIUM**

| World | q | answer \((\tau_{X\to Y},\tau_{Y\to X})\) |
|---|---|---|
| W+ | +1/4 | (+1/4, 0) |
| W- | -1/4 | (-1/4, 0) |

- A = 1 bit
- 最优策略：重复 `do(X=1)`，到达置信门槛后 STOP；P1
- `do(Y=*)` 是无信息 distractor
- N：`do(X=1)` 下两世界 Y 均值分别为 5/8 和 3/8，TV = 1/4

**TQ-A2-P1-N-MEDIUM**

| World | q | answer |
|---|---|---|
| W1 | +3/4 | (+3/4, 0) |
| W2 | +1/4 | (+1/4, 0) |
| W3 | -1/4 | (-1/4, 0) |
| W4 | -3/4 | (-3/4, 0) |

- A = 2 bits
- `do(X=1)` 下 Y 均值为 7/8, 5/8, 3/8, 1/8，四类全分离；P1
- N = 相邻最小 TV = 1/4，与 A1 匹配

配对结论：`TQ-A1-P1` → `TQ-A2-P1` 只改变 A，P/N 不变。

**N 轴配对**：`TQ-A1-P1` 同样给出：

| Seed | q 对 | N |
|---|---|---:|
| TQ-A1-P1-N-EASY | ±1/2 | 1/2 |
| TQ-A1-P1-N-MEDIUM | ±1/4 | 1/4 |
| TQ-A1-P1-N-HARD | ±1/20 | 1/20 |

### 3.3 P 轴：方向和符号都隐藏

**TQ-A2-P2-N-MEDIUM**

| World | 机制 | answer |
|---|---|---|
| W1 | Edge(X -> Y; +1/2) | (+1/2, 0) |
| W2 | Edge(X -> Y; -1/2) | (-1/2, 0) |
| W3 | Edge(Y -> X; +1/2) | (0, +1/2) |
| W4 | Edge(Y -> X; -1/2) | (0, -1/2) |

Z 在所有世界中 isolated。

- A = 2 bits
- 最优策略树：
  1. `do(X=1)`。Y 均值 3/4 → 指向 W1；1/4 → W2；1/2 → {W3,W4}。
  2. 若后验支持 {W3,W4}，改做 `do(Y=1)`。X 均值 3/4 → W3；1/4 → W4。
  3. STOP，返回后验均值效应向量。
- P = P2；第一轮观测真正改变第二轮目标。
- N：瓶颈在 `do(X=1)` 分离 W1/W2 与 {W3,W4}，TV = 1/4。

配对结论：`TQ-A2-P1-N-MEDIUM` → `TQ-A2-P2-N-MEDIUM` 只改变 P，A/N 不变。

---

## 4. Task family 2：Active causal discovery（AD）

终局输出：**焦点变量上的有向边集合**。V1 用 SHD 或 edge F1，不允许把候选等价类擅自压成唯一 DAG。
（若输出 equivalence class，则候选世界必须在 class 上分歧，且按 class 判分。）

### 4.1 AD-A1-P1-N 轴

变量 X,Y。候选世界：

- `FWD`: `Edge(X -> Y; q)`，q 公开为正但方向隐藏。
- `REV`: `Edge(Y -> X; q)`。

答案：`{X->Y}` 或 `{Y->X}`。

| Seed | q | 首动作 | 分离 | A | P | N |
|---|---:|---|---:|---:|---:|---:|
| AD-A1-P1-N-EASY | 1/2 | do(X=1) 或 do(Y=1) | 3/4 vs 1/2 | 1 bit | P1 | 1/4 |
| AD-A1-P1-N-MEDIUM | 1/4 | 同上 | 5/8 vs 1/2 | 1 bit | P1 | 1/8 |
| AD-A1-P1-N-HARD | 1/20 | 同上 | 21/40 vs 1/2 | 1 bit | P1 | 1/40 |

### 4.2 AD-A2-P2-N-MEDIUM

变量 X,Y,Z。公共 schema：三者之间恰好存在**一条正的有向边**，q=+1/2；其余节点 isolated。
六个候选世界：

\[
X\to Y,\quad Y\to X,\quad X\to Z,\quad Z\to X,\quad Y\to Z,\quad Z\to Y.
\]

- A = \(\log_2 6 \approx 2.585\) bits。
- 最优策略树：
  1. `do(X=1)`，观察 joint(Y,Z)。
     - Y 均值 3/4 且 Y⊥Z → `X->Y`，STOP。
     - Z 均值 3/4 且 Y⊥Z → `X->Z`，STOP。
     - Y,Z 都像 Bernoulli(1/2) 且独立 → 真边在 `{Y->X, Z->X}`。
     - Y 与 Z 相关但均值为 1/2 → 真边在 `{Y->Z, Z->Y}`。
  2. `do(Y=1)`：
     - 第一类歧义：X 均值 3/4 且 X⊥Z → `Y->X`；X 均值为 1/2 且 X 与 Z 相关 → `Z->X`。
     - 第二类歧义：Z 均值 3/4 → `Y->Z`；Z 均值 1/2 → `Z->Y`。
- P = P2。
- N：每处决策的最小 TV = 1/4。

这不是 CLadder 式的静态图题：模型必须主动做两次不同干预，且第二次目标由第一次观测决定。

---

## 5. Task family 3：Causal decision / RCT construction（CD）

终局输出：从公开干预集合中选择一个终局动作

\[
a^*\in\{\text{do}(T=0),\text{do}(T=1),\text{do}(X=0),\text{do}(X=1)\}.
\]

效用为 \(u(a;W)=P_W(Y=1\mid do(a))\)；评分为 regret：

\[
\operatorname{regret}(a,W)=\max_{a'}u(a';W)-u(a;W).
\]

### 5.1 CD-A1-P1-N 轴：两臂 RCT 的因果机制识别

变量 T, X, Y。X 是合法但无信息的 distractor。

候选世界：

- `W+`: `Edge(T -> Y; q)`，X isolated。
- `W-`: `Edge(T -> Y; -q)`，X isolated。

终局只允许选择 `do(T=0)` 或 `do(T=1)`。

| Seed | q | do(T=1) 下 Y 均值 | A | P | N |
|---|---:|---:|---:|---:|---:|
| CD-A1-P1-N-EASY | 1/2 | 3/4 或 1/4 | 1 bit | P1 | 1/2 |
| CD-A1-P1-N-MEDIUM | 1/4 | 5/8 或 3/8 | 1 bit | P1 | 1/4 |
| CD-A1-P1-N-HARD | 1/20 | 21/40 或 19/40 | 1 bit | P1 | 1/20 |

### 5.2 CD-A2-P2-N-MEDIUM：先筛选有效 treatment，再切换到有效变量

| World | 机制 | 最优终局动作 |
|---|---|---|
| W1 | Edge(T -> Y; +1/2)，X isolated | do(T=1) |
| W2 | Edge(T -> Y; -1/2)，X isolated | do(T=0) |
| W3 | Edge(X -> Y; +1/2)，T isolated | do(X=1) |
| W4 | Edge(X -> Y; -1/2)，T isolated | do(X=0) |

- A = 2 bits。
- CD 的 batch result 只含 outcome \(Y\) 的计数；因此 `do(T=1)` 下 W3 与 W4 都只显示 \(Y\sim\mathrm{Bernoulli}(1/2)\)，必须进入第二轮。
- 最优策略树：
  1. `do(T=1)`。
     - Y 均值 3/4 → W1，STOP `do(T=1)`。
     - Y 均值 1/4 → W2，STOP `do(T=0)`。
     - Y 均值 1/2 → T 无效，进入第 2 轮。
  2. `do(X=1)`。
     - Y 均值 3/4 → W3，STOP `do(X=1)`。
     - Y 均值 1/4 → W4，STOP `do(X=0)`。
- P = P2。
- N = 1/4。
- 这是 general CPT-World 上的 RCT/bandit 构建：模型不是读一篇临床摘要，而是在隐藏因果机制下决定测量谁、如何分配样本、何时停止并选哪个处理。

---

## 6. Seed 汇总与单因素配对

| Seed | Head | Worlds | A | P | N | 首动作桶 | 策略树 |
|---|---:|---:|---:|---:|---:|---|---|
| TQ-N-EASY/MEDIUM/HARD | query | 2 | 1 bit | P1 | 2/5, 1/5, 1/20 | do(X=1) 或 do(Y=1) | 无分支 |
| TQ-A1-P1-N-EASY/MEDIUM/HARD | query | 2 | 1 bit | P1 | 1/2, 1/4, 1/20 | do(X=1) | 无分支 |
| TQ-A2-P1-N-MEDIUM | query | 4 | 2 bits | P1 | 1/4 | do(X=1) | 无分支 |
| TQ-A2-P2-N-MEDIUM | query | 4 | 2 bits | P2 | 1/4 | do(X=1) | 一次分支 |
| AD-A1-P1-N-* | discovery | 2 | 1 bit | P1 | 1/4, 1/8, 1/40 | do(X=1) 或 do(Y=1) | 无分支 |
| AD-A2-P2-N-MEDIUM | discovery | 6 | 2.585 bits | P2 | 1/4 | do(X=1)/do(Y=1)/do(Z=1) 对称 | 两次不同干预 |
| CD-A1-P1-N-* | decision | 2 | 1 bit | P1 | 1/2, 1/4, 1/20 | do(T=1) | 无分支 |
| CD-A2-P2-N-MEDIUM | decision | 4 | 2 bits | P2 | 1/4 | do(T=1) | 一次分支 |

可验证的单因素配对：

- A 轴：`TQ-A1-P1-N-MEDIUM` vs `TQ-A2-P1-N-MEDIUM`，P/N 固定，A 从 1→2 bits。
- P 轴：`TQ-A2-P1-N-MEDIUM` vs `TQ-A2-P2-N-MEDIUM`，A/N 固定，P 从 P1→P2。
- N 轴：每个 family 内 `N-EASY / MEDIUM / HARD`，A/P 固定。

---

## 7. 防退化门（不接受即 REJECT）

每个 seed 必须通过：

1. **非平凡答案**：至少 2 个不同终局答案类；A > 0。
2. **非零设计价值**：至少一个干预区分至少一对答案类。
3. **不是单动作套壳**：P2 seed 必须由 oracle 证明存在两条不同最优动作序列。
4. **有 distractor**：至少一个合法干预对终局答案的 continuation value 恒为 0。
5. **跨 seed 不坍缩**：同一难度 cell 内，任务实例的图同构类、答案桶、最优首动作桶、策略树类不得全部相同。
6. **难度标签禁用项**：节点数、候选世界数、预算、round 数不得直接标成 A/P/N。
7. **隐藏性**：prompt 不泄漏 true world、oracle action、q 或难度标签。
8. **symbol orbit**：所有 semantic role、target order、effect order 与 label family 全交叉；重命名不改变 task verdict。
9. **oracle 证书**：\(V_{\rm random}<V_{\rm greedy}\le V_{\rm adaptive}\le V^*\)，且数值由同一 hard-do owner 重算。

若某 seed 的 oracle 证明 P1/P2 分类错误，该 seed 返回 `CANDIDATE` 并修正分类，不允许继续标为难度证据。

---

## 8. 当前缺口与下一步实现

本设计不引入第二个 world owner。要落地需要：

1. `world_space.py` 增加通用 `WorldSpec`：任意有限 DAG + 有理 CPT + 通用 truncated factorization。
2. 新增 `oracle.py`：belief posterior、动作价值、\(V^*\)、\(d_{\min}\)、P 分类、N 瓶颈。
3. `rendering.py` / `task_scoring.py` 增加三种 terminal head 的渲染/解析：
   - query：效应向量；
   - discovery：有向边集或 equivalence class；
   - decision：终局干预动作。
4. `seeds.py` 从证书文件加载本文档 seed，不硬编码。
5. 在 oracle 复算前，所有新 seed 保持 `CANDIDATE`；复算通过后升级 `VERIFIED`，再允许进入采样分布。

下一允许实现动作仍是：**先实现通用 hard-do owner，再实现 oracle，然后复算本表中的每个 seed。**
