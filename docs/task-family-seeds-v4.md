# CPT-World 任务族 Seed 设计 v4（HISTORICAL）

> **已被 `docs/task-family-seeds-v5.md` 取代。** v5 放弃了 A/P/N，只按真实需求设计；
> v4 的 anchor-minimal 思路保留，但其 A/P/N 难度格与单因素配对全部不再使用。

---

## 1. 对“当前思路能否做到”的直接回答

**不能。** v3 只解决了一半问题：

- v3 修掉了“环境自动返回 full joint”的坍缩；
- 但 TQ-CB/TQ-S/CD-S 仍然**显式指定了变量角色**（X/Y/S、T/S/Y），等于告诉模型
  “要用哪个变量、测什么类型”。这不符合“指定变量越少越好、探索哪些变量应由模型自己发现”的要求。
- AD-MEC3 / AD-PATH4 是正确方向的雏形，但仍太小、且公开了 path skeleton，
  模型只是在 3-4 个已命名节点上做 orientation，不是在“未知相关变量集合”上做结构发现。

因此 v4 的中心从“query 需要哪些变量”转向：

> **任务只锚定终局对象（0 个或 2 个变量），候选世界的相关变量集合、路径、调节变量、
> distractor 全部不公开；模型必须通过主动干预和显式局部 readout 自己找出并利用它们。
> “需要探索利用的变量数量”由候选世界族的规模与结构复杂度分级，并折算到 A/P/N 上。**

---

## 2. Anchor-minimal 原则

1. **AD（主动发现）**：0 个变量锚。只声明“输出这些 opaque 变量上的 DAG/edge set/equivalence class”。
2. **CD（因果决策）**：2 个变量锚（treatment T、outcome Y）。其余变量都是候选协变量，
   模型自己发现哪些变量必须进入 decision rule。
3. **TQ（目标查询）**：若仍是两个变量的**无条件总效应**，它数学上就不需要探索第三个变量；
   因此 TQ 只能当 parser/tape 诊断，不能作为自主探索主任务。
   若未来要让 TQ 具有自主探索性，唯一诚实做法是把终局改成
   “发现机制 + 数值 query”的结构化答案，这违反当前 D11 的二维数值合同，必须作为新的 head 合同提案，
   不得悄悄塞进 benchmark。

公共题面只允许出现：
- opaque 变量 id 与二元域；
- 终局对象与合法 JSON schema；
- 全局预算与 batch size；
- 动作/readout 的语法。

禁止出现：相关变量子集、path 长度、边数、MEC、调节变量、treatment/outcome 之外的语义角色、q/e 值。

---

## 3. 观测与动作合同（沿用 v3，进一步去角色化）

```json
{
  "type": "intervene",
  "target": "KJM",
  "value": 1,
  "measure": ["NGR", "LWH"],
  "batch_size": 8
}
```

- `target` 是任意 `manipulable=true` 变量；`value ∈ {0,1}`。
- `measure` 是任意可读变量非空子集，`target ∉ measure`，\(|measure|\le k_{\max}\)。
- 环境只返回 `measure` 的 joint counts。
- 成本 = `batch_size × |measure|`。
- 任务可以声明 `k_max`，但**不声明“哪个 measure 有用”**。
- 除终局锚点变量外，所有变量默认 `manipulable=true, readable=true`；
  CD 族的协变量按“基线协变量不可干预”声明为 readonly，但仍不透露其结构角色。

---

## 4. 探索利用变量数的难度分级

冻结的难度轴仍是 \(A,P,N\)。v4 引入一个**生成器参数**：

\[
r = \text{候选世界中非平凡变量的最大数量（需探索/利用的相关变量数）},\quad
d = \text{distractor 数},\quad N = r+d.
\]

\(r\) 不作为第四难度轴；它必须通过候选世界族的构造**转化为 A/P/N**：

- \(r\) 或候选结构数增加 → answer class 数量增加 → **A** 增大；
- 定位相关子集/逐段 orient 的决策树变深 → **P** 增大；
- 每步 TV 仍由机制强度 \(e\) 控制 → **N** 不变。

若团队想把“需探索变量数”变成公开的第一类难度标签，必须新增 FROZEN decision；
在未冻结前，只报告 \(r\) 作为 diversity/generator 诊断，不进入 reward 归一化。

### 难度阶梯（候选）

| 阶梯 | 世界族 | r | d | A（bits） | P（候选） | N |
|---|---|---|---:|---:|---:|---:|---|
| L0 | N=3，恰一条有向边 | 2 | 1 | log2 6 ≈ 2.585 | 搜索+定边，≥P2（oracle 待核） | e |
| L1 | N=4，恰一条有向边 | 2 | 2 | log2 12 ≈ 3.585 | 搜索+定边，≥P2 | e |
| L2 | N=3，三节点 path MEC（3 个定向） | 3 | 0 | log2 3 ≈ 1.585 | P2 | e |
| L3 | N=4，四节点 path MEC（4 个定向） | 4 | 0 | 2 | P3 | e |
| L4 | N=4，恰一条 2-path（两条相邻边） | 3 | 1 | log2 24 ≈ 4.585 | P3（待核） | e |
| L5 | N=5，恰一条 2-path | 3 | 2 | log2 60 ≈ 5.907 | P3（待核） | e |
| L6 | N≤5，DAG 且 \(1\le|E|\le2\)（非孤立候选） | ≤5 | 0 | log2|W| | 待 oracle | e |

同一行内只通过 \(e\in\{2/5,1/5,1/20\}\) 生成 N-EASY/MEDIUM/HARD。
候选世界集与 uniform prior 公开；只隐藏 true world index。这是 exact Bayesian oracle 的必要前提。

---

## 5. 核心族 AD-AUTO：0 锚点的自主结构发现

### 5.1 终局与评分

终局输出所有 N 个 opaque 变量上的有向边集合（允许空边）：

```json
{"type": "answer", "edges": [["KJM", "NGR"]]}
```

评分用 SHD 或 edge F1（与 active discovery head 的 D4 合同一致）。
答案等价类 = 精确 DAG；同构但变量 id 不同的世界在 symbol orbit 下必须同分。
连续化 proper score 暂不冻结。

### 5.2 候选世界构造

- \(N\) 个二元变量；候选 DAG 来自一个公开的有限结构族（如 L0-L6）。
- 每条有向边 \(U\to V\) 的 CPT：
  \(P(V=1|U=0)=1/2-e,\ P(V=1|U=1)=1/2+e\)。
- 根节点 Bern(1/2)；孤立节点 Bern(1/2)。
- 候选世界按“结构族 × 边参数”生成；先只使用同一边强度 \(e\)，保证 N 轴单一。
- 非平凡变量不标号；distractor 也不标号。模型只能从干预数据发现。

### 5.3 为什么必须自主探索

- `k_max=1` 时没有任何动作会返回 full joint；
- 单次 `do(U=1), measure={V}` 只给一对变量的一种条件分布，无法知道 U/V 是否相关变量；
- 定位相关子集至少需要跨变量的主动搜索；
- 定向一个 MEC 需要根据前一阶段结果切换下一阶段 target，天然产生 P2/P3。

### 5.4 已核验的可行性证据

本轮用 exact Fraction 核验了 L0/L1（N=3/4，恰一条有向边），见
`scripts/verify_task_family_seed_math.py`：
对每一对不同世界，都存在某个 `do(U=v), measure={W}` 动作使单样本 TV ≥ e（e=1/4 时 ≥1/4）。
因此不存在“所有动作都无法区分”的死任务。
完整最优策略树与 risk-cost envelope 仍待 `oracle.py`。

---

## 6. CD-AUTO：2 锚点的自主决策（DRAFT）

- 公共锚点只有 `T`（manipulable）与 `Y`（readonly）。
- 其余 \(m\) 个 opaque 协变量 readonly；其中最多 1 个是真实 effect modifier，其余是 distractor。
- 候选世界：`modifier ∈ {none, C1, ..., Cm}` × `sign ∈ {+q, -q}`。
- 终局输出 decision rule：`do(T=1)` 或 `do(T=0)`，允许按模型自行发现的协变量子集分层。
- 若模型声明使用某个协变量，scorer 按公开的 rule 语义评估 regret；模型必须为其 rule 负责。
- 该族满足“只指定 T,Y；哪些协变量需要探索由模型发现”。
- 在 decision head 的 rule 语法与 regret scorer 冻结前保持 DRAFT。

---

## 7. TQ 的诚实定位

无条件双向 target query 的最小充分集是 {X,Y}，因此它**永远不可能**靠增加 N 来要求结构探索。
v3 的条件双向/HTE 虽要求测 S，但 S 是指定的，仍不满足 v4 的 anchor-minimal 原则。
所以：

```text
TQ(ROLE-SPECIFIED CONDITIONAL/HTE): DIAGNOSTIC/INTERFACE-ONLY, 不再作为自主探索主任务
TQ(STRUCTURE-GATED NUMERICAL QUERY): FUTURE HEAD CONTRACT PROPOSAL, NOT FROZEN
```

---

## 8. 防坍缩与准入门（v4 增量）

在 v3 的 10 个 gate 之外增加：

11. **Anchor-minimality gate**：删除任何非锚变量的语义提示后，oracle value 与合法答案不变；否则 REJECT。
12. **Relevant-subset discovery gate**：oracle 必须证明最优策略的前缀包含“从 N 个变量中定位相关子集”的
    动作；若答案类与相关子集一一对应且题面泄露该子集，REJECT。
13. **Exploration-span gate**：报告 \(r,d,N\)，并证明在固定 \(A,P,N\) 的 cell 内
    候选世界具有不同的相关子集/最优首动作/策略树，防止表面多样性坍缩。
14. **Distractor 非冒充难度 gate**：distractor 只能扩大设计空间与 A，不得伪装成 P 或 N。
15. **Full-table 非支配**：在 AD-AUTO 中 universal “枚举所有 \(do(U=v),measure={W}\)” 策略
    必须在低/中成本 envelope 上被自适应搜索策略严格优于；否则该难度档 REJECT。

---

## 9. 实现顺序

1. `oracle.py` 先只实现 AD-AUTO-L0/L1（N=3/4，恰一条边）的 exact belief MDP；
   输出 posterior、Q-value、最优策略树、risk-cost envelope、A/P/N 证书。
2. 用 L0/L1 跑基线阶梯：random < pairwise-search < greedy < adaptive ≤ oracle。
3. 再实现 L2/L3（MEC orientation）与 L4/L5（2-path）。
4. `world_space.py`/`world_runtime.py` 支持通用 N 元 DAG + local readout，继续使用 action-keyed tape。
5. `seeds.py` 只加载通过全部 gate 的 seed；v3 的角色指定族降级为诊断族。
6. CD-AUTO 等 decision rule 合同冻结后再进入 seed。

---

## 10. 当前裁决

```text
V3 ROLE-SPECIFIED FAMILIES: DIAGNOSTIC_ONLY, NOT_MAIN
V4 AD-AUTO L0/L1: CANDIDATE (分布可行性已核验, oracle 未闭合)
V4 AD-AUTO L2-L6: CANDIDATE/DRAFT
V4 CD-AUTO: DRAFT_BY_HEAD_CONTRACT
TQ UNCONDITIONAL: DIAGNOSTIC_ONLY
VERIFIED-SEED: BLOCKED_BY_OWNER_AND_ORACLE
```

在 `oracle.py` 对 L0/L1 给出 envelope 前，不得宣称 v4 已满足“自主充分采样与探索”。
