# 可直接复用的真实世界 / 任务 / 查询定义来源（CANDIDATE）

> 目的：不自己编世界和图族。只登记能从互联网拿到代码、数据、公开 DAG/CPT 的 source。
> 筛选标准：
> 1. 有限离散 DAG + 可写成有理数的 CPT；
> 2. 机制可隐藏，干预为 hard-do；
> 3. 任务/查询定义已有明确 truth 或可精确重算；
> 4. 对每个 treatment 锚点，不得出现 \(P(Y|do(X))=P(Y|X)\) 的原始 QN 退化。

---

## 1. CLadder：世界、任务、查询都最接近，优先复用

仓库已 clone：`/tmp/cladder`（上游 `https://github.com/causalNLP/cladder`）。

### 1.1 世界定义（10 个二元 SCM motif，直接可读代码）

代码文件：`causalbenchmark/graphs/stories/phenomena.py`。观测变量全部二元，
参数由 builder 生成。原样结构：

| graph_id | 代码中的结构 | 备注 |
|---|---|---|
| chain | `X→V2→Y` | 纯 chain |
| fork | `X→Y←V2`（X,V2 为根） | fork |
| collision | `X→V3←Y` | collider |
| confounding | `V1→X, V1→Y, X→Y` | 有共同原因 |
| mediation | `X→V2, X→Y, V2→Y` | 直接+间接 |
| diamond | `X→V2, X→V3, V2→Y, V3→Y` | 双路径 |
| diamondcut | `V1→X, V1→V3, X→Y, V3→Y` | 有 confounder 的双路径 |
| IV | `V1→X, V2→X, V1→Y, X→Y`（V1 unobserved） | 含 unobserved |
| arrowhead | `X→V3, V2→V3, X→Y, V2→Y, V3→Y`（V2 unobserved） | 含 unobserved |
| frontdoor | `V1→X, X→V3, V1→Y, V3→Y`（V1 unobserved） | 含 unobserved |

**直接可用子集**：`chain, fork, collision, confounding, mediation, diamond, diamondcut`。
`IV, arrowhead, frontdoor` 含 unobserved 变量，当前 CPT-World 无 latent owner，暂不导入；
不是丢弃，是等 latent owner 扩展后再用。

### 1.2 查询定义（10 类，代码和正式形式可直接复用）

代码目录：`causalbenchmark/queries/`。

| query_type | rung | formal form | CPT-World 状态 |
|---|---|---|---|
| marginal | 1 | \(P(Y)\) | 可作 observational 诊断 |
| correlation | 1 | \(P(Y|X)\) | 可作 observational 诊断 |
| exp_away | 1 | 条件在 collider 上比较两 parent | 可作观测诊断；Earthquake 标准例 |
| ate | 2 | \(E[Y|do(X=1)]-E[Y|do(X=0)]\) | **直接可作主任务** |
| backadj | 2 | 给定 treatment/outcome 的 backdoor adjustment set | **直接可作主任务** |
| ett | 3 | \(E[Y_{X=1}-Y_{X=0}|X=1]\) | 当前 head 不支持，暂不导入 |
| nde | 3 | controlled direct effect 形式 | 当前 head 不支持，暂不导入 |
| nie | 3 | indirect effect 形式 | 当前 head 不支持，暂不导入 |
| det-counterfactual | 3 | counterfactual | 当前 head 不支持，暂不导入 |

CLadder 的每个数据条目还带 `treatment/outcome/mediators/formal_form/estimand/groundtruth`
（见 `data/cladder-v1-questions.json`，共 10,560 条）。导入 CPT-World 时：
- 用其 graph 和 query 定义，不用其自然语言 story；
- 把其公开 `given_info` 从 prompt 中移除，变成模型必须通过 `do + measure` 自己采出来的数据；
- truth 仍由 CLadder 的 SCM/estimand 重算。

### 1.3 CLadder 记录的 graph-query 兼容表（原文 Appendix A.4）

- NDE 只在 IV、Arrowhead、Confounding、Mediation、DiamondCut 上生成；
- NIE 只在 Mediation、Frontdoor、Arrowhead、Diamond、Chain 上生成；
- Collision / explaining-away 只作为 ATE 的 collider 结构诊断素材，不再单独生成任务；
- ATE 除 Collision 外所有图都生成；
- 反事实除 Collision 外所有图都生成；
- ATT 除 Collision、IV 外所有图都生成。

---

## 2. bnlearn：真实网络世界定义，直接提供 DAG + CPT

来源：`https://www.bnlearn.com/bnrepository/discrete-small.html`

已下载核验：

- `cancer.bif.gz`
- `earthquake.bif.gz`
- `asia.bif.gz`
- `survey.bif.gz`

这些文件直接定义有限离散 DAG 和 CPT，可直接成为 CPT-World 的 hidden world owner。
它们本身不带任务/查询定义；任务应从 CLadder/CauGym 的查询族中选配：

| 网络 | 可选查询 |
|---|---|
| Cancer | ATE：Pollution/Smoker→Cancer/Xray/Dyspnoea；backadj：Smoker→Dyspnoea；best-intervention：Dyspnoea |
| Earthquake | collider / explaining-away 结构作为 ATE 诊断切片；ATE：Burglary/Earthquake→JohnCalls/MaryCalls；best-intervention：MaryCalls |
| Asia | ATE：Smoke/Bronc/Lung→Dysp；backadj；mediator set：Smoke→Dysp；best-intervention：Dysp |
| Survey | ATE：E/O/R→Travel；backadj；best-intervention：Travel=car；多值域直接支持 |

注意：Cancer 的 Pollution/Smoker 是根节点，若作 treatment 有 \(P(Y|do(X))=P(Y|X)\)
的退化风险；作为 seed 时优先把 treatment 放在非根节点（Cancer、Education、Occupation 等），
或用干预目标/决策查询代替纯 ATE。

---

## 3. CauGym：任务/查询定义来源

论文：arXiv:2602.06337。世界是其采样的 10 节点 DAG + SCM（合成，不是直接复用）。
可直接复用其**任务定义**：

```text
ATE, CDE, ETT, NDE, NIE, PN, PS
```

其中当前 CPT-World 能直接用：`ATE, CDE`（需要 multi-target do 时用 CDE）。
`ETT/NDE/NIE/PN/PS` 属 counterfactual，暂不进入当前 head，但登记为未来查询。

---

## 4. CausalPitfalls：真实统计陷阱的任务/查询定义

论文：arXiv:2505.13770。6 类陷阱、15 个挑战、每挑战 5 个难度。
世界多为具体 SCM 场景。可直接复用其**任务族定义**：

- confounding / spurious association；
- intervention vs observational reasoning；
- mediation；
- collider / selection；
- causal discovery；
- effect estimation。

导入时只保留当前 CPT-World 支持的有限离散、无 latent 部分，不使用 story 和 rubric。
它是“一个任务对应一个明确因果陷阱”的模板来源。

---

## 5. CausaLab / SciGym / CausalGame：环境与协议来源，不是离散 CPT 世界

- CausaLab（arXiv:2605.26029，repo 已 clone）：提供 hidden SCM + 预算干预 + 任务分/机制恢复双评分；
  世界是连续 SCM，不能直接作为 CPT-world，但其**任务协议**可直接借鉴。
- SciGym（repo 已 clone）：真实 SBML 模型 + partial model + 实验 action；世界是连续 ODE，
  不能直接作为 CPT-world，但“用真实模型、遮住部分机制、允许实验操作”是 seed 构造范本。
- CausalGame（arXiv:2607.04293，repo 已 clone）：场景 SCM + 实验预算 + 效用；含 latent/selection，
  不能直接进入当前无 latent owner，但其“隐藏世界 + 明确效用 + analytic optimum”可直接用于 decision 任务。

---

## 6. 建议的直接导入顺序

1. 先导入 CLadder 的 7 个观测完整 graph motif + 2 个 query（ate/backadj）；collider 作为 ATE motif 标签。
2. 用 CLadder `data/cladder-v1-questions.json` 的 meta 生成 CPT-World episode：
   隐藏 `given_info`，模型通过 `do + measure` 获取证据。
3. 再用 bnlearn 四个 `.bif` 作为 hidden world，套用同一查询族。
4. 后续扩展 latent owner 后，再导入 CLadder 的 IV/arrowhead/frontdoor 与 CauGym 的 NDE/NIE。
