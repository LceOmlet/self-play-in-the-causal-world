# 2026-09-07：内核与官方 DAPO 的可复现证据

本目录保存持续 goal 的研究代码与轻量结果。生产实现位于 `src/cpt_world`，
正式训练入口为 `scripts/run_official_dapo.sh`。这里的解析参考、枚举、张量观察器
均不接管训练算法，不提供另一套 rollout、loss 或 optimizer。

## 当前状态

| 工作 | 证据及范围 |
| --- | --- |
| 反事实精确化简 | 二态中介、或三态中介二元事件商空间的适用条件下已合入。74 道冻结题仍有 51 道认证成功、23 道未认证；没有把未解题宣称为已解决。适用 4 题总计 8.28→1.60 秒。参见 `response-envelope-proof.md` 和 `results/diagonal-transport-acceptance-summary.json`。通用 envelope 候选导致一题退化，未合入。 |
| 根 CPT 强反转切片 | 72 个世界、144 个切片、4,032 次独立枚举核对。只证明可行参数区间和应保留的条件密度，不宣称已有无偏且更快的生产采样器。 |
| 二态 ET-V2 密度 | 150 次实际内核核对与二维、三维符号 Jacobian 检验；更高维多态机制仍有推导义务。 |
| 最优合法实验 | 特定弱边双世界在预算 131,072 下，最佳可达等先验错误率约 48.9788%。这是最坏情形反例，不是困难题占比。 |
| 官方 DAPO run-01 | **失败**：recipe 覆盖 action mask，导致工具 token 进入训练。局部公式检查的临时通过结论已撤销，相关检查点禁止初始化。 |
| 官方 DAPO run-02 | 已从原始基座完成一次隔离更新并通过实际张量验收，见 `official_execution/run-02-mask-v1-audit-only/execution-acceptance.json`。19,830 个动作 token、4,018 个工具 token 的掩码传播已逐段核对。一次更新不能证明收敛或长期能力改善。 |

完整活动清单和剩余义务见 [goal](../../docs/rl-correctness-goal-20260907.md)。

## 复现数学检查

需要项目依赖、SymPy、mpmath；不需要模型或 GPU。从仓库根目录运行：

```bash
python research/rl_correctness_20260907/finite_budget_witness.py
python research/rl_correctness_20260907/verify_optimal_legal_experiment.py
python research/rl_correctness_20260907/verify_root_probability_slice.py
python research/rl_correctness_20260907/verify_et_v2_binary_density.py
```

脚本会在自身目录写入新的 JSON，冻结结果位于 `results/`。前三个切片/密度/最优实验
脚本仅调整了项目与 witness 的定位，使其随仓库移动；实际核对公式不变。
`finite_budget_witness.py` 保存旧弱边反例，最优实验脚本将其通用下界加强为可达最优风险。
`verify_official_token_tis.py` 直接调用固定上游 Token-TIS、GRPO 与 DAPO loss，
需要官方环境；它检验解析 loss/梯度及 microbatch 分割，不能替代真实模型验收。

## 真实执行审计

`official_execution/observer` 使用 Python 调用观察机制，只记录实际官方所有者用到的
事件和张量，不替换函数、不采样随机数、不修改决策。观察器有复制与同步开销，
因此这些运行不作为吞吐基准。

`official_execution/analyze_execution.py` 读取 run-02 的实际张量，追踪
`AgentLoopOutput → pre_filter_batch → actor_batch → loss → Adam`。
它需要服务器原始 `.pt`；只读 JSON 结果不能重跑张量验收。
`diagnose_mask_overwrite.py` 保存 run-01 的失败证明；其中临时局部通过文件已标记
`passed=false`，不得选择性引用。`plot_verified_update.py` 复用旧绘图脚本的样式，
需要 matplotlib、NumPy 和中文字体，并且仅接受通过验收的 run-02 结果。

`control.py`、`control-run-01.py` 保留原运行的服务器路径、数据与一次更新配置。
它们是可审查的原始运行记录，迁移服务器时必须先改实际路径、准备数据并完成 preflight；
不得把历史 run-01 控制器当作当前训练入口。`archive_original/` 逐字保留本轮推导、
候选、安装及验证脚本；其中包含未合入候选和针对原服务器的迁移程序，不应批量执行。

## 数据与备份边界

原始完整证据目录：

```text
/home/chen/runs/rl-correctness-goal-20260907
/home/chen/runs/kernel-integration-20260907
/home/chen/runs/kernel-root-cause-20260907
```

仓库保存代码、证明、来源清单和轻量验收结果；模型权重、检查点、冻结任务 pickle、
完整轨迹和大型模型/梯度张量仍在审计服务器，不随 Git 上传。数据与张量重放依赖
这些外部证据，Git 备份本身不能保证它们永不丢失。任务数据是已检查的诊断集，
不是未接触的能力评测集。

`archive-manifest.json` 对归档文件逐字记录 SHA-256；研究目录和上游补丁的 Git
行尾转换已关闭，避免克隆破坏来源哈希。新增结果后需刷新清单。
