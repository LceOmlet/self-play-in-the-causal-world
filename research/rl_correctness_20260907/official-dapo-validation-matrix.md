官方 DAPO 训练验证：证据与缺口

本文件归属于持续 Goal B/E。保留官方训练算法，补实际执行路径证据；不使用历史 TRL 审计脚本，不把旧单步更新或旧检查点当作修复后验收。

| 路径 | 现有证据 | 需要补齐的验收 |
|---|---|---|
| 上游来源与调用入口 | 固定 verl/verl-recipe commit；427 个文件指纹；实际导入路径；两个工具终止补丁独立清单 | 环境修复合入后重新记录源指纹、数据指纹及实际 Hydra 配置 |
| 环境质量→奖励 | `verl_environment.py` 从真实 episode 取质量；工具步奖励为 0；官方 DAPORewardManager 拥有 overlong shaping；测试验证未完成题质量为 0、跨题状态拒绝 | 修复后的真实轨迹中，末次动作、一次终止质量和 shaping 逐条一致；区分 quality 与 shaped reward |
| 终止→token 序列 | 工具终止补丁与生命周期回归已通过；保留终止动作 token，不再追加终止反馈与后续生成 | 在真实多轮轨迹中检查 action token、tool token、终止标记、response_mask、attention_mask 和截断位置 |
| 动态过滤→优势 | 使用官方 metric=acc 动态组过滤、GRPO 标准化优势；上游相关测试及解析 loss/梯度核对已做 | 捕获被实际保留的 UID 组、acc、shaped reward、优势与 mask；确认同一组的世界和 tape 一致，记录丢弃组的生成/环境成本 |
| rollout→actor 概率 | 短固定 token 比较与 RoPE 缓冲类型例子已有证据；不能解释全部真实长轨迹差异 | 对齐实际生成 token 和 mask；核对温度、权重版本、rollout 概率的语义，以及官方 Token-TIS 的截断比例、权重尾部、有效样本量；不要求不同数值后端逐位相等 |
| token loss→梯度 | 官方不对称裁剪、token-mean 聚合及不同 microbatch 划分的解析导数核对已做；官方 torch chunked 输出梯度通过其精度容差 | 修复后实际 FSDP/LoRA 路径上的 loss、重要性权重、有效 token 分母、累积尺度、梯度有限性和一次优化步计数 |
| 梯度→新参数 | 旧隔离运行确有 496 个 LoRA 张量、248 个非零 B 张量、一次更新，进程正常退出 | 该运行早于终止/Token-TIS 修复，不能用于当前验收；从原始基座全新初始化适配器，核对修复后参数变化及冻结基座约束 |
| 奖励/任务误差→可视化 | 已有训练和环境可视化/分析产物可复用 | 将原始质量、shaping、实际任务误差与执行成本分开记录；一更新的执行验证不宣称长期能力提升 |

标准 DAPO 的代理目标与环境原始质量不完全相同：组内标准化不保留绝对奖励差值尺度；动态过滤改变被更新样本的分布；token-mean 聚合依赖有效动作长度；官方 Token-TIS 的逐 token 截断不是无偏整轨迹校正。应明确这些已选算法定义及适用范围，不能通过私自替换算法让说明看起来一致。

前置条件：待验收任务的标签与评分来自已核对内核，当前环境版本及数据来源明确；官方来源、终止协议、mask 与配置测试通过；诊断输出独立于训练决策。随后允许在隔离目录运行恰好一次官方更新，旧检查点初始化与恢复继续禁用，生产长训练保持停止。

优先使用上游现成日志和保存的实际数据。若必须增加观测，只记录真实所有者使用的张量或事件，不能替换 rollout、reward、advantage、loss 或 optimizer。既有 `scripts/audit_rl_execution.py` 和 `scripts/verify_dapo_model_gradients.py` 属于历史 TRL 路径，不用于这个验收。

本轮已补验官方 Token-TIS＋DAPO 的组合函数：交错 UID 分组、带孔动作 mask、长度不同的 4 条序列、非单位截断重要性权重、正负优势、不对称裁剪与配置中的负优势 dual clip。官方权重与解析权重一致，权重 detached，关闭 RS 时 mask 不变；1、2、4 行 microbatch 划分的 token-mean 损失均约 1.711663794947563，最大梯度误差 `1.11e-16`。该检查直接调用固定上游函数，无模型或优化器，不能替代后续真实 FSDP/LoRA 更新验收。证据为 `official-token-tis-acceptance.json`。

真实运行修正了此前的验收盲点：run-01 中虽然每个公式与其输入一致，recipe 却在 `compute_kl_related_metrics` 入口把原始 response_mask 覆盖为 attention mask。验收现在要求用实际 response token 序列定位每一行，逐项核对 `AgentLoopOutput → pre_filter_batch → actor_batch → microbatch loss` 的掩码传播；不能只分别核对两端的局部公式。run-01 的局部通过文件已撤销，实际结论为失败，其检查点禁止继续初始化。11,437 个动作 token 中混入了 2,976 个工具 token；33.61 的最大 log-prob 差异来自工具 token，真实动作最大差异为 0.558。

已新增 `verl-recipe-action-mask-v1.patch`，采用固定官方 RayPPOTrainer 的既有 guard 保留动作掩码。补丁明确修改 recipe 的数据接口，因此不能再把 recipe 整个文件称为逐字未修改；算法公式和官方训练入口保持原样。来源校验保留原哈希，同时将这个例外绑定到唯一的补丁 SHA256。run-02 从原始基座重新验证该修改，不使用 run-01 检查点；其结果尚待实际更新完成。


run-02 已完成并通过实际张量检查，详见 ../../docs/dapo-execution-audit-20260907.md。该更新的实际掩码、质量、优势、重要性权重、loss 和一次 Adam 参数更新均已核对；非零 overlong shaping、动态丢弃和 TIS 上截断没有在这一次运行中发生，其证据仍来自标准函数测试或历史实际执行。长期训练与任务分布校准尚未完成。
