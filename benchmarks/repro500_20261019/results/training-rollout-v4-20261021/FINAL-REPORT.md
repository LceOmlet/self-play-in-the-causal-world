# 两组 9B 正式评测：200 条完整结果

原版与 checkpoint2350 各运行同一组 100 题（五类各 20 题），共 200 条独立 rollout；单 GPU 串行执行，一题一次采样。冻结 on-demand-data-v1.1、提示词、工具、预算与评分；不混入旧 8K 结果、已停止的 v3 部分或原生 agent 结果。

## 结果

指标为 terminal-quality-v10 reward（0–1），不是正确率。

| 类别 | 原版平均 reward | checkpoint 平均 reward | 差值 | 提交数：原版 / checkpoint |
|---|---:|---:|---:|---:|
| ate | 0.1860 | 0.2073 | +0.0213 | 9 / 10 |
| backadj_minimal_sets | 0.5821 | 0.7036 | +0.1214 | 13 / 17 |
| best_intervention | 0.4843 | 0.6384 | +0.1541 | 14 / 17 |
| individual_counterfactual_probability | 0.2806 | 0.1571 | -0.1235 | 16 / 9 |
| mediator_set | 0.0715 | 0.1217 | +0.0502 | 10 / 14 |
| Overall | 0.3209 | 0.3656 | +0.0447 | 62 / 67 |

checkpoint 胜 / 平 / 原版胜：30 / 35 / 35。平均 reward 差值的配对 bootstrap 95% 区间为 [-0.0297, 0.1195]，跨过 0；本次样本不能确认稳定提升。四类均值上升，individual_counterfactual_probability 下降。逐题原始结果见 paired-results.json，直接差值与胜负见 per-case-differences.json；全部 100 对均有效终止，没有把基础设施故障作为零分。

个体反事实概率类的提交数从 16 降到 9，与总体 reward 下降同时出现；已提交子集的误差不能代表全部题。其他分类的条件误差、regret、编辑距离和 F1 保留在 comparison.json。终止原因与观测使用汇总见 termination-analysis.json。

## 实际耗时与开销

模型墙钟按 runner-source.json 写入至最终 progress.json 写入的文件时间测量，含该阶段模型加载与执行。事件时间来自 events.jsonl；生成和工具的区间并集已去除并发重叠，不能将每请求时长直接相加当作墙钟。

| 模型 | 墙钟小时 | 生成请求 | 工具调用 | 保留模型 token | 工具反馈 token | 生成区间并集秒 | 工具区间并集秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 6.267 | 4317 | 4215 | 1665752 | 724856 | 22108.54 | 279.79 |
| checkpoint2350 | 6.424 | 4217 | 4138 | 1694508 | 720237 | 22577.70 | 305.73 |

两模型串行总墙钟 12.695 小时（不含最后结果编译、打包和传输）；保留模型 token / 端到端秒分别为 73.83 与 73.27。这与固定 512 token、预热混批基准的 190–191 token/s 是不同统计口径。本地逐条计数的原始生成响应模型 token 为原版 1,665,752、checkpoint 1,694,508，与各自保留的 rollout 模型 token 总数完全一致；完整响应、工具反馈和输出均已保存，详见 local-verification.json。

## 方法与核验

复用原训练 TRL 生成、分词、解析、工具循环方法；32768 上下文、31497 累计 completion 预算（含工具反馈），无额外 8K 单轮上限；temperature=1、top_p=1、top_k=0、repetition_penalty=1、thinking=false。生成方法一致，当前冻结任务接口与训练旧自动反馈接口不同。种子：构造引擎前 seed42，引擎 seed0；MP0 会重设同进程 driver RNG，OutcomeTape 独立于全局 RNG。

正式执行采用通过官方分层检查的固定调度 decode Graph 配置（MP0，CompilationMode.NONE，FULL_DECODE_ONLY）。对应官方断言的适用场景与限制见 ../vllm-official-analysis-20261021/overview.json，不将它解释为任意未来采样轨迹保证。

配对提示词、工具、capsule、预算、两模型源码指纹及 31497 请求上限均验证通过；runner SHA256：4146f53f964b8f9aede8954faf5aebb59a7fa8084a547f8394ee96c112517073。

原始归档共 823 个文件，224,928,447 字节。远端保留完整 raw-rollouts.tar.gz；Git 中为三个不超过 80 MiB 的二进制分片，按 archive-restore-index.json 顺序拼接。整流 SHA256：4b0e2f00e8659b2abaa13382fffba17c148a14615e5b7a61b793ae12c20ccddc。逐成员 SHA、远端及本地复核结果分别见 archive-members-sha256.json、archive-verification.json、local-verification.json；原始编译文件校验见 all-result-checksums.json，下载导出文件校验见 export-checksums.json，最终交付目录全部文件校验见 delivered-checksums.json。
