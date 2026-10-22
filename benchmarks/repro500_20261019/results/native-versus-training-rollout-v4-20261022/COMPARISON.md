# 逐题子 agent 与本次两组 9B 的比较

复用已完成的 100 个逐题独立 native 子 agent 结果，与本次原版 / checkpoint2350 各 100 条 v4 轨迹配对。没有重新求解或重采样；不使用旧 8K 模型分数，也不使用早期保留 20 题历史的 agent 结果。

**指标为 reward，不是正确率。任务、工具、capsule、观测预算和逐题隔离已匹配；native 隐藏推理 / 生成算力、完整系统上下文及额外传输限制未与 9B 严格匹配。以下说明本次任务表现，不是严格等算力优劣结论。**

| 类别（每类 20 题） | 逐题子 agent | 原版 9B | checkpoint2350 |
|---|---:|---:|---:|
| ate | 0.6291 | 0.1860 | 0.2073 |
| backadj_minimal_sets | 0.9500 | 0.5821 | 0.7036 |
| best_intervention | 0.8757 | 0.4843 | 0.6384 |
| individual_counterfactual_probability | 0.6774 | 0.2806 | 0.1571 |
| mediator_set | 0.3938 | 0.0715 | 0.1217 |
| Overall | 0.7052 | 0.3209 | 0.3656 |

| 配对比较 | 平均 reward 差 | 95% bootstrap 区间 | 子 agent 胜 / 平 / 模型胜 |
|---|---:|---|---:|
| 子 agent − base | +0.3843 | [+0.3119, +0.4563] | 78 / 17 / 5 |
| 子 agent − checkpoint2350 | +0.3396 | [+0.2632, +0.4190] | 67 / 21 / 12 |

| 资源与完成情况 | 子 agent | 原版 9B | checkpoint2350 |
|---|---:|---:|---:|
| 正式提交数 | 100 | 62 | 67 |
| 平均工具调用 | 20.29 | 42.15 | 41.38 |
| 平均观测数量 | 44900.03 | 7792.74 | 11015.86 |
| 平均观测预算使用率 | 96.56% | 28.88% | 28.47% |

子 agent 的平均 reward 在五类题上都高于本次两组 9B。其观测预算使用率更高；隐藏推理 token 与生成开销不可对齐，不能用表中工具调用数替代总算力。子 agent 与 9B 的条件误差和 exact 指标也不能当作整体正确率。

区间沿用既定的 10,000 次配对 bootstrap（seed 20261019），只反映这组固定题目的差异，不涵盖重复运行 / 求解器随机性；每类仅 20 题。原版与 checkpoint 自身的差值和跨 0 区间见 ../training-rollout-v4-20261021/FINAL-REPORT.md。

输入及校验：native-agent-v2/native-results.json 与 training-rollout-v4-20261021/paired-results.json。重新读取并核验两个归档的全部成员；100 对 prompt JSON 语义完全一致，预算和初始公共 token 也一致。原始 prompt 文件的序列化格式不同，因此使用解码后的 JSON 比较，而非把原始文件字节哈希当作语义相等判据。详见 verification.json。
