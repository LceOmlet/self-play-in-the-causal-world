复用真实训练日志与 250 道生产诊断题，重新对齐任务价值和实际可学习性。

结论及限制见 `../../docs/task-value-and-learning-priority-20260907.md`。本归档没有新增强化学习更新。只读复核：

```sh
python research/task_value_learning_20260907/analyze_logs.py
python research/task_value_learning_20260907/analyze_task_value.py
```

前者输出 `training-priority.json`，后者输出 `task-value.json`。程序只需要 Python 标准库。`.gz` 文件解压后保留原始输入字节；`input-provenance.json` 保存原路径和未压缩 SHA-256，`SHA256SUMS.json` 保存归档文件哈希且排除自身。

数据来源：

- 官方运行：服务器 `/home/chen/runs/rl-correctness-goal-20260907/official_execution/run-02-mask-v1-audit-only` 的 train.log、rollouts/1.jsonl 及 environment/environment-78361.jsonl。
- 旧训练：`/home/chen/runs/dolens-grpo-rewardv10-cfisolated-85cb577-20260902-195713-resume250-10k/train.log`。旧实现已排除，不能用作当前官方 DAPO 的证据。
- 250 题诊断：`/home/chen/runs/environment-validation-20260907` 的汇总及两批 profiles。这里复核原始保存结果，没有用当前内核重新采样或重评分。
- `aborted_generation_profile.py` 和 `generation-profile-failed.log.gz` 保存本次转向前的失败纯推理尝试。初始化未找到 `ninja`，没有吞吐结论；不继续这项次要分析。该脚本依赖审计服务器的原始张量与模型，不是生产入口。

原临时日志分析把错误类型写成 `error`，遗漏了真实 `protocol_error`。归档脚本已经修正，完整 44 条动作记录分为 27 个有效实验、13 个协议错误与 4 个有效终局答案；不能引用临时的零错误结果。
