前 39 步结构答案审计

结果与限制见 `docs/structural-answer-signals-20260908.md`。本目录读取已发生的训练轨迹，不实现训练或公开求解策略。

`structural-answer-evidence.tar.gz` 为 441,541 字节，22 个文件，SHA-256：

`8c58ce15e1347153e0d0c517261c4eb97d0219bd5749e90de2441b89b00a417a`

内容包括 16 道原始世界、完整汇总、两条严格正确且非空的结构轨迹（全文与反馈）、原官方优势审计、accepted 元数据、实际执行脚本和所用既有输出读取器。逐文件指纹见 `structural-answer-evidence.sha256.json`。全部原始日志继续使用 `research/training_hold_20260908/user-hold-and-through39-evidence.tar.gz`，其 SHA-256 为 `3f13aaa1c3f090eb8e148700826ab323b31789ea4dd7d1dc49323fc3c18b0d98`。

本地轻量校验：

```text
python research/structural_signal_20260908/verify_evidence.py
```

它核对所有归档字节、执行脚本、前轮结果关联、所有已完成结构答案、16 道世界指纹、两条成功案例的完整性，并从原始联合直方图重算 RQI 干预下 OHT 的实际计数。无需 GPU。

计数限定模型原文比较的两批 50 样本、测量 `[OHT,RFU]` 的请求。轨迹还有早期 10／100 样本的 RQI 请求，不能合并后冒充该段引用的两批数据；本地初版校验因未限定这组请求触发唯一性断言，明确选择条件后重新通过。

服务器原始执行（已通过，输出目录需不存在）：

```text
/home/chen/.venvs/dolens-dapo-official/bin/python /home/chen/runs/heldout-learning-20260908/audit_structural_answers.py --project /home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831 --root /home/chen/runs/heldout-learning-20260908 --output /home/chen/runs/heldout-learning-20260908/structural-signal-01
```

脚本先核对前轮四份冻结快照及源文件前缀哈希，再按完整工具命令序列唯一关联成功输出。图上的非祖先关系提供干预无效的数学依据，生产分布核给出附加数值核对；未运行另一个模型或重新采样轨迹。总体难度、长期学习收益与所有分支正确性不由这份审计保证。
