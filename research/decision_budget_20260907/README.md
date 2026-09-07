相对决策的有限预算边界证据，基于提交 c20fec9。

这是已完成研究的归档，不是当前主线或题库比例估计。说明与证明见 `../../docs/decision-budget-certificate-20260907.md`；当前任务价值与学习验证优先级见 `../../docs/task-value-and-learning-priority-20260907.md`。

- `verify_decision_budget.py`：实际协议、评分、独立抽样模型最优风险与固定随机流穷尽证书。最终结果为 `decision-budget-acceptance.json`。
- `verify_et_v2_budget_bound.py`：ET-V2 二元单父结果机制的信息界与局部连续参数质量。结果为 `et-v2-budget-bound.json`。其条件结构类上的质量不是全体任务比例。
- `initial-serialization-assertion.log` 保留首次审计脚本断言失败；修正序列化类型后才得到最终通过结果。生产环境未因此改变。
- `SHA256SUMS.json` 固定归档内容，不包含自身。

服务器已有完成记录在 `/home/chen/runs/decision-budget-20260907/audit`。复现程序直接调用生产环境，未接入训练、生成器或奖励计算路径。
