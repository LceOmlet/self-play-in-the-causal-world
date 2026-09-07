真实最优干预任务的合法实验对照，基于生产提交 5ca4109。

结论见 `../../docs/active-decision-diagnostic-20260907.md`。这不是强化学习实现或模型训练结果；没有程序接入官方 DAPO、奖励或任务生成路径。

- `public_policy.py` 只接受公开提示与工具反馈，执行固定筛选和后门标准化。
- `run_diagnostic.py` 是修正后的诊断入口，最优干预使用当前真值计算路径。运行时指定新的输出目录；不要把初次失败运行的原始结果当作修正结果。
- `initial_cached_runner.py` 保留实际初次运行代码：它混用了旧概率缓存，造成 15 条终局评分失败。原始 85 条完成结果中的零遗憾判定也受影响，不应引用。
- `rescore_frozen_answers.py` 对所有已经锁定的答案、原观察基线答案统一使用当前生产评分器。没有重新采样、修改 CPT 或重选答案；结果为 `rescored-answers.json.gz`。
- `summarize_diagnostic.py` 按世界聚类计算配对差异及 bootstrap 区间，输出 `summary.json`。
- `measure_context.py` 用真实模型 tokenizer 计数命令与反馈的聊天重建，正确结果为 `context-measurement.json`。初次错误的字典字段数已在 evidence 中标为 `context-initial-invalid.json`，不能使用。
- `fresh-runner-check.json` 记录一条原失败案例在修正入口下的完整重跑：公开命令、反馈、答案和原轨迹哈希相同，终局通过。
- `inspect_overlap_case.py` 和 `overlap-case.json` 对已提交答案的一道真实失败题检查父机制，并枚举所有合法实验对关键机制行的单位成本覆盖；`check_mechanism_pooling.py` 与 `mechanism-pooling.json` 用同一批样本验证父机制蕴含的分层合并。这些使用揭示后的真实父集，不计入公开策略正确率。
- `frozen-worlds.tar.gz` 包含 50 个原始任务 pickle 和之前的观察基线记录；`evidence.tar.gz` 包含 100 份公开轨迹、原始失败及原始代码。`runtime-fingerprints.json` 固定生产来源和版本。

在具有项目运行依赖的环境中，将两个 tar 分别解到新的数据目录和证据目录，可以复核原答案：

```sh
python research/active_decision_diagnostic_20260907/rescore_frozen_answers.py \
  --cohort-root /path/to/unpacked-worlds \
  --results /path/to/unpacked-evidence/results
python research/active_decision_diagnostic_20260907/summarize_diagnostic.py
python research/active_decision_diagnostic_20260907/verify_archive.py
```

第一个命令将新的评分 JSON 写入所给 results 目录的父目录，避免修改归档。第二个命令读取仓库里已冻结的 gzip 评分输入。第三个只需要标准库，检查归档字节、全部答案来源和每次实验的实际收费。

服务器原始记录在 `/home/chen/runs/active-decision-diagnostic-20260907`。统计、置信区间和案例诊断只支持这批已用于分析的真实任务，不能当作模型学习、全分布难度校准或所有策略的风险界。
