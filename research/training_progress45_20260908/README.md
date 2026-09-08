前 45 步实际训练记录与全局影响归纳

本轮没有修改训练算法、环境、奖励、任务生成、预算或运行配置。当前选题要求同时考虑全局影响和明确的可改进性；长跑成本作为约束记录，不据此新开局部优化。见 `docs/global-task-impact-20260908.md` 与整理后的 `docs/rl-correctness-goal-20260907.md`。

`figures/trusted-rewards-and-time.png` 和 `figures/trusted-task-errors.png` 复用原绘图脚本，覆盖 45 次真实更新、52 个生成题组、180 条保留轨迹，其中 174 条提交答案。实际环境误差来自唯一匹配的原始事件；未完成答案不填零误差，严格失败率则计入未完成。不同步使用不同题，不能据这些连线声称学习提高。

显示改动仅为：后续步数刻度、显示实际负的训练奖励、单独注明周期验证耗时、将中介任务的 order 字段准确称为路径边。两张图已实际渲染检查，保留 PNG／SVG／PDF。

新快照 `through45-progress-evidence.tar.gz` 为 299182 字节、13 个文件，SHA-256 为 `7350ec66d242691158f8dd2042696463336d60261c388ded8b8449b2ae413f33`。它保存第 40–45 步完整日志、环境事件、rollout 和既有读取器的 progress 输出。逐文件校验通过。前 39 步复用 `research/training_hold_20260908/user-hold-and-through39-evidence.tar.gz`，哈希 `3f13aaa1c3f090eb8e148700826ab323b31789ea4dd7d1dc49323fc3c18b0d98`。

`inputs/` 的五个 progress 文件均逐字节核对其原归档成员。绘图器核对连续实际步号、每步四条同题轨迹、奖励聚合与官方指标、累计生成组数和环境关联。复现（项目根目录，需要已有 Matplotlib、NumPy 和中文字体环境）：

```python
import runpy, sys
from pathlib import Path
p = "research/dapo_progress_20260908/plot_trusted_progress.py"
inputs = sorted(Path("research/training_progress45_20260908/inputs").glob("*.json"))
sys.argv = [p, "--inputs", *map(str, inputs), "--output", "reproduced-progress45"]
runpy.run_path(p, run_name="__main__")
```

`resource-scale.json` 按第 16–45 步日志汇总真实耗时，保存每步原始值。换算公式为当前均值乘 10000；验证单独按当前 test_freq=25 与一次完整验证耗时计算。仅代表“工作量与速度不变”的情景，不是预测或效率改进计划，也没有因此修改最终更新要求。

现有 `compare_matched_validation.py` 已移除写死的 0→25 标签，支持按实际步号比较完整同题报告，仍要求相同原 25 道任务、输入和采样带身份。用原真实基座／第 25 步重放后，除脚本身份和新增步号元数据外所有结果一致，PNG 逐字节相同；见 `matched-comparator-replay-check.json`。没有伪造第 50 步数据。第 50 步仍需等待完整官方输出与指标，原读取器不会将部分记录当作整轮结果。
