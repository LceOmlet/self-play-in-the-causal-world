训练曲线更新至第 51 步

按用户最终要求取消全部额外推理验证，训练继续。这里只读取已产生的训练日志，复用既有绘图器；没有新增模型生成。

`figures/trusted-rewards-and-time.png` 展示分任务奖励与训练耗时，`figures/trusted-task-errors.png` 展示环境记录的数值误差和结构严格失败比例。覆盖 51 次原生更新、59 个生成题组、204 条保留轨迹。训练题随步数变化，连线用于观察训练表现和异常，不解释为同题能力提升。

第 50 步已保存原生模型／优化器／数据进度并成功恢复至第 51 步，但取消同步验证时尚未打印第 50 步最终训练指标。因此保留该步已有的四条环境奖励／误差，官方聚合奖励和耗时留空，不推算或补写。绘图器新增可选 `--native-progress`：只有有原生已完成进度时才允许这种缺口；其余四轨迹、事件唯一关联、连续步号及累计生成数检查保留。

复现：在项目根目录运行已有 Python／Matplotlib 环境，输入 `research/training_progress45_20260908/inputs/` 中截至第 39 步的四个文件，加本目录 `inputs/through50.json`、`inputs/through51.json`；调用 `research/dapo_progress_20260908/plot_trusted_progress.py --inputs ... --native-progress research/training_progress51_20260908/native50-progress.json --output reproduced-progress51`。全部输入路径和 SHA-256 见 `figures/manifest.json`。

新输入逐字节来自 `../background_validation_20260908/validation-disabled-and-through51-evidence.tar.gz` 的两段快照。第 50 步证明是已保全原生检查点的 dapo_progress.json；复制、恢复及真实第 51 步证据见该目录 README。历史第 25 步固定验证耗时保留在图中，作为已发生的成本，不代表仍在运行验证。
