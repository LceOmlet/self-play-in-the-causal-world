# PPO 小时检查与原生指标曲线

## 当前 b32 续训与曲线分界

2026-10-22 用户要求 optimizer minibatch=`4×8=32`，microbatch=1、GA=32，生成仍为4路；明确接受从第1步 checkpoint 立即续训并舍弃未保存更新。远端 `active-run.json` 指定当前 `project-b32`、`training-b32`、`launch-b32-receipt.json` 和服务 `cpt-numerics-ppo-ondemand-b32-20261022.service`。采集器优先读此清单，旧默认路径仅为兼容历史运行。过渡阶段完整负载预检服务为 `cpt-numerics-ppo-batch32-preflight-20261022.service`，进度文件 `batch32-preflight-progress.json`，授权停止的旧服务不应被自动重启。

新系列只收集 `training-b32` 的正式更新；继承 global step=1，但旧训练第2步以后不拼入新曲线。每次更新含8组，按各组原始 task row 评分，再按 case.family 统计五类指标，比例和长度用实际 case 数归一化。奖励/原生误差趋势用完整5/20个已完成优化步；家族窗口只包括含该家族的优化步。首次b32正式更新未完成时，图中留空，状态报告当前 active_group/8 与 completed_cases；不能把预检或单组生成冒充正式更新。旧时间戳快照保留原数据；资源曲线在服务 invocation 改变处断开。

恢复经逐张量核对的actor/value head/Adam状态；旧checkpoint没有RNG，任务流位置恢复，vLLM采样随机状态重新初始化。checkpoint每步原子保存，保留最近三份及每50步存档。PPO数学继续直接调用固定版本未修改的VERL官方函数，验证沿用官方测试与断言，不新增影子计算或容差。以下早期b4说明为历史口径，当前负载以上述配置和实际receipt为准。

2026-10-22 用户要求每小时检查训练、绘制多奖励/多误差/多检查曲线，并明确允许必要时重启训练。对应当前聊天 heartbeat 自动化为 **`ppo`**，每小时执行。旧 GRPO 的 `cpt-world` 自动化仍暂停，本监控不混入旧训练曲线。

本轮设置时训练健康；监控采集器只运行 CPU 代码，不加载模型或申请 GPU，不需要重启训练。已完成首轮真实采集与原生评分一致性检查。

## 每小时执行命令

在 Windows PowerShell 中执行：

```powershell
Set-Location 'C:/Users/Chen/Documents/ChatGPT/credit/repro-benchmark500-20261019'
& 'C:/Users/Chen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' research/ppo_ondemand_20261022/monitor/monitor_once.py
```

固定远端：`chen@e1b4acb6c4c142d5a00d326cec07fb11.hn.takin.cc:10179`；根目录 `/home/chen/runs/ppo-ondemand-from-base-20261022`；服务与运行目录由 `active-run.json` 指定（当前 b32 服务与 `training-b32`）。采集器部署于远端 `monitor/capture_remote.py`，不修改原始训练源码、日志或轨迹。

本地绘图依赖安装在本目录 `.deps`，与训练环境隔离。缺失时执行：

```powershell
& 'C:/Users/Chen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m pip install --target research/ppo_ondemand_20261022/monitor/.deps matplotlib
```

本次环境版本：Matplotlib 3.11.2、NumPy 2.5.4；绘图使用 Agg，无需打开桌面应用。

## 输出与读法

每轮建立新时间戳目录，下载归档并核 SHA256、安全解压后绘图。`latest/` 仅在整个采集和绘图成功后更新；采集失败保留 `latest-check.json` 中的失败原因与最后可信时间，不能将旧图说成本次读取。原始归档在 `archives/`，时间戳快照在 `checks/`，采集尝试在 `check-attempts/`，小时资源序列在 `history.jsonl`。这些目录已排除 Git，任务题目、隐含真值、轨迹与监控数据不自动公开推送。

成功后展示本地绝对路径图片，优先奖励、原生误差和检查图，并链接其他图：

- `latest/rewards.png`：总体及五任务家族的每条回答奖励、组均值；完整 5/20 优化步尾随趋势。总体奖励会受当前家族构成影响，不能把不同家族的首批差异称为学习改善。
- `latest/task-errors.png`：ATE MSE/RMSE/TV；反事实端点 MSE/RMSE/MAE；决策 regret/normalized_regret；后门 edit_distance；中介节点/顺序的 1−F1；提交覆盖率。
- `latest/discovery.png`：中介节点和顺序各自的 precision/recall/F1；后门有效集、中介节点严格匹配、顺序严格匹配的原生布尔字段。
- `latest/checks.png`：PPO policy/value loss、裁剪前梯度范数、近似 KL/clipfrac、观察到的工具错误计数、逐步一致性检查矩阵。
- `latest/efficiency-budget.png`：生成/更新耗时、实际吞吐、模型/工具/总上下文长度、提交率和恰到上下文上限比例、原生预算使用比例与调用数。真实零预算比例在 log 轴外单列，不加 epsilon。
- `latest/context-length.png`：独立上下文图，逐批展示实际32条回答的最终学习序列长度（各点、最小/最大、均值）、固定 32768 上限、初始 prompt/模型输出/工具反馈的精确长度构成，以及模型和工具 token 的分布。总上下文不是只计算模型输出，也不等同于每次生成调用的 prefill prompt 长度。
- `latest/optimizer-state.png`：实际学习率、组合 PPO 损失、更新权重同步耗时、有效策略 token 的整批归一化分母。
- `latest/resources.png`：真实 NVML GPU 利用率、物理显存、功耗；训练 cgroup（含回答沙箱）RAM、相同 boot 与 invocation 内 CPU 计数差分的区间平均、距离最新完整文件 checkpoint 的步数。GPU 利用率每次检查采 5 个相隔约 1 秒的实测值，画本次短窗口均值及 min/max，不冒称整小时平均；物理显存/功耗为本轮最后一个实测点。CPU 首次样本没有区间值，不填零。

`latest/summary.json` 为简要状态；`capture.json` 包含服务、进程、cgroup、学习负载和源码检查；`series.jsonl` 保存每个完成优化步及逐回答原生诊断；`plot-series.json` 保存绘图实际数据、有效/总计数和每个完整窗口的步区间，便于审计。

采集不暂停训练，指标读取与服务状态读取之间可能刚好完成一个新优化步。`captured_global_step`/`completed_updates` 是本次已完整采集的曲线前缀，`live_status_global_step` 是随后读取到的实时状态；两者分别记录，不能将尚未采集的步画进曲线。

图表不足窗口时只显示实际点，不把初始几组扩展成“5 组均值”。每家族窗口按含该家族的已完成优化步计算，按有效回答池化。RMSE 对有效终局的原生 MSE 池化后开方；不是先逐回答开方再平均。没有有效终局时误差缺失，既不补零，也不使用 reward 推断答案误差。布尔字段直接读原生字段，1−F1 为显式数学变换。

## 直接取得误差，而非奖励反推

现有 PPO rollout 保存 reward、accepted answer 的工具事件和原始 task row，未保存内存中的 `episode.terminal_score`。采集器优先读取未来日志已包含的原生 `terminal_score`；现有记录则用**同版原始 owner** 从原始 task row 重建同一 world/seed，并把工具事件中唯一实际被接受的答案交给同一 `CPTWorldEnvironment.act` 评分。反事实直接使用 task row 中训练时已缓存的 truth/certificate，不重新求解或估计。

取得评分器返回的完整 `terminal_score`、精确 Fraction reward，并要求重新计算的 reward 与正式记录**完全相等**；不相等则此次采集失败，不能展示冒充正确的新误差。终局评分只依赖原始世界、任务与答案，不需要重新执行数值工具或重采样实验。该来源在每条记录中标明为 same-version native owner scorer；它与训练时直接写入的 live owner 字段有明确区分。

原始成功查询的 cost、remaining_budget 和 rollout 的 `observations_used` 用于核对预算账本；总预算直接从同一任务的 owner 对象读取。检查 cost=n×测量列数、剩余预算链、费用和原生已用预算一致。工具错误曲线记录明确可观察的 Python 错误输出、协议错误、工具异常和 state_lost，不把代码错误计作基础设施故障，也不声称匹配未明示的所有可能错误类型。

已完成步的诊断缓存到远端 `monitor/diagnostics/`。缓存保留输入 SHA256；每轮核对已完成日志的大小/mtime，避免反复扫描全部历史大轨迹，变化则拒绝静默覆盖。训练源码必须逐文件匹配实际 launch receipt，否则停止原版评分器回算并报告原因。

## 检查口径与恢复授权

32K、optimizer batch32、生成batch4、microbatch1、GA32 以及原学习/生成参数持续核对。检查矩阵包括实际32条轨迹、请求响应配对、采样 logprob 有限、生成/学习 token 对齐、reward 一致、损失/梯度有限、LoRA 改变、梯度累积、回执字段、预算链、原生评分、上下文、掩码及工具调用数。LoRA “改变”在逐步日志中检查首层 B；首个 checkpoint 的 248 层非零另有原先的独立校验，不把前者冒称所有层每步改变。

当前每组只有一次 PPO 更新。KL、policy/value clipfrac 是 optimizer.step 前记录的同权重前后向比较，出现 0 正常；不能因此宣称模型没有学习。policy/value loss 已按整批有效 token 归一化，所以四个 microbatch 的 loss 要**求和而非再平均**；KL/clipfrac 用各回答有效模型 token 数加权。

2026-10-22 14:15 CST 已恢复任务范围内的真实 NVML 监控：运行中内核驱动 595.84，系统用户态库已升级为 595.99.02；从 Ubuntu 官方的 595.84 包仅提取匹配的 NVML 与 nvidia-smi 到远端 `nvml-repair-20261022/runtime`，没有执行驱动安装、系统库替换或训练重启。修复前后训练 PID 都为 597828。`gpu_probe.py` 在独立监控子进程中选择该匹配库；内核版本改变后自动回到系统 nvidia-smi。包来源和 SHA256、提取文件 SHA256、实测结果见远端 `nvml-repair-20261022/repair-receipt.json`。

可手动运行 `/home/chen/runs/ppo-ondemand-from-base-20261022/monitor/nvidia-smi`；这是任务范围的修复，系统默认 `/usr/bin/nvidia-smi` 的安装仍由管理员管理。监控通过实测 NVML 获取 GPU 利用率、物理显存、功耗和温度；修复前的历史数据仍缺失，不能回填。查询失效时保留缺失和原始错误，不估算。PPO 原日志 `peak_cuda_allocated` 仍然是 vLLM CuMem sleep 环境下的 PyTorch allocator 计数，不能当成物理显存峰值。原始值保留在 `series.jsonl` 的 metrics 中，物理占用图只使用新恢复的 NVML 实测。

用户允许必要时修复和重启。健康时保持当前进程，不能为了常规检查重启，也不能因一次 SSH 失败或长 rollout 盲目重启。真实失败时先保留失败日志、checkpoint、未保存区间轨迹和源码，再做最小修复。恢复应校验并加载 actor/LoRA、value head、optimizer、LR/global_step、采样进度及可用 RNG，记录恢复来源和回退区间；不能从基座重训冒称续训，不能把回退重跑区间拼成新增独立步。

**当前入口只有 fresh-run，尚无完整 resume 路径；当前 checkpoint 保存 adapter、value head、optimizer 和 global_step，没有单独保存 RNG 与预取采样器状态。** 因此发生故障时不可直接对旧启动器按“重启”并假装无损续训；需要先实现并验证恢复路径，明确缺失随机状态/回退范围，不能捏造连续性。监控的“完整文件 checkpoint”仅指必需文件存在且非空，不等价于每小时完成完整加载验证。已有 checkpoint-1 的 CPU 加载验证见上级 `evidence/checkpoint-validation.json`。

每小时保存时间和曲线，有新增完成优化步、实质变化、故障/修复/重启、完成或需要用户处理时在当前聊天汇报；无变化且无异常时安静。训练全部完成后生成最终汇总并删除 heartbeat `ppo`。
