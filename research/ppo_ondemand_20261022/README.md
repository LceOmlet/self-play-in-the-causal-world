# PPO：32K、按需数据沙箱；首发 b4，后续 b32 续训

## 2026-10-22 用户要求的 batch=32 续训

用户要求 optimizer minibatch 改为 `4 × 8 = 32`，并明确选择立即从完整的第 1 步 checkpoint 续训，接受舍弃之后未保存的更新。microbatch 仍为 1，梯度累积为 32；每次先用同一版策略连续收集 8 组四路生成，再只执行一次 optimizer 更新。32 条轨迹共用整批有效模型 token 分母；GAE、policy loss、value loss 及归一化继续直接调用固定 commit 的未修改 VERL 官方函数，没有新增 PPO 算法实现或数值容差。

续训源码为远端 `project-b32`，数据为 `training-b32`，服务为 `cpt-numerics-ppo-ondemand-b32-20261022.service`。根目录 `active-run.json` 指定当前运行及 `launch-b32-receipt.json`。旧 `project`、`training` 和启动证据保持为历史记录。第 2 步之后被舍弃的旧更新不计入新曲线或样本数。

恢复通过 PEFT 官方 `set_peft_model_state_dict` 和 PyTorch `load_state_dict` 执行，并逐张量精确核对 adapter、value head、optimizer 状态和 optimizer step。任务流跳过 checkpoint 已消费的组；旧 checkpoint 没有 RNG，vLLM 采样 RNG 重新初始化，因此不声称随机轨迹完全延续。学习率按继承的 global step 继续；其他模型、上下文、采样、学习率和优化参数不变。

`scripts/validate_ppo_batch32.py` 原样运行 45 项官方测试和断言，再直接执行正式 runner 的完整 `32 × 32768` 更新预检，不写替代参考算法；预检参数不用于正式续训。`scripts/launch_ppo_batch32.py` 检查测试、完整负载预检、学习函数 AST 和 checkpoint SHA256 后启动。checkpoint 改为每次更新后写临时目录，再原子 rename；保留最近三份、step 1 和每 50 步长期存档。

实际验证见 [batch32-validation.json](batch32-validation.json)：45 项官方测试通过；完整预检 1,048,576 个上下文 token、806,304 个有效策略 token，old-policy/GAE 191.539 秒、更新 624.171 秒，独立更新进程的 PyTorch allocated 峰值 36.625 GiB，32 次累积后执行一次 optimizer 更新，梯度有限且 LoRA B 改变。该预检不是正式训练样本。15:00 启动正式续训，invocation `bca34792783c4ad2914bf4908b3ca32a`，actor/value head/498 份 Adam 状态精确恢复通过；15:05 实测已进入第 2 步的第 1/8 组 rollout，尚未完成首个正式 b32 优化步。运行时记录四个实际官方函数的源文件路径及 SHA256。

下面各节为 12:07 首发 b4 的历史配置与验证；当前状态以续训 receipt 和小时检查为准。

## 学习和环境约束

基座为 `/home/chen/models/Qwen/Qwen3.5-9B`。正式进程重新加载基座，创建全新 LoRA（r=8、alpha=32）和值头，起始 LoRA B 全零；不继承旧训练或预检权重。总步数 10000，32K 上下文、31497 completion 上限（含工具反馈）、四路生成、microbatch=1、梯度累积=4、学习率 1e-6、线性调度、BF16、非重入梯度检查点及原采样参数保持不变。`learning-load-contract.json` 记录与原配置逐项对照。

学习使用 PPO：GAE（gamma=1、lambda=0.95）、vanilla clipped policy loss（clip=0.2、dual clip=3）及 clipped value loss（clip=0.2、系数=0.1），每批更新一次。共享 actor backbone，独立 FP32 value head；LoRA 使用 FP32 master 参数。四个 microbatch 的损失以整批有效模型 token 数归一化。观察反馈保留在 GAE 序列位置，但不参加策略损失。TRL 仅提供原始多轮生成、工具解析和调用循环，不计算 GRPO advantage 或执行 GRPO 更新。

环境为冻结的 `on-demand-data-v1.1`：`act` 的成功查询只返回 `batch_stored`、id、n、cost、remaining_budget；完整合法数据自动进入独立沙箱的 `public['observations']`，代码直接读取，无需 LLM 转抄。`workspace` 在同一回答内持久化，回答之间隔离。模型自行选择 print/display 的结果仍然可见。原有无限、五类平衡训练采样流保持不变。

## 正确性证据及范围

直接导入未经修改的 [VERL 源码](https://github.com/volcengine/verl/tree/d1390eea0c94467d4ae1387eeda9e2f78f8d73ea)，固定 commit `d1390eea0c94467d4ae1387eeda9e2f78f8d73ea`。`upstream/` 保存所用源码、测试和许可证。实际运行上游原始测试及原始断言：

- multi-turn GAE、value-loss microbatch 归一化及 loss aggregation：34 passed。
- policy loss / rollout correction integration：11 passed。
- 保留上游各自的 `torch.equal`、`torch.testing.assert_close` 等断言和适用 dtype；未借用 GRPO、vLLM 解码或其他低精度算子的误差阈值。

45 项通过不等于存在官方“整条 BF16 PPO 训练误差阈值”。这里没有做这种声明。额外的项目桥接检查与官方测试分开记录：普通 PyTorch linear + FP32 cross entropy 的分块实现，对所测 BF16 tensor 的 logprob 及隐藏层梯度要求 bitwise equal；128 分块梯度不相等而被拒绝，512/1024 通过。248 个含非零 LoRA 的权重层，实际流式传输与 PEFT 官方 merge 的 BF16 结果逐元素完全相等，并恢复冻结基座权重。

## 效率与显存

在完整 4×32768=131072 上下文 token、100788 个有效模型 token 下，预热后测量同一学习负载：

| logprob chunk | old logprob/value + GAE | 更新 | 峰值 allocated |
|---|---:|---:|---:|
| 512 | 26.697 s | 79.093 s | 36.71 GiB |
| 1024 | 26.683 s | 77.960 s | 36.71 GiB |

选取所测合格配置中更快的 1024，差距较小；没有宣称穷尽所有配置的全局最优。所有试验均完成有限梯度更新且 LoRA B 改变。正式训练不复用这些参数。

生成使用 vLLM 原生模型、MP=0、FULL_DECODE_ONLY CUDA Graph、prefix cache、max_num_seqs=4、prefill batch=4096。沿用已有 `TRL 1.14.0+cpt.rollout1` 的驻留调度补丁；本次 PPO 使用其中现有的生成、工具解析和工具循环方法。生成期间 actor 在 CPU；整组工具交互结束后 vLLM level-2 sleep，actor 回 GPU 更新。更新后的权重逐层传输，避免原地 BF16 merge/unmerge 漂移。GDN 使用本地 FLA CUDA kernel，未退回 GDN 的慢速实现。`runtime-source/` 保存实际安装的生成相关源码及既有解码测试报告；该解码报告不作为 PPO 容差。

宿主 NVML 库与驱动版本不一致，但 Torch CUDA 正常。平台入口插件只选择 vLLM 自带的 `CudaPlatform`，由其官方别名启用 `NonNvmlCudaPlatform`，未替换计算 kernel，也未修改宿主驱动。Transformers 需要 kernels <0.17，使用独立的 kernels 0.16.2；依赖装在本任务目录，不污染旧训练环境。

## 运行与复核

远端根目录：`/home/chen/runs/ppo-ondemand-from-base-20261022`。

正式服务：`cpt-numerics-ppo-ondemand-from-base-20261022.service`，Delegate=yes，2026-10-22 12:07:02 CST 启动，invocation `ea894dbbe5d54d5ebf9226291e13f544`。

启动门禁要求官方测试、沙箱预检、完整学习负载更新和权重桥接全部通过。初始化失败尝试单独保留，没有计入采样、reward 或优化步。`fresh-start.json`、`launch-receipt.json`、逐文件 SHA256、原始 rollout/工具事件、metrics 和首步 checkpoint 共同提供可追踪证据。

`source.tar.gz` 保存正式启动 receipt 中的 26 个源文件，`source-files-sha256.json` 为其清单。本地文件与正式启动清单已逐一验证。`evidence.tar.gz` 与 `evidence/` 保存证据快照；运行中的完整日志、后续 rollout 和 checkpoint 继续留在远端。`scripts/collect_ppo_evidence.py` 可重新收集快照，不干扰训练。

2026-10-22 12:31:18 CST，首个真实 PPO 优化步已完成；12:32 起第二批已使用更新后的权重生成。归档快照中第二批已完成 19 次生成调用，随后任务继续运行。

| 正式首批证据 | 实际值 |
|---|---:|
| 四条轨迹的上下文长度 | 32642、32768、26506、32768 |
| 有效模型 token | 81978 |
| 优化步 / 梯度累积 | 1 / 4 |
| 梯度范数（裁剪前，裁剪上限为 1） | 1.048510909，有限 |
| checkpoint 中非零 LoRA B 层 | 248 |
| checkpoint 中已更新至 step=1 的 optimizer state | 498 |
| 首批 rollout 耗时 | 1102.277 s |
| 首步 old-policy/GAE/更新合计，含实际形状首次调优 | 219.191 s |
| 第二批更新权重同步 | 1.155 s |

首批 45 次生成调用的生成时间合计 1082.813 秒，实际整体生成吞吐约 75.71 token/s（包含各轮 prefill；不与短序列纯 decode 吞吐直接比较）。59 个成功查询返回均符合仅回执协议。四条轨迹中一条提交答案，reward 为 0.5262356762，其余未提交而得到 0；这是正常训练数据，不作模型质量结论，也没有重新采样替换。

`checkpoint-validation.json` 重新从 CPU 读取真实 checkpoint：value head 为有限 FP32，全部 adapter 参数有限，248 层 LoRA B 非零，498 份 optimizer state 均为 step=1。checkpoint 的 adapter SHA256 为 `d4a98c5bf7680b07d752da0260b62e552778b3a107a9f3129e3781a95845a843`。完整 adapter、value head 和 optimizer 保存在远端 `training/checkpoints/checkpoint-1/`；仓库保存清单、SHA256 和验证结果。

正式 metrics 原样记录的 `peak_cuda_allocated=77453691904` **不是物理显存峰值**：此进程使用 vLLM CuMem sleep，张量和虚拟地址保留，物理映射通过 `unmap_and_release` 释放；PyTorch allocator 的分配计数不能据此直接解释为物理驻留显存（见实际运行源码 `runtime-source/cumem.py`）。该原始计数约 72.13 GiB，超过设备实际 47.37 GiB，故不将它宣传为硬件占用。上文 36.71 GiB 来自独立、无 vLLM allocator 的完整学习负载预检；正式物理峰值未单独测量，实际四路更新及后续生成已在该设备完成。

源码归档 SHA256：`39546de1d68c5861b7b1571c5d551279cf22fbbcc5514ad32f4ce87e1dada58d`。本次完成的是“正确任务经核验跑起来”：正式服务继续执行剩余训练，尚未完成全部 10000 步。
