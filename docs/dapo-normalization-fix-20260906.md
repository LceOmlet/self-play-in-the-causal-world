# DAPO：采用上游实现后的状态（2026-09-06）

本文现为历史审计记录。当前入口迁移至完整官方 verl DAPO，见
[接入与正确性状态](official-dapo-integration-20260906.md)。旧 TRL 入口已停用，
旧强化学习检查点已被用户判定不可用于初始化或评估证据；下文不是当前运行配置。

长期目标仍是环境与奖励的能力判别性、强化学习效率，以及整个链路的正确性。算法固定为用户指定的 DAPO。RLOO 和本地 DAPOTrainer 子类已撤回。

## TRL 可以直接使用什么

官方 TRL 1.12.0 提供 GRPOTrainer 的 DAPO 损失、组内标准化和非对称裁剪配置。训练类的名称叫 GRPOTrainer，不代表 loss_type="dapo" 时使用了别的损失。

已在独立目录下载官方 TRL 1.12.0、Liger Kernel 0.8.2 wheel，未修改其中源码。实际 CUDA 融合损失、TRL 非融合损失与独立论文公式对照通过：覆盖变长输出、工具反馈掩码、正负优势、0.2/0.28 裁剪、累计比例 4/4 和 2/4、评估模式及可选采样修正。比较损失、隐藏状态梯度和输出头梯度，FP32 容差 rtol=2e-4、atol=2e-6；两项测试通过，其中融合对照包含六组配置。

这些测试是标准实现的验收，不是用于训练的第二份算法。它们没有运行 Qwen9B 完整在线训练，因此不能单独证明端到端正确或训练会提升能力。

## 已撤回的自写代码

撤下自行添加到 TRL 的动态采样、对应配置字段和两个控制流测试，移除入口中的动态采样参数及其专用恢复限制。原始补丁与当时的审计数据归档保留，不再作为当前实现的证据。

撤下自行推导的归一化修法。当前补丁逐字采用官方 TRL 1.12.0 的 GRPOTrainer.compute_liger_loss，回移到现有 TRL 1.10.0 的拥有者类中；入口直接实例化 trl.GRPOTrainer。没有本地 Trainer 子类、备用训练循环或自写梯度更新。

采用回移是当前兼容状态，不表示已经整包升级：旧环境仍有 Qwen3.5/vLLM 性能补丁，入口所用 vllm_rollout_residency、vllm_enable_prefix_caching、vllm_speculative_config、vllm_sleep_level 不属于 1.12.0 原生配置。整包升级需要迁移这些接口；这不构成重写算法的理由，也不能声称迁移做不到。

新补丁只把全局有效 token 数传给 Liger，并使用上游的累计缩放。旧版每条答案分别归一化后再平均，在答案长度不同时不等于全局 token 均值，这是源码确定的损失偏差。该偏差不等于训练必然没有效果。

上游函数源码 SHA-256：`7adcac0dac85b016d5b2a796917475df201de62c1dd11b656bc3529ad0b529d9`。依赖及加载源码见 `patches/rl-owner-manifest.json`；哈希用于追踪实际执行代码，不代替数学或运行验证。

## 正确性与完整 DAPO 的区别

标准强化学习更新是否实现正确，和是否复现 DAPO 论文全部训练配方，是不同的判断。缺少动态采样可以造成零优势组消耗训练步，但不能单凭这一点宣称强化学习实现错误。

核对的 TRL 1.12.0 主训练路径没有筛掉零方差奖励组并补采样的流程；is_std_zero 用于记录指标。因此只设置 loss_type="dapo" 不能声称启用了完整 DAPO 动态采样。官方 verl-recipe 的 DAPO 实现提供了该流程；如必须使用这一部分，应接入官方实现，不能继续私写训练调度来冒充标准实现。

当前入口配置为 DAPO 损失、组内优势标准化、0.2/0.28 裁剪、beta=0；原始环境奖励仍只计入一次。当前仍保留已有的 vLLM 采样修正，不能将它未加区分地称为论文中所有细节的逐项复现。

## 验证与运行状态

正式训练保持停止：最后日志步 899，最近保存 checkpoint-850。

原生上游验证目录：`/home/chen/runs/algorithm-audit-20260906/upstream-trl-1.12.0/`，其中 `native-dapo-tests.txt`、`native-provenance.json` 保存测试结果和实际加载路径。

当前上游回移版本通过 26 项回归测试（59.084 秒）：DAPO 损失/梯度、原奖励、采样 logprob 防护及环境适配。实际安装函数与官方 wheel 中函数逐字一致；安装结果记录于 `backport-install-report.json`，回归日志为 `backport-regression-tests.txt`。修改的入口、观察脚本和测试通过 Ruff。

此前 `owner-audit/runtime-fixed` 的 9B 一步审计使用了现已撤下的自写动态采样，属于历史实验，不能当作当前标准路径的端到端验收。没有据此恢复正式训练。

目前不能宣称整个项目的强化学习已经全部验证正确：整包原生 TRL 与现有入口的接口迁移、实际完整模型和 rollout 的最终验收尚未完成；完整动态采样也尚未接入官方实现。这是具体未完成项，不是上游不可用的证明。环境可识别性另有独立问题，不能与训练实现正确性混为一谈。

## 官方来源

- [TRL 1.12.0 GRPOTrainer 源码](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py)
- [TRL GRPOConfig 文档](https://huggingface.co/docs/trl/v1.12.0/en/grpo_trainer)
- [DAPO 原论文](https://arxiv.org/html/2503.14476v1)
- [官方 verl DAPO recipe](https://github.com/verl-project/verl-recipe/tree/main/dapo)
