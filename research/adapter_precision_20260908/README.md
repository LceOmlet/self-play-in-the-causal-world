可训练参数精度：真实局部问题，尚非学习主因，2026-09-08

结论与当前决定

这次检查发现了当前官方 FSDP2 接入中的数值损耗，并实现了一个通过 CPU 检查的隔离候选。但不能把它当成“训练约 800 步仍未明显提高”的解释。用户质疑选题后，补查了旧训练的直接对照：第 300、800、850 步保存的 496 个 LoRA 张量均为 FP32，第 800 步的 992 个 Adam 动量张量也均为 FP32。当前 BF16 可训练参数／动量的现象不是两段训练共同的原因。这里只读取旧检查点元数据与优化器状态，没有将它们作为可信能力证据或用于初始化。

候选没有接入训练，没有改活动源码、运行配置或来源验证器，没有暂停 GPU 训练。最终检查仍为原 PID 605645／创建时间 1788844173.03，已记录第 61 次更新；额外验证目录为空。训练继续使用 progress-v1。该精度候选降级保留，继续扩大它的数值验收不会直接定位当前主要学习障碍。

实际影响的量化

取保全的原生第 57、58 步；可训练参数共 21,639,168 个。参数、exp_avg 和 exp_avg_sq 都为 BF16，优化器为 torch.optim.AdamW，学习率 1e-6、betas=(0.9,0.999)、weight_decay=0.1，没有额外 FP32 主参数状态。

固定第 58 步已记录的动量，将高精度参数更新写作

`δθ = -lr*weight_decay*θ57 - lr*(m58/(1-β1^58))/(sqrt(v58/(1-β2^58))+eps)`。

这是对已发生更新的数值分析，不是训练实现，也不是重建整个 FP32 学习历史。特别是 m58、v58 自身已经经过 BF16 舍入，不能把参照称为完整 FP32 训练结果。实际合成权重增量由 `B58*A58 - B57*A57` 计算；利用低秩 Gram 矩阵得到完整 Frobenius 范数，未抽取少数层或按单个元素比例替代整体影响。

| 指标 | 第 58 步结果 | 对主问题的含义 |
|---|---:|---|
| A 实际发生变化的元素 | 39,754 / 10,223,616 | 单看该比例容易夸大问题 |
| B 实际发生变化的元素 | 10,620,967 / 11,415,552 | 主要更新通路仍在改变 |
| 参照 A 项与 B 项的合成更新范数比 | 0.0014075 | 本步 A 项仅约 B 项的 0.14%，不能据 A 大量不动宣称训练基本冻结 |
| 合成更新相对数值差异 | 0.0669794 | 不是 6.7% 的学习损失 |
| 实际／参照更新方向余弦 | 0.9977667 | 不支持“更新方向基本丢失”的推断 |
| 实际／参照更新范数比 | 1.0027246 | 不支持“主要更新幅度消失”的推断 |

实际 BF16 二阶动量的 21,639,168 个非零元素，在 CPU 按其存储 dtype 逐项乘 0.999 时全部保持原值；这与已安装 AdamW 对 exp_avg_sq 先乘 beta2、再加平方梯度的路径对应。说明衰减确有数值损失，但没有测出它对长期能力的损失，不能把它升级成旧训练无收益的主因。对任务能力敏感的方向也不能仅靠全权重范数排除；此处结论是“影响学习的主导性未成立”，不是证明完全无害。

候选与 CPU 检查

固定官方 verl 的 FSDPEngine._build_lora_module 在创建 PEFT adapter 后，为 FSDP1 flat group 的 dtype 约束把可训练 adapter 转成冻结基座 dtype；这个条件也作用于当前 FSDP2。候选仅将该转换限制为 strategy="fsdp"，使 FSDP2 保留 PEFT 默认的 FP32 adapter。前后向仍由原生 MixedPrecisionPolicy 以 BF16 计算，冻结 9B 基座保持 BF16，官方 DAPO、AdamW 公式、学习率、奖励、预算、任务均未改。

1288 个源文件比较仅改变 verl/workers/engine/fsdp/transformer_impl.py；补丁和前后指纹见 candidate.json。未加入生产 source profile，不能把候选视为已验收的生产来源。官方文档依据：[PEFT 默认提升 adapter 精度](https://huggingface.co/docs/peft/developer_guides/troubleshooting)、[FSDP2 原精度优化器与混合精度计算](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)。运行时判断以实际安装的 PyTorch 2.11.0 和固定源码为准。

probe_native_fsdp2.py 直接调用候选官方 LoRA builder、真实 FSDP2、真实 torch.optim.AdamW。使用小型 Qwen2 模块和非零、BF16 可表示的 LoRA B，确认修正前后初始输出逐位相同；修正后参数、梯度、两个 Adam 动量为 FP32，冻结基座无梯度；原生 load_state_dict 将 BF16 参数／动量无损提升到 FP32 并继续第二次更新。该测试只证明框架的混合 dtype 与恢复接口，不代替真实 Qwen3.5-9B GPU 的加载、同步、内存、更新和任务收益验收。

若后续决定接入，额外存储量按本模型参数数目计算：可训练参数、梯度、两个 Adam 动量从两字节变四字节，共约 165.1 MiB，分布于 GPU 和被卸载的优化器内存；这只是张量存储差，不是实际峰值显存保证。应先补全来源声明与真实运行验收，不能覆盖活动 vendor 文件或偷偷切换 dtype。

证据与复现

服务器保全目录为 /home/chen/runs/heldout-learning-20260908/adapter-precision-01；原生权重和动量未放入 Git，文件指纹见 evidence-manifest.json。parameter-order.json 由实际保存的 Qwen3.5 配置在 meta device 上通过原生 Transformers／PEFT 构建，1256 个 named_parameters 与优化器索引、496 个可训练名称／形状、760 个空冻结状态核对。其用途仅为关联原生状态，不产生模型推理。

在服务器原官方 Python 环境运行：

```bash
CUDA_VISIBLE_DEVICES= python audit_saved_updates.py --root /home/chen/runs/heldout-learning-20260908/adapter-precision-01 --output reproduced-update-audit.json
CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/chen/vendor/dapo-official-20260906/verl-fp32-lora-candidate-v1 python probe_native_fsdp2.py --output reproduced-cpu-probe.json
```

本轮实际进展是限定了一个真实问题的作用范围、保留了可复查候选，并排除其作为旧 800 步共同原因的解释。模型能否产生并强化有用解法这一主问题仍未解决；不以候选、测试通过或更多精度报告宣布学习改进。
