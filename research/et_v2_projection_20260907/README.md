# ET-V2 投影与一般密度验收

生产改动是 `src/cpt_world/world_space.py::_parent_interaction_projection`。
基线提交为 `9f8a343fca62febfb3f962b5e264d7f399f3f161`；审计脚本从 Git 读取原函数，
只在离线成对比较中替换此计算 owner。没有接入新的训练器或改变生成随机律。
一般证明见 [推导文档](../../docs/et-v2-projection-and-density-20260907.md)。

最终源码 SHA-256：`b4984a18d9a13650231c75663cec5131ba697628e9fa09a9dc4d2d8bb5a41e13`。
最终测量见 `projection-acceptance.json`：

| 路径 | 原耗时 | 新耗时 | 范围 |
|---|---:|---:|---|
| 64 个世界构建 | 3.776 秒 | 2.459 秒 | 结构及每次构建后 RNG 状态相同，CPT 最大差异 1.89e-15 |
| 25 道完整任务生成 | 8.879 秒 | 7.121 秒 | 全部五类任务；107 次世界构建；选择及非 CPT 字段相同 |
| 五父节点五态全交互投影 | 0.437 秒 | 0.073 秒 | 约 5.96 倍；峰值 Python 分配 4,367,924→2,180,900 字节 |

单项投影耗时为三次测量的中位数；完整生成是一次成对测量，不是置信区间。
`tracemalloc` 测的是 Python 分配峰值，不能当作进程 RSS。最小二态单父轴例的
峰值多 64 字节，不能声称所有输入都节省内存。浮点求和顺序改变，不保证所有种子
逐字节相等；等价性在实数算子层面成立，固定题的选择一致属于有限验收。

最终 `final-tests.log`：**107 passed, 41 subtests passed**，72.38 秒。
覆盖新增独立有理数投影参照、轴与状态重标号、原 world_space、人口可识别性、
任务评分、环境接口和 world_runtime 回归。`test_trl_environment.py` 是历史命名的
环境测试文件，其通过不表示训练使用 TRL。训练仍使用已审计的官方 verl DAPO。
不将这一组测试表述为整个仓库 CI 全部通过。

`general-geometry-acceptance.json`：43 个精确基向量、两个直接符号 Jacobian、
2 至 5 态 CLR Jacobian、96 个真实机制核对通过；最大尺度误差 2.22e-16，
log-score 重建误差 1.29e-14，二态密度退化关系通过。
`weak-outcome-bound-acceptance.json`：48 个真实合法任务与奖励 owner 检查通过，
验证弱结果机制的效应上界和零答案质量下界；不是总体易题比例估计。

保留 `projection-initial-acceptance.json` 和
`projection-before-single-axis-acceptance.json` 作为中间候选记录；它们不是最终源码
的验收结果。`environment-regression.log` 保存首次测试命令引用不存在的
`tests/test_rendering.py` 而未执行测试的错误；最终命令已更正，该错误不算通过。

在项目根目录、具备项目依赖和 pytest、SymPy 的 Python 环境中复现：

```bash
PYTHONPATH=src python -m pytest -q tests/test_anova_axis_projection.py tests/test_world_space.py tests/test_population_identification.py tests/test_task_scoring.py tests/test_trl_environment.py tests/test_world_runtime.py
python research/et_v2_projection_20260907/benchmark_projection.py
python research/et_v2_projection_20260907/verify_general_geometry.py
python research/et_v2_projection_20260907/verify_weak_outcome_bound.py
```

三个脚本重新生成相应 JSON，运行前应保留原验收文件。服务器复现环境为
`/home/chen/.venvs/dolens-dapo-official/bin/python`，原执行目录为
`/home/chen/runs/et-v2-projection-20260907/candidate`。
本轮不重复 GPU 更新：已序列化训练题的 CPT、算法及运行配置未改变。

归档原始字节可在本目录运行 `sha256sum -c SHA256SUMS` 核对；它用于验证保存完整性，
不替代上面的数学与运行验收。
