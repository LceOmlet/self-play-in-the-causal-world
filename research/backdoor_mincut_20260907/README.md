# 最小后门调整集整数割验收

根因与证明见 [推导文档](../../docs/backdoor-minimum-cut-20260907.md)。
基线提交：`dd62d1d340fbdd162ac84d63b80b0941c2abb281`。
生产改动仅为 `world_space.py` 中共享祖先道德化步骤及最小后门调整集基数算法。
`mincut-acceptance.json` 绑定实际受测源码 SHA-256；每项比较均从该基线 Git 对象
提取原组合枚举函数，仅在离线比较中替换此 owner，不接入任何训练路径。

| 成对比较 | 原耗时 | 新耗时 | 不变项 |
|---|---:|---:|---|
| 96 个结构 × 五类任务，480 个角色列表 | 4.937 秒 | 0.530 秒 | 所有有序角色列表相同 |
| 20 道 best_intervention 完整生成 | 26.883 秒 | 18.935 秒 | 完整任务含 CPT、262 次世界构建的提案顺序相同 |
| 25 道全部五类任务完整生成 | 7.017 秒 | 4.791 秒 | 完整任务含 CPT、107 次世界构建的提案顺序相同 |

对应速度比约 9.31、1.42、1.46；完整生成耗时分别减少约 29.6% 和 31.7%。
这是固定输入的一次成对测量，没有置信区间，不是训练吞吐。并未减少提案个数或更改
强反转接受事件。完整生成中的最小基数函数调用次数也保持 32,667 和 15,704。

`regression-v2.log`：**109 passed, 27855 subtests passed**，53.67 秒。
其中新增测试的 **6,550 个因果角色**来自穷举 2 至 5 节点 DAG，并额外检查节点反向
重标号；独立参照按碰撞点及其后代状态枚举活跃路径，不调用生产道德图函数。
回归还涵盖原有后门集参照、结构生成、任务评分、人口可识别性及环境接口。
这不代表全部仓库 CI 或全部反事实认证已完成。

`baseline-profile.txt` 是改动前 20 道最优干预题的 cProfile 记录，发现 297,950 次
逐子集分离调用。`profile_strong_reversal.py` 是原执行脚本，保留当时的绝对输出路径；
重跑该基线需让 `PYTHONPATH` 指向基线提交的 `src`。不比较带 profiler 与不带 profiler
的秒数。`initial-regression-command-error.log` 保存最初指定错误测试类名导致未执行
测试的命令错误；随后使用正确类名得到 `regression-v2.log`，前者不算通过。

在项目根目录、已有项目依赖和 pytest 的环境中复现：

```bash
PYTHONPATH=src python -m pytest -q tests/test_minimum_backdoor_cut.py tests/test_world_space.py tests/test_population_identification.py tests/test_task_scoring.py tests/test_trl_environment.py tests/test_world_runtime.py tests/test_query_truth.py::QueryTruthOwnerTests::test_backdoor_adjustment_sets_match_known_dag_motifs tests/test_query_truth.py::QueryTruthOwnerTests::test_backdoor_moral_graph_algorithm_matches_path_enumeration
python research/backdoor_mincut_20260907/benchmark_mincut.py
```

复现环境为 `/home/chen/.venvs/dolens-dapo-official/bin/python`，原执行目录为
`/home/chen/runs/strong-reversal-kernel-20260907/candidate`。
运行 benchmark 会重新生成 JSON，应先保留原结果。在本目录运行
`sha256sum -c SHA256SUMS` 可核对保存字节；它不替代数学与运行验收。
