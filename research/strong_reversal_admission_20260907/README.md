强反转接受条件的等价化简验收

证明与适用范围见 [推导文档](../../docs/strong-reversal-admission-20260907.md)。
基线提交 `aa86218d0280b8abdfd34bdc07788501cad17d10`。
生产改动仅为 `world_space.py` 的接受谓词：原因果值范围低于强反转阈值时，
省去观察分布推断。关系函数的输出语义及完整任务提案、接受事件保持不变。

| 固定生成任务 | 观察查询次数 | 完整生成平均秒数（原→新） | 世界构建数 |
|---|---:|---:|---:|
| 20 道最优干预 | 242→167 | 18.753→18.208 | 262 |
| 25 道全部五类任务 | 62→41 | 4.806→4.690 | 107 |
| 偏移槽的另 10 道最优干预 | 168→102 | 12.165→11.716 | 178 |

每条路径两次执行，交换先后顺序；每次都严格比较包含 CPT 的完整输出和有序世界提案
序列。干预查询次数没有变化。`admission-acceptance.json` 保存各次时间、全部 seed ID、
输出/提案序列哈希、受测源码哈希。这是固定输入的成本比较，没有置信区间，
不代表其他配置或训练吞吐。少算了必定失败候选的一部分，没有提高原事件接受概率。

`independent-tests.log` 保存独立有理数枚举、并列最优和浮点阈值边界结果：
3 项测试、3,344 项子测试通过。原生成、总体可识别性、评分与环境接口回归见
`regression.log`：107 项测试、3,349 项子测试通过，50.39 秒。
首次本地尝试因 PySCIPOpt 缺失在导入阶段结束，不计为测试执行。

在本仓库根目录、已有项目依赖的环境中复现：

```bash
PYTHONPATH=src python -m pytest -q tests/test_strong_reversal_admission.py tests/test_world_space.py tests/test_population_identification.py tests/test_task_scoring.py tests/test_trl_environment.py tests/test_world_runtime.py
python research/strong_reversal_admission_20260907/benchmark_admission.py
python research/strong_reversal_admission_20260907/verify_archive.py
```

原执行 Python 为 `/home/chen/.venvs/dolens-dapo-official/bin/python`；工作树为
`/home/chen/runs/strong-reversal-admission-20260907/candidate`。
重跑 benchmark 会覆盖结果文件，应先另存原证据。源码中的旧 trainer 接口回归文件名
不代表训练使用 TRL；实际训练依然要求固定官方 verl DAPO，此项未启动训练。
