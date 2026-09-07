# 数值终端系数消元：已验证等价，但本轮不合入

详见 [证明与结论](../../docs/numeric-terminal-compilation-20260907.md)。
基线提交为 `95b84544f70aabdca74d73de3523e29e50549354`。
候选 counterfactual_solver.py 的 SHA-256 为
`b73d9d91111a7c35d72b78314c96e823cdf521da9ffe43c82b3a7e230319e80d`。

保留 `numeric-terminal-candidate.patch` 和独立测试，以便后续复用等价化简。
正式求解器保留基线，当前 tests 目录不包含依赖未合入候选的测试。

- `numeric-v1-summary.json`：完整 74 题对比；原 51 道成功全部保留，无端点误差契约
  不一致，也无新增认证；仍有 9 次整题超时、14 次未认证错误。
- `numeric-focus-summary.json`：六道题的同期串行比较。`complete: false` 明确表示
  这不是另一轮完整 74 题，而不是进程还在运行；六道均已完成观测。
- `core-tests.log`：43 项测试、20 项子测试通过，含独立有理数系数和运输端点参照。
- `case-43-first-contraction.json`：生产基线首个消息作用域及实际上下文分量。
- `response-cut-identity.json`：15 个原响应 LP 核对；均匀固定边缘仍可表达完整切割
  目标，独立成对极值不能代替联合响应。
- `replay-evidence.tar.gz`：完整候选回放、六题同期双版本回放、初始探针和核心测试日志。
- `inputs.tar.gz`：74 道原始冻结题和原 candidates.json；逐题 SHA-256 与完整对比报告
  的 inputs_sha256 一一核对。没有重新生成这些世界和 CPT。

第 43 题首个符号消息 360,000→9,000 单元，模型约 3.4 秒建成，但仍未全局认证。
同期第 14、16、69 题总耗时上升；第 61 题下降。该题从未建完模型推进到求解阶段后，
整题峰值 RSS 从约 178.5 MiB 上升到 374.8 MiB，不能声称整体节省内存。
这些结果支持继续研究系数化简后的整体响应约束，不支持直接部署本候选。

在隔离检出基线的工作树中应用补丁，并用已有项目依赖、pytest 和 PySCIPOpt 的环境复现：

```bash
git apply research/symbolic_contraction_20260907/numeric-terminal-candidate.patch
PYTHONPATH=src python -m pytest -q research/symbolic_contraction_20260907/candidate_test_numeric_terminal_compilation.py tests/test_counterfactual_solver.py tests/test_indirect_diagonal_transport.py
```

完整回放脚本支持 `CPT_CF_INPUT_ROOT` 指向解压后的冻结输入目录，
`CPT_CF_AUDIT_ROOT` 指向新的输出目录；新目录可避免与旧观测混用：

```bash
CPT_CF_INPUT_ROOT=/tmp/cf-frozen CPT_CF_AUDIT_ROOT=/tmp/cf-replay PYTHONPATH=src python research/symbolic_contraction_20260907/replay_numeric_contraction.py run numeric-v1
python research/symbolic_contraction_20260907/verify_response_cut_identity.py
```

冻结回放保持 8 GiB 地址空间上限、每端点 5 秒、父进程约 11 秒整题限制。
`CASE_INDICES` 可用于明确的少量题诊断，但不能据此声称完整回放。
脚本的 `prepare` 是历史输入生成入口，本次复现使用归档输入，不调用该入口。
原执行环境为 `/home/chen/.venvs/dolens-dapo-official/bin/python`；输入 pickle
只用于本仓库自身生成并核对哈希的冻结数据。

在本目录运行 `sha256sum -c SHA256SUMS` 验证归档原始字节。Ruff 检查源码与新增
验证程序通过；未对历史观察脚本作无关格式清理，未宣称整个仓库 CI 全部通过。
