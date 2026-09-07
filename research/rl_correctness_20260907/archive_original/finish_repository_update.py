"""Update the saved project with verified run-02 evidence, figures and status."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1] / 'self-play-in-the-causal-world'
RESEARCH = PROJECT / 'research/rl_correctness_20260907'
execution = RESEARCH / 'official_execution'
shutil.copytree(ROOT / 'official_execution/figures', execution / 'figures', dirs_exist_ok=True)
plot = (ROOT / 'official_execution/plot_verified_update.py').read_text(encoding='utf-8')
plot = plot.replace("HERE.parents[1]/'rl_server_20260906_1725/plot_training_curves.py'", "HERE/'original_plot_training_curves.py'")
(execution / 'plot_verified_update.py').write_text(plot, encoding='utf-8', newline='\n')

goal_path = PROJECT / 'docs/rl-correctness-goal-20260907.md'
goal = goal_path.read_text(encoding='utf-8')
goal += '''

2026-09-07 本轮实际执行验收完成：修复后的 run-02 完成一次全新基座官方更新，质量逐条传入奖励，rollout→actor→四个 loss microbatch 的动作掩码、优势和重要性权重逐项对齐；19,830 个动作 token 进入全局分母，4,018 个工具 token 排除。248 个 LoRA B 张量完成一次与解析 Adam 首步一致的更新，760 个冻结张量无梯度。详见 `dapo-execution-audit-20260907.md` 及其中实际验收 JSON。该有限配置的一次执行检查通过，B/E 不再处于“尚无修复后实际更新”的状态；跨分支、任务分布与长期收益仍不由这一次运行保证。

后续活动优先回到 C/C2 的联合生成条件及基线误差分布：本次最高质量 0.95531 的真实答案为零向量，而隐藏 ATE 的总变差幅度仅 0.01355869。已有强反转或观察/因果差异条件不自动约束零答案误差；应由实际筛选不等式推导其可行区域和推送分布，再判断是否需要改契约。单题事实不能当作总体坍缩比例，不能通过临时调奖励、改配比或课程掩盖。A/A2 的 23 道未认证题、D 的总体预算校准和 C 的密度正确高效提案继续保留。

代码保存已纳入原 GitHub 仓库及 `codex/rl-correctness-official-dapo-20260907` 分支；本地正式副本为 `credit/self-play-in-the-causal-world`，不再依赖 audit 中的无独立 Git 镜像。代码、数学证明、复现程序、失败记录及通过结果及时提交推送；原始模型/梯度张量仍在审计服务器。见 `repository-maintenance.md`。
'''
goal_path.write_text(goal, encoding='utf-8', newline='\n')

integration = PROJECT / 'docs/official-dapo-integration-20260906.md'
content = integration.read_text(encoding='utf-8')
old = '''is validating this repair from the original base. Its execution acceptance must
pass before claiming that this actual update is correct.'''
new = '''completed the repair audit from the original base and passed actual
trajectory-to-loss, gradient and first-step Adam checks. See the
[2026-09-07 execution report](dapo-execution-audit-20260907.md) for evidence and
the limits of this single-update result.'''
assert old in content
integration.write_text(content.replace(old, new), encoding='utf-8', newline='\n')

readme_path = RESEARCH / 'README.md'
readme = readme_path.read_text(encoding='utf-8')
old = '显式修复 action mask 后，从原始基座进行一次隔离更新。仍以实际 `execution-acceptance.json` 为准；文件不存在时表示尚未完成验收。一次更新不能证明收敛或长期能力改善。'
new = '已从原始基座完成一次隔离更新并通过实际张量验收，见 `official_execution/run-02-mask-v1-audit-only/execution-acceptance.json`。19,830 个动作 token、4,018 个工具 token 的掩码传播已逐段核对。一次更新不能证明收敛或长期能力改善。'
assert old in readme
readme_path.write_text(readme.replace(old, new), encoding='utf-8', newline='\n')

matrix = RESEARCH / 'official-dapo-validation-matrix.md'
text = matrix.read_text(encoding='utf-8')
text += '\n\nrun-02 已完成并通过实际张量检查，详见 ../../docs/dapo-execution-audit-20260907.md。该更新的实际掩码、质量、优势、重要性权重、loss 和一次 Adam 参数更新均已核对；非零 overlong shaping、动态丢弃和 TIS 上截断没有在这一次运行中发生，其证据仍来自标准函数测试或历史实际执行。长期训练与任务分布校准尚未完成。\n'
matrix.write_text(text, encoding='utf-8', newline='\n')

for name in ('verify_repository_copy.py', 'import_verified_evidence.py', 'finish_repository_update.py'):
    shutil.copyfile(ROOT / name, RESEARCH / 'archive_original' / name)
manifest = {file.relative_to(RESEARCH).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(RESEARCH.rglob('*')) if file.is_file() and '__pycache__' not in file.parts
            and file.name != 'archive-manifest.json' and file.suffix != '.pyc'}
(RESEARCH / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')

files = [PROJECT / 'docs' / name for name in ('rl-correctness-goal-20260907.md',
         'official-dapo-integration-20260906.md', 'dapo-execution-audit-20260907.md', 'repository-maintenance.md')]
files += [file for file in RESEARCH.rglob('*') if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.pyc']
with tarfile.open(ROOT / 'verified-repository-update.tar.gz', 'w:gz') as archive:
    for file in files:
        archive.add(file, arcname=file.relative_to(PROJECT).as_posix())
print(json.dumps({'files': len(files), 'bytes': (ROOT / 'verified-repository-update.tar.gz').stat().st_size}))
