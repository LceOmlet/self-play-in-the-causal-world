"""Compare the archived per-case native run with the completed v4 9B runs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

from compile_results import paired_summary, summarize
from verify_rollout_download import Parts, sha

SUITE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def extract_verified(tar, expected):
    selected, seen = {}, set()
    for member in tar:
        assert member.isfile() and member.name not in seen
        seen.add(member.name)
        blob = tar.extractfile(member).read()
        assert hashlib.sha256(blob).hexdigest() == expected[member.name], member.name
        if member.name.endswith(('/prompt.json', '/private-result.json')):
            selected[member.name] = json.loads(blob)
    assert seen == set(expected)
    return selected, len(seen)


def compare(native, v4, output):
    assert read(native/'final-audit.json')['native_terminal_cases'] == 100
    assert read(v4/'verification.json')['passed'] and read(v4/'local-verification.json')['passed']
    for root, manifest in ((native, 'export-sha256.json'), (v4, 'delivered-checksums.json')):
        expected = read(root/manifest)
        for name in ('native-results.json', 'native-owner-transcripts.tar.gz', 'archive-members-sha256.json') if root == native else ('paired-results.json', 'archive-members-sha256.json', 'archive-restore-index.json'):
            assert sha(root/name) == expected[name], name
    native_hashes = read(native/'archive-members-sha256.json')
    with tarfile.open(native/'native-owner-transcripts.tar.gz', mode='r|gz') as tar:
        a, native_count = extract_verified(tar, native_hashes)
    index = read(v4/'archive-restore-index.json')
    parts = [v4/p['name'] for p in index['parts_in_order']]
    for p, expected in zip(parts, index['parts_in_order']):
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256']
    with tarfile.open(fileobj=Parts(parts), mode='r|gz') as tar:
        q, v4_count = extract_verified(tar, read(v4/'archive-members-sha256.json'))
    audit = read(native/'owner-audit.json')
    for name in ('server.py', 'capsule.py', 'data/manifest.json', 'data/eval100.json'):
        assert native_hashes['source/'+name] == audit['source_sha256'][name] == sha(SUITE/name), name
    evaluation = read(SUITE/'data/eval100.json')
    agents, models = read(native/'native-results.json'), read(v4/'paired-results.json')
    ids = evaluation['task_ids']
    assert set(agents) == set(models) == set(ids) and len(ids) == 100
    rows = {}
    for task in ids:
        ap = 'owner/agent/'+task+'/'
        assert a[ap+'private-result.json'] == agents[task]
        assert agents[task]['submitted'] and agents[task]['reason'] == 'terminal_answer'
        for model in ('base', 'checkpoint2350'):
            prefix = model+'/cases/'+task+'/'
            assert a[ap+'prompt.json'] == q[prefix+'prompt.json'], task
            assert q[prefix+'private-result.json'] == models[task][model]
            assert models[task][model]['valid_terminated']
            assert agents[task]['budget'] == models[task][model]['budget']
            assert agents[task]['initial_public_qwen_tokens'] == models[task][model]['initial_prompt_tokens']
        rows[task] = {'native_agent': agents[task], **models[task]}
    output.mkdir(parents=True, exist_ok=False)
    (output/'.gitattributes').write_text('* -text\n', encoding='utf-8', newline='\n')
    report = {'metric': 'terminal-quality-v10 reward, not accuracy', 'paired_cases': 100,
              'native_run': 'native-subagent-per-case-v2; one fresh fork_turns=none solver per case',
              'model_run': 'training-rollout-v4-20261021; no extra per-turn 8K cap',
              'new_solver_runs': 0, 'legacy_20_case_history_agents_included': False,
              'strict_hidden_compute_matched': False, 'groups': {}}
    lines = ['# 逐题子 agent 与本次两组 9B 的比较', '',
             '复用已完成的 100 个逐题独立 native 子 agent 结果，与本次原版 / checkpoint2350 各 100 条 v4 轨迹配对。没有重新求解或重采样；不使用旧 8K 模型分数，也不使用早期保留 20 题历史的 agent 结果。', '',
             '**指标为 reward，不是正确率。任务、工具、capsule、观测预算和逐题隔离已匹配；native 隐藏推理 / 生成算力、完整系统上下文及额外传输限制未与 9B 严格匹配。以下说明本次任务表现，不是严格等算力优劣结论。**', '',
             '| 类别（每类 20 题） | 逐题子 agent | 原版 9B | checkpoint2350 |', '|---|---:|---:|---:|']
    for group, tasks in {**evaluation['task_ids_by_family'], 'Overall': ids}.items():
        metrics = {m: summarize([rows[t][m] for t in tasks]) for m in ('native_agent', 'base', 'checkpoint2350')}
        comparisons = {}
        for right in ('base', 'checkpoint2350'):
            pairs = [(rows[t]['native_agent'], rows[t][right]) for t in tasks]
            d = paired_summary(pairs)
            deltas = [left['reward']-other['reward'] for left, other in pairs]
            assert (d['agent_wins'],d['ties'],d['qwen_wins']) == (sum(x>0 for x in deltas),sum(x==0 for x in deltas),sum(x<0 for x in deltas))
            comparisons['native_minus_'+right] = {'mean_difference': d['mean_agent_minus_qwen'], 'paired_bootstrap_95pct': d['paired_bootstrap_95pct'], 'native_wins': d['agent_wins'], 'ties': d['ties'], 'model_wins': d['qwen_wins'], 'bootstrap_draws': d['bootstrap_draws'], 'bootstrap_seed': d['bootstrap_seed']}
        report['groups'][group] = {'n': len(tasks), 'models': metrics, 'comparisons': comparisons}
        lines.append('| '+group+' | '+' | '.join(f"{metrics[m]['mean_reward']:.4f}" for m in ('native_agent','base','checkpoint2350'))+' |')
    overall = report['groups']['Overall']
    lines += ['', '| 配对比较 | 平均 reward 差 | 95% bootstrap 区间 | 子 agent 胜 / 平 / 模型胜 |', '|---|---:|---|---:|']
    for model in ('base','checkpoint2350'):
        d = overall['comparisons']['native_minus_'+model]
        ci = d['paired_bootstrap_95pct']
        lines.append(f"| 子 agent − {model} | {d['mean_difference']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {d['native_wins']} / {d['ties']} / {d['model_wins']} |")
    lines += ['', '| 资源与完成情况 | 子 agent | 原版 9B | checkpoint2350 |','|---|---:|---:|---:|']
    resources = [('submitted','正式提交数'), ('mean_tool_calls','平均工具调用'),
                 ('mean_observations','平均观测数量'), ('mean_budget_fraction','平均观测预算使用率')]
    for field, label in resources:
        values = [overall['models'][m][field] for m in ('native_agent','base','checkpoint2350')]
        cells = [str(x) if field == 'submitted' else f'{x:.2%}' if field == 'mean_budget_fraction' else f'{x:.2f}' for x in values]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '子 agent 的平均 reward 在五类题上都高于本次两组 9B。其观测预算使用率更高；隐藏推理 token 与生成开销不可对齐，不能用表中工具调用数替代总算力。子 agent 与 9B 的条件误差和 exact 指标也不能当作整体正确率。', '',
              '区间沿用既定的 10,000 次配对 bootstrap（seed 20261019），只反映这组固定题目的差异，不涵盖重复运行 / 求解器随机性；每类仅 20 题。原版与 checkpoint 自身的差值和跨 0 区间见 ../training-rollout-v4-20261021/FINAL-REPORT.md。', '',
              '输入及校验：native-agent-v2/native-results.json 与 training-rollout-v4-20261021/paired-results.json。重新读取并核验两个归档的全部成员；100 对 prompt JSON 语义完全一致，预算和初始公共 token 也一致。原始 prompt 文件的序列化格式不同，因此使用解码后的 JSON 比较，而非把原始文件字节哈希当作语义相等判据。详见 verification.json。', '']
    (output/'COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    write(output/'comparison.json', report)
    write(output/'threeway-results.json', rows)
    table = io.StringIO(newline='')
    writer = csv.writer(table, lineterminator='\n')
    writer.writerow(['task_id','family','native_reward','base_reward','checkpoint2350_reward','native_minus_base','native_minus_checkpoint2350'])
    for task in ids:
        a,b,c = [rows[task][m]['reward'] for m in ('native_agent','base','checkpoint2350')]
        writer.writerow([task,rows[task]['native_agent']['family'],a,b,c,a-b,a-c])
    (output/'threeway-results.csv').write_text(table.getvalue(), encoding='utf-8', newline='\n')
    write(output/'verification.json', {'passed': True, 'paired_cases':100, 'native_archive_members_verified':native_count,'v4_archive_members_verified':v4_count,'all_prompt_json_values_identical':True,'all_budgets_and_initial_public_tokens_identical':True,'native_owner_and_dataset_source_sha256_match':True,'all_archived_private_results_match_inputs':True,'inputs':{p.relative_to(SUITE).as_posix():sha(p) for p in (native/'native-results.json',native/'archive-members-sha256.json',v4/'paired-results.json',v4/'archive-members-sha256.json',v4/'archive-restore-index.json')}})
    write(output/'checksums.json', {p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(overall, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,default=SUITE/'results/native-agent-v2')
    p.add_argument('--v4',type=Path,default=SUITE/'results/training-rollout-v4-20261021')
    p.add_argument('--output',type=Path,default=SUITE/'results/native-versus-training-rollout-v4-20261022')
    a = p.parse_args()
    compare(a.native,a.v4,a.output)
