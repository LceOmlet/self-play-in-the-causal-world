"""Preserve the current research in the existing project, without model artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'rl_algorithm_fix_20260906/project'
DEST = PROJECT / 'research/rl_correctness_20260907'
DEST.mkdir(parents=True, exist_ok=True)
with tarfile.open(ROOT / 'current-evidence.tar.gz') as archive:
    archive.extractall(ROOT, filter='data')

originals = DEST / 'archive_original'
originals.mkdir(exist_ok=True)
for file in sorted(ROOT.glob('*.py')):
    if file.name in {'prepare_repository_archive.py', 'repository_state.py'}:
        continue
    shutil.copyfile(file, originals / file.name)
for name in ('root_probability_slice.py', 'verify_root_probability_slice.py',
             'verify_et_v2_binary_density.py', 'verify_optimal_legal_experiment.py',
             'finite_budget_witness.py', 'verify_official_token_tis.py'):
    content = (ROOT / name).read_text(encoding='utf-8')
    if name in {'verify_root_probability_slice.py', 'verify_et_v2_binary_density.py'}:
        content = content.replace("ROOT.parent/'rl_algorithm_fix_20260906/project'", 'ROOT.parents[1]')
    if name == 'verify_optimal_legal_experiment.py':
        content = content.replace("Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')", 'ROOT.parents[1]')
        content = content.replace("Path('/home/chen/runs/environment-validation-20260907/prove_finite_budget_limit.py')", "ROOT/'finite_budget_witness.py'")
    if name == 'finite_budget_witness.py':
        content = content.replace('from cpt_world import', "import sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))\n\nfrom cpt_world import", 1)
        content = content.replace("Path('/home/chen/runs/environment-validation-20260907/finite-budget-proof.json')", "Path(__file__).resolve().parent/'finite-budget-proof.json'")
    (DEST / name).write_text(content, encoding='utf-8', newline='\n')

for pattern in ('*-proof.md', 'official-dapo-validation-matrix.md'):
    for file in ROOT.glob(pattern):
        shutil.copyfile(file, DEST / file.name)
results = DEST / 'results'
results.mkdir(exist_ok=True)
for pattern in ('*-acceptance.json', '*acceptance-summary.json', '*mask*.json', 'final-integration-manifest.json'):
    for file in ROOT.glob(pattern):
        shutil.copyfile(file, results / file.name)

execution = DEST / 'official_execution'
execution.mkdir(exist_ok=True)
for file in (ROOT / 'official_execution').glob('*.py'):
    shutil.copyfile(file, execution / file.name)
shutil.copytree(ROOT / 'official_execution/observer', execution / 'observer', dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ('data-acceptance.json',):
    shutil.copyfile(ROOT / 'official_execution' / name, execution / name)
for run in (ROOT / 'official_execution').glob('run-*-audit-only'):
    target = execution / run.name
    target.mkdir(exist_ok=True)
    for name in ('execution-acceptance.json', 'mask-overwrite-proof.json',
                 'incomplete-component-checks-v1.json', 'run-manifest.json',
                 'CHECKPOINTS_DISQUALIFIED.txt', 'exit.json'):
        if (run / name).is_file():
            shutil.copyfile(run / name, target / name)
for preflight in (ROOT / 'official_execution').glob('preflight*'):
    if not preflight.is_dir():
        continue
    target = execution / preflight.name
    target.mkdir(exist_ok=True)
    for name in ('acceptance.json', 'source.json', 'checks.log'):
        if (preflight / name).is_file():
            shutil.copyfile(preflight / name, target / name)

# The standalone plot retains the previous visualization script and its style.
plot_source = ROOT.parent / 'rl_server_20260906_1725/plot_training_curves.py'
shutil.copyfile(plot_source, execution / 'original_plot_training_curves.py')
plot = (execution / 'plot_verified_update.py').read_text(encoding='utf-8')
plot = plot.replace("HERE.parents[1]/'rl_server_20260906_1725/plot_training_curves.py'", "HERE/'original_plot_training_curves.py'")
(execution / 'plot_verified_update.py').write_text(plot, encoding='utf-8', newline='\n')

manifest = {str(file.relative_to(DEST)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(DEST.rglob('*')) if file.is_file() and file.name != 'archive-manifest.json'}
(DEST / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
print(json.dumps({'files': len(manifest), 'bytes': sum(file.stat().st_size for file in DEST.rglob('*') if file.is_file())}))
