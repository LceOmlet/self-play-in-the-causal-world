"""Import the final lightweight server evidence and verify every archive byte."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
with tarfile.open(ROOT / 'current-evidence.tar.gz') as archive:
    archive.extractall(ROOT, filter='data')
manifest = json.loads((ROOT / 'current-evidence-manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
project = ROOT.parents[1] / 'self-play-in-the-causal-world'
execution = project / 'research/rl_correctness_20260907/official_execution'
shutil.copyfile(ROOT / 'official_execution/analyze_execution.py', execution / 'analyze_execution.py')
run = 'run-02-mask-v1-audit-only'
for name in ('execution-acceptance.json', 'exit.json'):
    shutil.copyfile(ROOT / 'official_execution' / run / name, execution / run / name)
assert json.loads((execution / run / 'execution-acceptance.json').read_text())['passed']
print(json.dumps({'passed': True, 'verified_evidence_files': len(manifest)}))
