"""Bundle only diagnostics; keep large raw model/gradient snapshots on server."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
selected = []
for pattern in ('*-acceptance.json', '*-proof.md', '*mask*.json', 'official-dapo-validation-matrix.md'):
    selected.extend(ROOT.glob(pattern))
execution = ROOT/'official_execution'
selected.extend(execution.glob('*.py'))
selected.extend(execution.glob('*.json'))
selected.extend((execution/'observer').glob('*.py'))
for run in sorted(execution.glob('run-*-audit-only')):
    selected.extend(run.glob('*.json'))
    selected.extend(run.glob('*.txt'))
    selected.extend(run.glob('*.log'))
    selected.extend((run/'environment').glob('*.jsonl'))
    selected.extend((run/'observation').glob('*/events.jsonl'))
    selected.extend((run/'observation').glob('installed-*.json'))
for preflight in sorted(execution.glob('preflight*')):
    selected.extend(preflight.glob('*.json'))
    selected.extend(preflight.glob('*.log'))
selected.extend((ROOT/'evidence/before-recipe-mask').rglob('*'))
files = sorted({path for path in selected if path.is_file()})
manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
(ROOT/'current-evidence-manifest.json').write_text(json.dumps(manifest, indent=2))
with tarfile.open(ROOT/'current-evidence.tar.gz', 'w:gz') as archive:
    for path in files:
        archive.add(path, arcname=str(path.relative_to(ROOT)))
    archive.add(ROOT/'current-evidence-manifest.json', arcname='current-evidence-manifest.json')
print(json.dumps({'files': len(files), 'archive_bytes': (ROOT/'current-evidence.tar.gz').stat().st_size}))
