"""Refresh audit fingerprints and package only reviewable repository additions."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'rl_algorithm_fix_20260906/project'
DEST = PROJECT / 'research/rl_correctness_20260907'
for run in (ROOT / 'official_execution').glob('run-*-audit-only'):
    target = DEST / 'official_execution' / run.name
    target.mkdir(exist_ok=True)
    for name in ('launch.json', 'official-source-verification.json'):
        if (run / name).is_file():
            shutil.copyfile(run / name, target / name)
for name in ('repository_maintenance.py', 'prepare_repository_archive.py', 'pack_repository_archive.py'):
    shutil.copyfile(ROOT / name, DEST / 'archive_original' / name)
manifest = {str(file.relative_to(DEST)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(DEST.rglob('*')) if file.is_file() and file.name != 'archive-manifest.json'}
(DEST / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
files = [PROJECT / name for name in ('.gitattributes', '.gitignore', 'README.md',
         'patches/README.md', 'docs/official-dapo-integration-20260906.md', 'docs/rl-correctness-goal-20260907.md')]
files += [file for file in DEST.rglob('*') if file.is_file()]
with tarfile.open(ROOT / 'repository-docs-and-research.tar.gz', 'w:gz') as archive:
    for file in files:
        archive.add(file, arcname=file.relative_to(PROJECT).as_posix())
print(json.dumps({'files': len(files), 'archive_bytes': (ROOT / 'repository-docs-and-research.tar.gz').stat().st_size}))
