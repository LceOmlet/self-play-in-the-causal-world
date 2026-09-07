"""Validate provenance against final files and make the exact integration diff."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'rl_algorithm_fix_20260906/project'
EVIDENCE = ROOT / 'evidence'
provenance = json.loads((EVIDENCE / 'official-source-verification-final.json').read_text())
module = 'src/cpt_world/counterfactual_solver.py'
assert provenance['project_sources'][module] == hashlib.sha256((PROJECT / module).read_bytes()).hexdigest()
assert sum(r['verified_files'] for r in provenance['repositories'].values()) == 427
manifest = json.loads((EVIDENCE / 'diagonal-integration-manifest.json').read_text())
diffs = []
for relative, record in manifest['files'].items():
    final = PROJECT / relative
    previous = EVIDENCE / 'before-diagonal' / relative
    old = previous.read_text(encoding='utf-8').splitlines(keepends=True) if previous.exists() else []
    new = final.read_text(encoding='utf-8').splitlines(keepends=True)
    diffs.extend(difflib.unified_diff(old, new, fromfile=f'a/{relative}' if old else '/dev/null',
                                     tofile=f'b/{relative}'))
    record['final_after_lint'] = hashlib.sha256(final.read_bytes()).hexdigest()
    target = ROOT / 'production_snapshot' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(final, target)
(ROOT / 'diagonal-kernel.patch').write_text(''.join(diffs), encoding='utf-8')
(ROOT / 'final-integration-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
paths = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.md', '.json', '.py', '.patch')
         and p.name != 'artifact-manifest.json']
paths += [p for directory in ('production_snapshot', 'evidence') for p in (ROOT / directory).rglob('*') if p.is_file()]
artifacts = {'goal_status':'active', 'integrated_candidate':'diagonal_candidate',
             'unmerged_candidate':'candidate', 'production_training_started':False,
             'files':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}
(ROOT / 'artifact-manifest.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
print(json.dumps({'official_sources_verified':427, 'cf_sha256':provenance['project_sources'][module],
                  'evidence_files':len(paths), 'goal_status':'active'}, indent=2))
