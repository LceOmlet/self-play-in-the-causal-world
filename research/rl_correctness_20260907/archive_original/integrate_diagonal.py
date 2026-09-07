"""Install only the reviewed exact reduction, with production hash precondition."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/home/chen/runs/rl-correctness-goal-20260907')
PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
expected = 'c7a4100f07ab0cc4a2f40a01bfafeef6c2ac200a8b762c6756345b417a34b2f2'
module = PROJECT / 'src/cpt_world/counterfactual_solver.py'
assert hashlib.sha256(module.read_bytes()).hexdigest() == expected, 'Production changed since audit'
candidate = ROOT / 'diagonal_candidate/src/cpt_world/counterfactual_solver.py'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == '4249c54301fe001ecb109749a41f293ff514f04b9e004af8b174e9c7b5e263b3'
files = {
    'src/cpt_world/counterfactual_solver.py': candidate,
    'tests/test_counterfactual_solver.py': ROOT / 'test_counterfactual_solver_diagonal.py',
    'tests/test_indirect_diagonal_transport.py': ROOT / 'test_indirect_diagonal_transport.py',
}
manifest = {'utc':datetime.now(timezone.utc).isoformat(), 'production_training_started':False,
            'official_algorithm_changed':False, 'files':{}}
for relative, source in files.items():
    destination = PROJECT / relative
    before = None
    if destination.exists():
        before = hashlib.sha256(destination.read_bytes()).hexdigest()
        backup = ROOT / 'before-diagonal' / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        assert not backup.exists(), 'Do not overwrite original integration backup'
        shutil.copyfile(destination, backup)
    shutil.copyfile(source, destination)
    manifest['files'][relative] = {'before':before, 'after':hashlib.sha256(destination.read_bytes()).hexdigest()}
(ROOT / 'diagonal-integration-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
