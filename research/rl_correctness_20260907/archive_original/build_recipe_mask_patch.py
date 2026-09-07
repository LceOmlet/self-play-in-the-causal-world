"""Create the exact upstream-style mask guard and its explicit provenance."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent/'rl_algorithm_fix_20260906/project'
SOURCE = ROOT.parent/'official_dapo_20260906/upstream/verl-recipe/dapo/dapo_ray_trainer.py'
original = SOURCE.read_bytes()
assert hashlib.sha256(original).hexdigest() == '5d3e76c0cd5a9b9e01b1a796d36659e6ad3104739fe9bdc4ae4953603cf47a2c'
newline = b'\r\n' if b'\r\n' in original else b'\n'
before = b'        batch.batch["response_mask"] = compute_response_mask(batch)' + newline
after = b'        if "response_mask" not in batch.batch.keys():' + newline + b'            batch.batch["response_mask"] = compute_response_mask(batch)' + newline
assert original.count(before) == 1
patched = original.replace(before, after)
candidate = ROOT/'recipe_mask_candidate/dapo/dapo_ray_trainer.py'
candidate.parent.mkdir(parents=True, exist_ok=True)
candidate.write_bytes(patched)
diff = ''.join(difflib.unified_diff(original.decode().splitlines(True), patched.decode().splitlines(True),
                                  fromfile='a/dapo/dapo_ray_trainer.py', tofile='b/dapo/dapo_ray_trainer.py'))
patch = PROJECT/'patches/verl-recipe-action-mask-v1.patch'
patch.write_bytes(diff.encode())
manifest_path = PROJECT/'configs/verl/upstream_sources.json'
manifest = json.loads(manifest_path.read_text())
entry = {'repository': 'verl-recipe', 'base_commit': manifest['repositories']['verl-recipe']['commit'],
         'path': 'patches/verl-recipe-action-mask-v1.patch', 'sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
         'reason': 'Preserve the official tool action mask when recomputing old log probabilities, matching the pinned official RayPPOTrainer guard. No advantage, filtering, loss, importance-weight or optimizer formula changes.',
         'files': {'dapo/dapo_ray_trainer.py': {'upstream_sha256': hashlib.sha256(original).hexdigest(),
                                               'patched_sha256': hashlib.sha256(patched).hexdigest()}}}
assert not any(p['repository'] == 'verl-recipe' for p in manifest['reviewed_runtime_patches'])
manifest['reviewed_runtime_patches'].append(entry)
manifest['description'] = 'Pinned official DAPO sources with original hashes retained. Explicit tool-termination and recipe action-mask compatibility patches are separately fingerprinted. Algorithm formulas remain upstream-owned.'
manifest_path.write_bytes((json.dumps(manifest, indent=2)+'\n').encode())
(ROOT/'recipe-mask-patch-manifest.json').write_text(json.dumps(entry, indent=2))
print(json.dumps(entry, indent=2))
