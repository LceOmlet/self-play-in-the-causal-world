"""Stage an isolated recipe copy; apply only the individually listed project files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
ORIGINAL = Path('/home/chen/vendor/dapo-official-20260906/verl-recipe')
CANDIDATE = Path('/home/chen/vendor/dapo-official-20260906/verl-recipe-mask-v1')
STAGE = ROOT/'mask_project_stage'
FILES = ['configs/verl/upstream_sources.json', 'scripts/run_official_dapo.sh',
         'scripts/verify_official_dapo.py', 'patches/verl-recipe-action-mask-v1.patch',
         'tests/test_official_dapo_action_mask.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage():
    manifest = json.loads((STAGE/FILES[0]).read_text())
    baseline = json.loads((PROJECT/FILES[0]).read_text())
    assert manifest['repositories'] == baseline['repositories']
    assert manifest['reviewed_runtime_patches'][:-1] == baseline['reviewed_runtime_patches']
    entry = manifest['reviewed_runtime_patches'][-1]
    assert entry['repository'] == 'verl-recipe'
    assert not CANDIDATE.exists()
    for relative, expected in manifest['repositories']['verl-recipe']['files'].items():
        assert sha(ORIGINAL/relative) == expected
        destination = CANDIDATE/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ORIGINAL/relative, destination)
    patched = ROOT/'recipe_mask_candidate/dapo/dapo_ray_trainer.py'
    assert sha(patched) == entry['files']['dapo/dapo_ray_trainer.py']['patched_sha256']
    shutil.copy2(patched, CANDIDATE/'dapo/dapo_ray_trainer.py')
    (ROOT/'recipe-mask-stage.json').write_text(json.dumps({'candidate': str(CANDIDATE), 'patch': entry}, indent=2))
    print('Staged isolated recipe with one mask guard; original recipe retained.')


def apply():
    assert (ROOT/'recipe-mask-tests-passed.json').exists()
    backup = ROOT/'evidence/before-recipe-mask'
    backup.mkdir(parents=True, exist_ok=False)
    records = []
    for relative in FILES:
        source, destination = STAGE/relative, PROJECT/relative
        old_hash = sha(destination) if destination.exists() else None
        if destination.exists():
            saved = backup/relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        records.append({'path': relative, 'before': old_hash, 'after': sha(destination)})
    (ROOT/'recipe-mask-integration.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'apply'])
    if parser.parse_args().action == 'stage':
        stage()
    else:
        apply()
