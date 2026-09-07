"""Prepare an isolated upstream source candidate; never edit the live runtime."""

import difflib
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = Path('/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1')
TARGET = Path('/home/chen/vendor/dapo-official-20260906/verl-data-resume-candidate-v1')
OUT = Path('/home/chen/runs/checkpoint-continuity-20260908')
RELATIVE = Path('verl/trainer/ppo/ray_trainer.py')


def files(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()
            and '.git' not in path.parts and '__pycache__' not in path.parts
            and path.suffix != '.pyc'}


def main():
    before = (SOURCE / RELATIVE).read_text()
    assert hashlib.sha256((SOURCE / RELATIVE).read_bytes()).hexdigest() == 'c930623489f724e9abbf1dab41ed617da1be2f0d60485d4c0ee5354c145dd809'
    start = before.index('            steps_per_epoch = len(self.train_dataloader)', before.index('    def _load_checkpoint('))
    end = before.index('\n        else:\n            print(f"Warning: No dataloader state found', start)
    replacement = '''            # Restore the sampler RNG and cursor even at an epoch boundary.
            # DAPO optimizer steps do not count generated batches after filtering.
            # StatefulDataLoader preserves the exhausted iterator's sampler state;
            # the next outer-loop iteration advances to the next shuffled epoch.
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)'''
    after = before[:start] + replacement + before[end:]
    original_files = files(SOURCE)
    shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    (TARGET / RELATIVE).write_text(after)
    copied_files = files(TARGET)
    assert set(copied_files) == set(original_files)
    changed = [name for name in original_files if original_files[name] != copied_files[name]]
    assert changed == [str(RELATIVE)]
    assert files(SOURCE) == original_files
    patch = ''.join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                       fromfile='a/' + str(RELATIVE), tofile='b/' + str(RELATIVE)))
    (OUT / 'data-resume-only.patch').write_text(patch)
    report = {'source': str(SOURCE), 'candidate': str(TARGET), 'files_checked': len(original_files),
              'changed_files': changed, 'source_unchanged': True,
              'before_sha256': original_files[str(RELATIVE)], 'after_sha256': copied_files[str(RELATIVE)],
              'patch_sha256': hashlib.sha256((OUT / 'data-resume-only.patch').read_bytes()).hexdigest(),
              'status': 'Isolated data-stream candidate only; DAPO epoch/gen_steps continuation and actual GPU resume remain unverified. Not admitted to the live pinned runtime.'}
    (OUT / 'candidate-source.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
