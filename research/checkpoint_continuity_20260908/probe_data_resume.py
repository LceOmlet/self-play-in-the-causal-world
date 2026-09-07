"""Execute the actual upstream data-resume method with the GPU RPC mocked.

The fixture contains only shuffled row indices and dataloader state. It is not
a model checkpoint or a training implementation. No reward/filter/loss is copied.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

VERL = Path('/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1')
CONFIG = Path('/home/chen/runs/inference-efficiency-20260908/configuration-v2/current.yaml')
DATA = Path('/home/chen/runs/training-submission-20260907/data/train.parquet')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(VERL))
    import torch
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.utils import create_rl_sampler

    original = OmegaConf.load(CONFIG)
    n = pq.ParquetFile(DATA).metadata.num_rows
    assert n == 500 and original.data.shuffle and original.data.seed == 42
    dataset = list(range(n))

    def loader():
        return StatefulDataLoader(dataset, batch_size=1, num_workers=0, drop_last=True,
                                  sampler=create_rl_sampler(original.data, dataset))

    def advance(iterator, current):
        try:
            return iterator, int(next(iterator).item())
        except StopIteration:
            iterator = iter(current)
            return iterator, int(next(iterator).item())

    results = []
    for name, updates, generated, should_match in (
        ('current_early_checkpoint_shape', 5, 5, True),
        ('filtered_updates_at_false_epoch_boundary', 500, 600, False),
        ('unfiltered_true_epoch_boundary', 500, 500, False),
    ):
        baseline = loader()
        iterator = iter(baseline)
        for _ in range(generated):
            iterator, _ = advance(iterator, baseline)
        state = copy.deepcopy(baseline.state_dict())
        folder = args.output / name / f'global_step_{updates}'
        folder.mkdir(parents=True)
        torch.save(state, folder / 'data.pt')
        expected = []
        for _ in range(8):
            iterator, index = advance(iterator, baseline)
            expected.append(index)

        trainer = object.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create({'trainer': {
            'resume_mode': 'resume_path', 'default_hdfs_dir': None,
            'default_local_dir': str(folder.parent), 'resume_from_path': str(folder),
            'del_local_ckpt_after_load': False,
        }})
        trainer.train_dataloader = loader()
        trainer.actor_rollout_wg = Mock()
        trainer.use_critic = False
        # This is the real production method, not a reproduced conditional.
        trainer._load_checkpoint()
        trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(
            str(folder / 'actor'), del_local_after_load=False)
        iterator = iter(trainer.train_dataloader)
        actual = []
        for _ in range(8):
            iterator, index = advance(iterator, trainer.train_dataloader)
            actual.append(index)
        assert (actual == expected) is should_match
        results.append({'case': name, 'updates': updates, 'generated_groups': generated,
                        'saved_groups_in_current_epoch': state['_num_yielded'],
                        'saved_iterator_finished': state['_iterator_finished'],
                        'expected_next_row_indices': expected, 'actual_next_row_indices': actual,
                        'matches_uninterrupted_stream': actual == expected})

    source = VERL / 'verl/trainer/ppo/ray_trainer.py'
    report = {
        'defect_reproduced': True, 'actual_upstream_method': 'RayPPOTrainer._load_checkpoint',
        'gpu_checkpoint_rpc_mocked': True, 'real_sampler_factory': 'verl.trainer.ppo.utils.create_rl_sampler',
        'dataset': {'path': str(DATA), 'sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(), 'rows': n},
        'source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
        'cases': results, 'cuda_initialized': torch.cuda.is_initialized(),
        'scope': 'CPU data-stream control probe with the actual upstream method and sampler. No model or optimizer resume is validated by this probe.',
    }
    assert not report['cuda_initialized']
    (args.output / 'resume-probe.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
