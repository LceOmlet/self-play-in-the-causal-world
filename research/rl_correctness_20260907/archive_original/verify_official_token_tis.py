"""Verify the selected upstream Token-TIS + DAPO loss and derivatives.

Pure CPU audit: no model, optimizer, replacement trainer, or training entry.
"""
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.workers.config import ActorConfig

ROOT = Path(__file__).resolve().parent
mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0],
                     [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1]], dtype=torch.float64)
token_count = mask.sum().item()
old = torch.full_like(mask, -4.0)
importance_ratios = torch.tensor([[0.2, 0.7, 1.0, 1.5, 2.5, 8.0]] * 4, dtype=torch.float64)
rollout = old - importance_ratios.log()
raw = torch.tensor([0.9, 0.2, 0.8, 0.4], dtype=torch.float64)
penalties = torch.tensor([0, 0, -0.1, -0.25], dtype=torch.float64)
shaped = raw + penalties
rewards = torch.zeros_like(mask)
for i in range(4):
    last = mask[i].nonzero()[-1].item()
    rewards[i, last] = shaped[i]
uids = np.array(['a', 'b', 'a', 'b'])
advantages, _ = compute_grpo_outcome_advantage(rewards, mask, uids)
expected_adv = torch.zeros_like(mask)
for indices in ([0, 2], [1, 3]):
    mean = shaped[indices].mean()
    # The selected official estimator uses the sample standard deviation.
    sigma = ((shaped[indices[0]]-shaped[indices[1]]).abs() / (2**0.5)).item()
    for index in indices:
        expected_adv[index] = mask[index] * (shaped[index] - mean) / (sigma + 1e-6)
torch.testing.assert_close(advantages, expected_adv, atol=1e-12, rtol=1e-12)
batch = DataProto.from_dict(tensors={'old_log_probs':old,'rollout_log_probs':rollout,
                                   'response_mask':mask.clone()})
config = OmegaConf.create({'rollout_is':'token', 'rollout_is_threshold':2.0,
                          'rollout_is_batch_normalize':False, 'rollout_rs':None,
                          'rollout_rs_threshold':None})
batch, metrics = compute_rollout_correction_and_add_to_batch(batch, config)
weights = batch.batch['rollout_is_weights']
torch.testing.assert_close(weights, importance_ratios.clamp(max=2) * mask, atol=1e-12, rtol=1e-12)
assert not weights.requires_grad
assert torch.equal(batch.batch['response_mask'], mask)
actor = ActorConfig(strategy='fsdp2', rollout_n=4, ppo_micro_batch_size_per_gpu=1,
                    clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=10.0,
                    loss_agg_mode='token-mean')
actor.global_batch_info.update(dp_size=1, batch_num_tokens=token_count, global_batch_size=4)
policy_ratios = torch.tensor([[0.6, 0.9, 1.1, 1.4, 3.0, 12.0]]*4, dtype=torch.float64)
expected_loss = 0.0
expected_gradient = torch.zeros_like(mask)
for row in range(4):
    for col in range(6):
        if mask[row, col] == 0:
            continue
        a, r, w = advantages[row, col].item(), policy_ratios[row, col].item(), weights[row, col].item()
        # Piecewise closed derivative, including the configured negative-advantage dual clip.
        chosen = min(r, 1.28) if a >= 0 else min(max(r, 0.8), 10.0)
        expected_loss -= a * chosen * w / token_count
        active = r < 1.28 if a >= 0 else 0.8 < r < 10.0
        if active:
            expected_gradient[row, col] = -a * r * w / token_count
records = []
for chunk in (1, 2, 4):
    current = (old + policy_ratios.log()).requires_grad_()
    loss = sum(compute_policy_loss_vanilla(
        old[i:i+chunk], current[i:i+chunk], advantages[i:i+chunk], mask[i:i+chunk],
        'token-mean', actor, rollout_is_weights=weights[i:i+chunk],
    )[0] for i in range(0, 4, chunk))
    gradient, = torch.autograd.grad(loss, current)
    torch.testing.assert_close(loss, torch.tensor(expected_loss, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
    records.append({'microbatch_rows':chunk, 'loss':loss.item(),
                    'max_gradient_error':(gradient-expected_gradient).abs().max().item()})
result = {'official_functions': {f.__name__:f.__code__.co_filename for f in
          (compute_grpo_outcome_advantage, compute_policy_loss_vanilla,
           compute_rollout_correction_and_add_to_batch)},
          'grouped_advantages_match':True, 'masked_tokens_have_zero_gradient':True,
          'importance_weights_match':True, 'weights_detached':True,
          'response_mask_unchanged_without_rs':True, 'valid_tokens':token_count,
          'effective_weight_sample_size':float(weights.sum()**2 / weights.square().sum()),
          'records':records, 'real_model_execution':False}
(ROOT/'official-token-tis-acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
