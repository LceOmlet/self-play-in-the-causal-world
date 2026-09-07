"""Independent offline checks against captured actual official execution.

This script cannot create rollouts or training updates. It reads audit files.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE/'run-02-mask-v1-audit-only'


def read(kind):
    paths = sorted((ROOT/'observation').glob(f'*/{kind}-*.pt'))
    return [torch.load(path, map_location='cpu', weights_only=True) for path in paths]


def close(left, right, atol=1e-6, rtol=1e-5):
    torch.testing.assert_close(torch.as_tensor(left).double(), torch.as_tensor(right).double(), atol=atol, rtol=rtol)


assert json.loads((ROOT/'exit.json').read_text())['exit_code'] == 0
assert not list((ROOT/'observation').glob('observer-errors-*'))
actors, before, after, done = (read(kind) for kind in ('actor_batch', 'adam_call', 'adam_return', 'optimizer_done'))
assert len(actors) == len(before) == len(after) == len(done) == 1
actor = actors[0]
batch, config = actor['batch'], actor['config']
tensors, non_tensors = batch['tensors'], batch['non_tensors']
assert config['trainer']['total_training_steps'] == 1
assert config['trainer']['resume_mode'] == 'disable'
assert config['actor_rollout_ref']['model']['lora_adapter_path'] is None
assert config['algorithm']['adv_estimator'] == 'grpo'
assert config['algorithm']['norm_adv_by_std_in_grpo']
assert config['algorithm']['rollout_correction']['rollout_is'] == 'token'
assert config['algorithm']['rollout_correction']['rollout_is_threshold'] == 2
mask = tensors['response_mask'].bool()
attention = tensors['attention_mask'][:, -mask.shape[1]:].bool()
assert not torch.any(mask & ~attention)
raw = torch.as_tensor(non_tensors['acc']).double()
response_lengths = attention.sum(-1)
quality_shaping = -(response_lengths.double() - (30720 - 4096)).clamp(min=0) / 4096
close(tensors['token_level_scores'].sum(-1), raw + quality_shaping)
close(tensors['token_level_scores'], tensors['token_level_rewards'], atol=0, rtol=0)
assert (tensors['token_level_scores'] != 0).sum(-1).max().item() <= 1
uids = non_tensors['uid']
groups = defaultdict(list)
for row, uid in enumerate(uids):
    groups[uid].append(row)
reward = tensors['token_level_rewards'].sum(-1).double()
expected_adv = torch.zeros_like(tensors['advantages'], dtype=torch.float64)
for rows in groups.values():
    assert len(rows) == 4
    assert raw[rows].std() > 0
    scores = reward[rows]
    for row in rows:
        expected_adv[row] = (reward[row]-scores.mean())/(scores.std()+1e-6) * mask[row]
    extras = [non_tensors['extra_info'][row] for row in rows]
    assert len({extra['tape_key'] for extra in extras}) == 1
close(tensors['advantages'], expected_adv, atol=2e-6)
old, rollout = tensors['old_log_probs'], tensors['rollout_log_probs']
expected_weights = torch.exp(torch.clamp(old.double()-rollout.double(), -20, 20)).clamp(max=2)*mask
close(tensors['rollout_is_weights'], expected_weights, atol=2e-6)
# Snapshot copies are detached by the observer. Detachment in the live graph
# is established by the pinned upstream call and the separate function audit.

pre_filter = read('pre_filter_batch')
# A tensor can match every local formula yet be the wrong action mask. Follow
# each exact response from rollout ownership into the actor batch explicitly.
for row, ids in enumerate(tensors['responses']):
    matches = [(entry['batch']['tensors'], i) for entry in pre_filter
               for i, candidate in enumerate(entry['batch']['tensors']['responses'])
               if torch.equal(ids, candidate)]
    assert len(matches) == 1
    source, index = matches[0]
    assert torch.equal(tensors['response_mask'][row], source['response_mask'][index]), 'Action mask changed between rollout and actor'
all_groups = {}
for entry in pre_filter:
    b = entry['batch']
    uid_groups = defaultdict(list)
    for row, uid in enumerate(b['non_tensors']['uid']):
        uid_groups[uid].append(row)
    for uid, rows in uid_groups.items():
        scores = torch.tensor([b['non_tensors']['acc'][row] for row in rows])
        all_groups[uid] = {'eligible': bool(scores.std() > 0), 'rows': len(rows),
                           'response_tokens': int(b['tensors']['attention_mask'][rows, -mask.shape[1]:].sum())}
assert all(all_groups[uid]['eligible'] for uid in groups)
assert all(uid in groups for uid, item in all_groups.items() if item['eligible'])

losses = read('policy_loss')
assert len(losses) == 4
loss_records = []
loss_actor_rows = []
for entry in losses:
    old_log, log, advantage = [entry[key].double() for key in ('old_log_prob', 'log_prob', 'advantages')]
    selected = entry['response_mask'].bool()
    weight = entry['rollout_is_weights'].double()
    assert old_log.shape[1] == old.shape[1]
    matched_rows = []
    for micro_row in range(old_log.shape[0]):
        candidates = [row for row in range(old.shape[0])
                      if torch.equal(old_log[micro_row], old[row].double())]
        assert len(candidates) == 1, 'Loss row must map uniquely to actual actor data'
        row, = candidates
        assert torch.equal(selected[micro_row], mask[row]), 'Action mask changed before loss'
        close(advantage[micro_row], tensors['advantages'][row], atol=0, rtol=0)
        close(weight[micro_row], tensors['rollout_is_weights'][row], atol=0, rtol=0)
        matched_rows.append(row)
    loss_actor_rows.extend(matched_rows)
    ratio = (log-old_log).clamp(-20, 20).exp()
    chosen = torch.where(advantage >= 0, ratio.clamp(max=1.28), ratio.clamp(min=.8, max=10))
    info = entry['config']['global_batch_info']
    denominator = float(info['batch_num_tokens'])
    assert denominator == mask.sum().item()
    expected_loss = (-advantage * chosen * weight * selected).sum() / denominator * info['dp_size']
    close(entry['result'][0], expected_loss, atol=3e-5, rtol=3e-4)
    assert entry['loss_agg_mode'] == 'token-mean'
    loss_records.append({'actual': float(entry['result'][0]), 'reference': float(expected_loss),
                         'actor_rows': matched_rows,
                         'valid_tokens': int(selected.sum()), 'global_tokens': denominator,
                         'max_current_old_difference': float((log-old_log)[selected].abs().max())})
assert sum(item['valid_tokens'] for item in loss_records) == mask.sum().item()
assert sorted(loss_actor_rows) == list(range(old.shape[0])), 'Every actor row must enter loss exactly once'

pre, post = before[0], after[0]
assert pre['optimizer_type'] == post['optimizer_type'] == 'torch.optim.adamw.AdamW'
assert pre['state'] == {}
assert set(pre['trainable']) == set(post['trainable'])
assert set(post['state']) == set(pre['trainable'])
assert all(not item['has_grad'] for item in pre['frozen'] + post['frozen'])
pre_clip, = read('pre_clip')
assert not pre_clip['has_scaler']
clip = min(1., float(pre_clip['optimizer_config']['clip_grad']) / (float(done[0]['grad_norm'])+1e-6))
parameter_groups = {name: group for group in pre['param_groups'] for name in group['params']}
max_adam_rounding_units = 0.
changed, initial_nonzero_b, finite_gradient = [], [], []
for name, item in pre['trainable'].items():
    assert 'lora_' in name
    value, gradient = item['value'], item['grad']
    if 'lora_B' in name and value.count_nonzero():
        initial_nonzero_b.append(name)
    assert gradient is not None and torch.isfinite(gradient).all()
    if 'lora_A' in name:
        assert gradient.count_nonzero() == 0
    before_clip = pre_clip['trainable'][name]['grad']
    eps = torch.finfo(gradient.dtype).eps
    expected_clip = before_clip.double()*clip
    clip_error = (gradient.double()-expected_clip).abs()
    assert torch.all(clip_error <= expected_clip.abs()*eps*4 + torch.finfo(gradient.dtype).tiny)
    finite_gradient.append(name)
    if not torch.equal(value, post['trainable'][name]['value']):
        changed.append(name)
    state = post['state'][name]
    assert float(state['step']) == 1
    group = parameter_groups[name]
    assert not group['amsgrad'] and not group['maximize']
    beta1, beta2 = group['betas']
    mean, variance = state['exp_avg'].double(), state['exp_avg_sq'].double()
    expected_mean = gradient.double()*(1-beta1)
    expected_variance = gradient.double().square()*(1-beta2)
    for actual, expected in ((mean, expected_mean), (variance, expected_variance)):
        assert torch.all((actual-expected).abs() <= expected.abs()*eps*4 + torch.finfo(gradient.dtype).tiny)
    update = float(group['lr'])*(mean/(1-beta1))/(torch.sqrt(variance/(1-beta2))+float(group['eps']))
    decayed = value.double()*(1-float(group['lr'])*float(group['weight_decay']))
    expected_parameter = decayed-update
    actual_parameter = post['trainable'][name]['value'].double()
    scale = eps*(decayed.abs()+update.abs()) + torch.finfo(value.dtype).tiny
    units = ((actual_parameter-expected_parameter).abs()/scale).max().item()
    max_adam_rounding_units = max(max_adam_rounding_units, units)
    # A first-step identity, evaluated in float64 from actual moments. Eight
    # dtype rounding units cover intermediate sqrt/div/add/mul and final store.
    assert units <= 8, (name, units)
assert not initial_nonzero_b
assert changed
assert torch.isfinite(torch.tensor(done[0]['grad_norm']))

trajectories = read('trajectory')
generations = read('generation')
transitions = read('tool_transition')
trajectory_records = []
for trajectory in trajectories:
    request = trajectory['request_id']
    output = trajectory['output']
    ids, tmask, logs = output['response_ids'], output['response_mask'], output['response_logprobs']
    assert len(ids) == len(tmask) == len(logs)
    gs = sorted([g for g in generations if g['request_id'] == request], key=lambda g: g['assistant_turns'])
    ts = sorted([t for t in transitions if t['request_id'] == request], key=lambda t: t['assistant_turns'])
    for step in gs:
        generated_ids, generated_logs = step['output']['token_ids'], step['output']['log_probs']
        length = len(generated_ids)
        assert step['prompt_ids'][-length:] == generated_ids
        assert step['response_mask'][-length:] == [1]*length
        close(step['response_logprobs'][-length:], generated_logs, atol=0, rtol=0)
    terminals = [step for step in ts if any(response[0]['terminate'] for response in step['responses'])]
    completed = output['extra_fields'].get('cpt_world', {}).get('completed', False)
    if completed:
        assert len(terminals) == 1
        terminal = terminals[0]
        last_generation = gs[-1]
        assert terminal['state'] == 'AgentState.TERMINATED'
        assert terminal['assistant_turns'] == last_generation['assistant_turns']
        assert terminal['prompt_ids'] == last_generation['prompt_ids']
        assert terminal['response_mask'] == last_generation['response_mask']
        assert ids == terminal['prompt_ids'][-len(ids):]
        assert tmask[-1] == 1
    assert all(reward == 0 for reward in output['extra_fields']['tool_rewards'])
    # Locate exact response in the pre-filter batch, which also includes dropped groups.
    matches = []
    for entry in pre_filter:
        b = entry['batch']['tensors']
        for row in range(b['responses'].shape[0]):
            if b['responses'][row, :len(ids)].tolist() == ids:
                matches.append((b, row, entry['batch']['non_tensors']))
    assert len(matches) == 1
    b, row, extras = matches[0]
    assert b['response_mask'][row, :len(ids)].tolist() == tmask
    close(b['rollout_log_probs'][row, :len(ids)], logs, atol=1e-6)
    assert b['attention_mask'][row, -b['responses'].shape[1]:].sum() == len(ids)
    environment = output['extra_fields'].get('cpt_world', {})
    close(extras['acc'][row], environment['raw_reward'], atol=1e-7, rtol=1e-7)
    trajectory_records.append({'request_id': request, 'response_tokens': len(ids),
                               'action_tokens': sum(tmask), 'tool_tokens': len(ids)-sum(tmask),
                               'completed': completed, 'assistant_turns': len(gs),
                               'environment': output['extra_fields'].get('cpt_world')})

weights = tensors['rollout_is_weights'][mask].double()
result = {'passed': True, 'official_updates': 1, 'trainable_tensors': len(pre['trainable']),
          'changed_tensors': len(changed), 'changed_name_groups': dict(Counter('B' if 'lora_B' in name else 'A' for name in changed)),
          'frozen_tensors_without_grad': len(pre['frozen']), 'initial_lora_B_all_zero': True,
          'grad_norm_before_clip': done[0]['grad_norm'], 'groups_generated': len(all_groups),
          'max_first_step_adam_rounding_units': max_adam_rounding_units,
          'groups_discarded': sum(not item['eligible'] for item in all_groups.values()),
          'total_action_tokens': int(mask.sum()), 'raw_quality': raw.tolist(),
          'overlong_shaping': quality_shaping.tolist(), 'losses': loss_records,
          'tis': {'mean': float(weights.mean()), 'clipped_fraction': float((weights==2).double().mean()),
                  'effective_sample_size': float(weights.sum().square()/weights.square().sum()),
                  'max_abs_logprob_difference': float((old-rollout)[mask].abs().max())},
          'trajectories': trajectory_records,
          'limitations': ['One actual update, not evidence of capability improvement.',
                          'Frozen parameters checked by requires_grad/absent gradients and optimizer semantics; no full 9B before/after hash.',
                          'Model gradients captured for finite updates; no independent full-model backward implementation.']}
(ROOT/'execution-acceptance.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'trajectories'}, indent=2))
