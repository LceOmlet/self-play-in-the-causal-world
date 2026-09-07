"""Certify the real multi-turn mask overwrite and invalidate incomplete checks."""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent/'run-01-audit-only'
actor = torch.load(next((ROOT/'observation').glob('*/actor_batch-*.pt')), weights_only=True)['batch']
original = torch.load(next((ROOT/'observation').glob('*/pre_filter_batch-*.pt')), weights_only=True)['batch']
rows = []
for row, ids in enumerate(actor['tensors']['responses']):
    index = next(i for i, candidate in enumerate(original['tensors']['responses']) if torch.equal(ids, candidate))
    before = original['tensors']['response_mask'][index].bool()
    after = actor['tensors']['response_mask'][row].bool()
    attention = actor['tensors']['attention_mask'][row, -len(after):].bool()
    assert torch.equal(after, attention)
    tool = attention & ~before
    assert tool.sum() > 0 and torch.all(after[tool])
    rollout = actor['tensors']['rollout_log_probs'][row]
    old = actor['tensors']['old_log_probs'][row]
    assert torch.all(rollout[tool] == 0)
    rows.append({'row': row, 'actual_action_tokens': int(before.sum()), 'overwritten_mask_tokens': int(after.sum()),
                 'tool_tokens_wrongly_in_loss': int(tool.sum()),
                 'max_abs_logprob_delta_action_tokens': float((old-rollout)[before].abs().max()),
                 'max_abs_logprob_delta_tool_tokens': float((old-rollout)[tool].abs().max())})
previous = ROOT/'execution-acceptance.json'
if previous.exists():
    content = json.loads(previous.read_text())
    if content.get('passed'):
        content.update(passed=False, superseded=True,
                       reason='Incomplete checker compared trajectory to pre-filter mask but omitted propagation into actor batch. Not an accepted training run.')
        (ROOT/'incomplete-component-checks-v1.json').write_text(json.dumps(content, indent=2))
report = {'passed': False, 'reason': 'Official DAPO recipe unconditionally overwrites the tool action mask with attention mask.',
          'owner': 'dapo/dapo_ray_trainer.py:51 compute_kl_related_metrics',
          'rows': rows, 'checkpoints_disqualified': True,
          'actual_action_tokens': sum(r['actual_action_tokens'] for r in rows),
          'incorrect_denominator': sum(r['overwritten_mask_tokens'] for r in rows),
          'tool_tokens_wrongly_in_loss': sum(r['tool_tokens_wrongly_in_loss'] for r in rows)}
(ROOT/'mask-overwrite-proof.json').write_text(json.dumps(report, indent=2))
previous.write_text(json.dumps(report, indent=2))
(ROOT/'CHECKPOINTS_DISQUALIFIED.txt').write_text('Mask overwrite observed in real execution. Every checkpoint in this run is failure evidence only. Never initialize or resume training from it.\n')
print(json.dumps(report, indent=2))
