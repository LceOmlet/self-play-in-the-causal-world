"""Summarize captured real trajectories while the single update is pending."""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent/'run-01-audit-only'
records = []
for path in (ROOT/'observation').glob('*/trajectory-*.pt'):
    item = torch.load(path, weights_only=True)
    output = item['output']
    request = item['request_id']
    matching = []
    for transition_path in (ROOT/'observation').glob('*/tool_transition-*.pt'):
        transition = torch.load(transition_path, weights_only=True)
        if transition['request_id'] == request and transition['state'] == 'AgentState.TERMINATED':
            matching.append(transition)
    if output['extra_fields'].get('cpt_world', {}).get('completed'):
        assert len(matching) == 1
        terminal = matching[0]
        generation = next(torch.load(p, weights_only=True) for p in (ROOT/'observation').glob('*/generation-*.pt')
                          if (lambda g: g['request_id']==request and g['assistant_turns']==terminal['assistant_turns'])(torch.load(p, weights_only=True)))
        assert terminal['prompt_ids'] == generation['prompt_ids']
        assert terminal['response_mask'] == generation['response_mask']
        assert output['response_ids'] == terminal['prompt_ids'][-len(output['response_ids']):]
        assert output['response_mask'][-1] == 1
    records.append({'request': request, 'tokens': len(output['response_ids']),
                    'action_tokens': sum(output['response_mask']), 'num_turns': output['num_turns'],
                    'environment': output['extra_fields'].get('cpt_world'), 'terminal_no_feedback_check': bool(matching)})
print(json.dumps(records, indent=2))
