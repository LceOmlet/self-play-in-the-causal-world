"""Offline budget accounting from saved TRL text logs; never executes model text.

The Parquet export is decoded text, not an authenticated environment ledger.
Only unambiguous act/response pairs with a consistent cost chain are accepted.
Numerics output is never interpreted as a causal observation. Missing/truncated
or ambiguous traces are reported as such, not as zero budget consumption.
"""
import hashlib
import json
import re

RESPONSE = re.compile(r'\nuser\n<tool_response>\n(.*?)\n</tool_response>\nassistant\n', re.S)
BUDGET = re.compile(r'The total observation budget is (\d+) scalar values')
TERMINAL = 'Terminal answer accepted. Episode complete; make no more tool calls.'


def budget_record(prompt, completion):
    result = {'status': 'unavailable', 'source': 'decoded TRL completion log',
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    try:
        budgets = BUDGET.findall(prompt)
        if len(budgets) != 1 or int(budgets[0]) <= 0:
            raise ValueError('Missing or ambiguous initial budget')
        initial = remaining = int(budgets[0])
        queries = rejected = numerical = post_terminal = 0
        terminal = False
        end = 0
        for match in RESPONSE.finditer(completion):
            assistant = completion[end:match.start()]
            # Restrict accounting to a single complete native tool call immediately
            # preceding its response. This is not a tool dispatcher/parser replacement.
            if assistant.count('<tool_call>') != 1 or assistant.count('</tool_call>') != 1:
                raise ValueError('Ambiguous call/response association')
            call = assistant.split('<tool_call>', 1)[1]
            names = re.findall(r'<function=([^>]+)>', call)
            if len(names) != 1:
                raise ValueError('Ambiguous function')
            response = match.group(1)
            if terminal:
                if names[0] not in ('act', 'numerics') or response != 'Episode already complete. Make no more tool calls.':
                    raise ValueError('Unexpected feedback after terminal acknowledgement')
                post_terminal += 1
                end = match.end()
                continue
            if names[0] == 'numerics':
                numerical += 1
            elif names[0] == 'act':
                parameters = re.findall(r'<parameter=command>\s*(.*?)\s*</parameter>', call, re.S)
                if len(parameters) != 1:
                    raise ValueError('Ambiguous act command')
                command = json.loads(parameters[0])
                if response == TERMINAL:
                    if command.get('type') != 'answer':
                        raise ValueError('Terminal acknowledgement without answer command')
                    terminal = True
                else:
                    feedback = json.loads(response.split('\n', 1)[0])
                    now = feedback['remaining_budget']
                    if type(now) is not int or not 0 <= now <= remaining:
                        raise ValueError('Invalid remaining budget')
                    if feedback['type'] == 'protocol_error':
                        if feedback.get('accepted') is not False or feedback.get('budget_consumed') != 0 or now != remaining:
                            raise ValueError('Invalid rejected-command accounting')
                        rejected += 1
                    elif feedback['type'] == 'batch_result':
                        batch = feedback['batch']
                        n = batch['n']
                        hist = batch['joint_histogram']
                        cols = hist['columns']
                        if (command.get('type') not in ('observe', 'intervene') or
                                type(n) is not int or n <= 0 or n != command.get('batch_size') or
                                not cols or cols != command.get('measure') or len(set(cols)) != len(cols)):
                            raise ValueError('Command/batch mismatch')
                        if any(type(count) is not int or count < 0 or len(cells) != len(cols)
                               for cells, count in hist['rows']):
                            raise ValueError('Invalid histogram')
                        if sum(count for _, count in hist['rows']) != n or remaining - now != n * len(cols):
                            raise ValueError('Histogram/cost chain mismatch')
                        queries += 1
                    else:
                        raise ValueError('Unknown causal feedback')
                    remaining = now
            else:
                raise ValueError('Unknown function in saved log')
            end = match.end()
        # Terminal runs are complete. Other traces yield only observed consumption;
        # an executed final tool may be absent from a clipped exported completion.
        result.update(status='complete_logged' if terminal else 'partial_logged',
                      initial=initial, observed_consumed=initial-remaining,
                      observed_remaining=remaining, observed_fraction=(initial-remaining)/initial,
                      accepted_queries=queries, rejected_act_calls=rejected,
                      numerics_calls=numerical, post_terminal_rejected_calls=post_terminal,
                      terminal_acknowledged=terminal)
        if completion.count('\nuser\n<tool_response>\n') != len(list(RESPONSE.finditer(completion))):
            raise ValueError('Incomplete or ambiguous feedback frame')
    except (ValueError, TypeError, KeyError, IndexError) as error:
        result = {k: v for k, v in result.items() if k in ('source', 'prompt_sha256')}
        result.update(status='unavailable', reason=str(error))
    return result


def collect(parquet_dir, completed):
    import pandas as pd

    records, sources = [], []
    for path in sorted(parquet_dir.glob('completions_*.parquet')):
        step = int(path.stem.rsplit('_', 1)[1])
        if step > completed:
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        sources.append({'file': path.name, 'sha256': digest, 'size': len(raw)})
        frame = pd.read_parquet(path)
        if len(frame) != 4 or set(frame['step']) != {step}:
            raise ValueError(f'Expected four answers at step {step}: {path.name}')
        for answer_index, row in enumerate(frame.to_dict('records')):
            records.append({'step': step, 'answer_index': answer_index,
                            **budget_record(row['prompt'], row['completion'])})
    return {'records': records, 'sources': sources,
            'note': 'Decoded-log reconstruction, not authenticated owner telemetry. Partial traces are lower bounds; unavailable values are missing.'}
