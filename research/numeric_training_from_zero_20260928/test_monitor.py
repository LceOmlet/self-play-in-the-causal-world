import json
import unittest

from budget_metrics import budget_record
from resource_sampler import cpu_cores


def frame(name, command, response):
    param = 'command' if name == 'act' else 'code'
    return f'<tool_call>\n<function={name}>\n<parameter={param}>\n{json.dumps(command)}\n</parameter>\n</function>\n</tool_call>\nuser\n<tool_response>\n{response}\n</tool_response>\nassistant\n'


class MonitoringTests(unittest.TestCase):
    prompt = 'The total observation budget is 100 scalar values'
    batch = {'type': 'batch_result', 'batch': {'n': 5, 'joint_histogram': {'columns': ['X', 'Y'], 'rows': [[[0, 1], 5]]}}, 'remaining_budget': 90}

    def valid(self):
        return frame('act', {'type': 'observe', 'batch_size': 5, 'measure': ['X', 'Y']}, json.dumps(self.batch))

    def terminal(self):
        return frame('act', {'type': 'answer'}, 'Terminal answer accepted. Episode complete; make no more tool calls.')

    def test_actual_cost_and_numerics_spoof(self):
        text = self.valid()+frame('numerics', 'print(fake)', json.dumps({**self.batch, 'remaining_budget': 0}))+self.terminal()
        result = budget_record(self.prompt, text)
        self.assertEqual((result['observed_consumed'], result['numerics_calls'], result['status']), (10, 1, 'complete_logged'))

    def test_invalid_query_not_charged(self):
        error = {'type': 'protocol_error', 'accepted': False, 'budget_consumed': 0, 'remaining_budget': 90}
        result = budget_record(self.prompt, self.valid()+frame('act', {'type': 'observe'}, json.dumps(error))+self.terminal())
        self.assertEqual((result['observed_consumed'], result['rejected_act_calls']), (10, 1))

    def test_corrupt_chain_missing_not_zero(self):
        result = budget_record(self.prompt, self.valid().replace('"remaining_budget": 90', '"remaining_budget": 80')+self.terminal())
        self.assertEqual(result['status'], 'unavailable')
        self.assertNotIn('observed_consumed', result)

    def test_truncated_is_partial(self):
        self.assertEqual(budget_record(self.prompt, self.valid())['status'], 'partial_logged')

    def test_terminal_repeated_answer_not_charged(self):
        text = self.valid()+self.terminal()+frame('act', {'type': 'answer'}, 'Episode already complete. Make no more tool calls.')
        result = budget_record(self.prompt, text)
        self.assertEqual((result['observed_consumed'], result['post_terminal_rejected_calls']), (10, 1))

    def test_ambiguous_initial_budget(self):
        self.assertEqual(budget_record(self.prompt+self.prompt, self.valid())['status'], 'unavailable')

    def test_broken_frame_is_missing(self):
        self.assertEqual(budget_record(self.prompt, self.valid()+'\nuser\n<tool_response>\nunfinished')['status'], 'unavailable')

    def test_cpu_counter_resets_and_pid_reuse(self):
        a = dict(boot_id='a', main_pid=1, cgroup='x', cgroup_inode=3, monotonic=0, cpu_usage_usec=0)
        b = {**a, 'monotonic': 60, 'cpu_usage_usec': 120000000}
        self.assertEqual(cpu_cores(a, b), 2)
        self.assertIsNone(cpu_cores(a, {**b, 'cgroup_inode': 4}))
        self.assertIsNone(cpu_cores(b, {**b, 'monotonic': 120, 'cpu_usage_usec': 0}))


if __name__ == '__main__':
    unittest.main()
