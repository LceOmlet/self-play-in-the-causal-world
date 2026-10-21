"""CPU-only checks that the deployment gate rejects numeric, rank and speed failures."""
import copy
from pathlib import Path
import tempfile
import unittest
import vllm_speed_tolerance as lab


class GateTests(unittest.TestCase):
    def test_acceptance_and_independent_rejections(self):
        official = (lab.LAB/'official-utils.py').read_bytes()
        previous = lab.LAB
        with tempfile.TemporaryDirectory() as folder:
            lab.LAB = Path(folder)
            try:
                (lab.LAB/'official-utils.py').write_bytes(official)
                baseline = {'greedy':[[[1],'a',[{'1':-.1,'2':-2}]]],
                    'probes':[{'fixture':0,'offset':0,'target':1,
                               'prefix_sha256':'fixed','logprob':-.1}]}
                def fixtures(change=None):
                    for model in ('base','checkpoint2350'):
                        for mode in ('eager','graph'):
                            for repeat in range(2):
                                row = copy.deepcopy(baseline)
                                row.update(model=model,mode=mode)
                                row['timings'] = [{'shape':shape,'tokens_per_second':
                                    100 if mode == 'eager' else 120} for shape in ('mixed','clone')]
                                if mode == 'graph' and change:
                                    change(row)
                                lab.write(lab.LAB/f'{model}-{mode}-{repeat}.json',row)
                    lab.compare()
                    return lab.read(lab.LAB/'report.json')
                self.assertTrue(fixtures()['passed'])
                numeric = fixtures(lambda r:r['probes'][0].update(logprob=-.2))
                self.assertFalse(numeric['passed'])
                self.assertFalse(numeric['models']['base']['fixed_prefix_numeric_passed'])
                speed = fixtures(lambda r:r.update(timings=[{'shape':s,'tokens_per_second':104}
                    for s in ('mixed','clone')]))
                self.assertFalse(speed['passed'])
                rank = fixtures(lambda r:r.update(greedy=[[[9],'z',[{'9':-.1,'8':-2}]]]))
                self.assertFalse(rank['passed'])
                self.assertFalse(any(rank['models']['base']['official_rank_passes']))
            finally:
                lab.LAB = previous


if __name__ == '__main__':
    unittest.main()
