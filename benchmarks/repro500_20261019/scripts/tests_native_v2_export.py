"""Completion and capability gates must fail before opening any private score."""
import tempfile
from pathlib import Path
import unittest

from compile_results import read, write, SUITE
from compile_native_v2 import export


class ExportGateTests(unittest.TestCase):
    def fixture(self, root, completed=100, terminal_files=False):
        runtime = root/'isolated-agent-v2-runtime'
        runtime.mkdir()
        write(runtime/'progress.json',{'agent':{'total':100,'completed':completed}})
        ids = read(SUITE/'data/eval100.json')['task_ids']
        for task in ids:
            folder = runtime/'agent'/task
            folder.mkdir(parents=True)
            if terminal_files:
                # Reading this invalid score would raise ValueError, not AssertionError.
                (folder/'private-result.json').write_text('UNREADABLE PRIVATE SCORE')
        return runtime, ids

    def test_incomplete_progress_blocks_unsealing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root,completed=99)
            with self.assertRaises(AssertionError):
                export(root,root/'output')
            self.assertFalse((root/'output').exists())

    def test_counter_alone_cannot_unseal_missing_terminals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            with self.assertRaises(AssertionError):
                export(root,root/'output')

    def test_cross_case_roles_block_before_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime, ids = self.fixture(root,terminal_files=True)
            write(runtime/'private-control.json',{'roles':[
                {'token':str(i),'kind':'agent','task_ids':ids} for i in range(100)]})
            with self.assertRaises(AssertionError):
                export(root,root/'output')


if __name__ == '__main__':
    unittest.main()
