"""Small synthetic restore tests; run directly with Python, no 500-task data needed."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from capsule import HERE, FrozenEnvironment, canonical, restore_episode, sha256, world_from_data, world_to_data
from cpt_world import OutcomeTape, WorldSpec, WorldSpecEpisode
from cpt_world.world_space import assemble_seed
from verify_suite import protocol_module, replay_capsule, verify_capsule


def fixture(family="ate"):
    world = WorldSpec("synthetic", "four-node-chain", ("A", "Z", "C", "D"),
        (2, 2, 2, 2), (("0", "1"),) * 4, ((0, 1), (1, 2), (2, 3)),
        {0: (), 1: (0,), 2: (1,), 3: (2,)},
        {0: ((0.1, 0.9),), 1: ((0.2, 0.8), (0.7, 0.3)),
         2: ((0.4, 0.6), (0.3, 0.7)), 3: ((0.9, 0.1), (0.8, 0.2))})
    seed = assemble_seed(world, "mechanism_hidden", family,
                         "discovery" if family == "mediator_set" else "target_query",
                         seed_id="synthetic-restore", observation_bandwidth=2,
                         observation_budget_exponent=11)
    # This arbitrary cached vector is a serialization/scorer fixture, not a
    # mathematical assertion about the synthetic world's causal effect.
    truth = {"type": "ate", "effect": [0.0, 0.0]}
    if family == "mediator_set":
        truth = {"type": family, "mediators": ["Z", "C"], "order": [["Z", "C"]]}
    episode = WorldSpecEpisode(world, seed, OutcomeTape("synthetic-tape"), terminal_truth=truth)
    row = protocol_module(HERE).prepare_row({"task_id": "synthetic-000", "prompt": list(episode.initial_messages()),
        "sample_index": 0, "anchor_index": 0, "query_type": family, "tape_key": "synthetic-tape",
        "terminal_truth_json": json.dumps(truth)})
    capsule = {"format": "cpt-world-capsule-v1", "task_id": "synthetic-000", "family": family, "ordinal": 0,
        "world": world_to_data(world), "seed": seed, "tape_key": "synthetic-tape", "terminal_truth": truth,
        "max_graph_nodes": 16, "row": row}
    capsule = json.loads(canonical(capsule))
    capsule["components_sha256"] = {name: sha256(capsule[name])
        for name in ("world", "seed", "terminal_truth", "tape_key")}
    capsule["components_sha256"]["prompt"] = sha256(row["prompt"])
    return capsule


class RestoreTests(unittest.TestCase):
    def test_binary64_roundtrip(self):
        capsule = fixture()
        restored = world_from_data(json.loads(canonical(capsule["world"])))
        self.assertEqual(canonical(world_to_data(restored)), canonical(capsule["world"]))
        for node, table in restored.cpt.items():
            for row, original in zip(table, capsule["world"]["cpt"][str(node)]):
                self.assertEqual([p.hex() for p in row], [p.hex() for p in original])

    def test_no_sampler_and_prompt_integrity(self):
        capsule = fixture()
        with patch("cpt_world.trl_environment.assemble_sampled_anchor_tasks", side_effect=AssertionError("sampler forbidden")):
            env = FrozenEnvironment()
            self.assertEqual(env.reset(capsule=capsule, prompt=capsule["row"]["prompt"]), "")
            verify_capsule(capsule, protocol_module(HERE))
        with self.assertRaises(ValueError):
            env.reset(capsule=capsule, prompt=[])

    def test_mutable_state_isolated(self):
        capsule = fixture()
        a, b = restore_episode(capsule), restore_episode(capsule)
        a.seed["readable"]["A"] = False
        self.assertTrue(b.seed["readable"]["A"])
        self.assertTrue(capsule["seed"]["readable"]["A"])

    def test_replay_and_native_score(self):
        self.assertEqual(replay_capsule(fixture()), replay_capsule(fixture()))

    def test_native_mediator_container_restore(self):
        capsule = fixture("mediator_set")
        episode = restore_episode(capsule)
        self.assertIsInstance(episode.terminal_truth["order"][0], tuple)
        self.assertEqual(canonical(episode.terminal_truth), canonical(capsule["terminal_truth"]))
        self.assertEqual(replay_capsule(capsule), replay_capsule(capsule))

    def test_cold_process_replay(self):
        code = "from tests_restore import fixture; from verify_suite import replay_capsule; print(replay_capsule(fixture()))"
        command = [sys.executable, "-c", code]
        self.assertEqual(subprocess.check_output(command, cwd=HERE), subprocess.check_output(command, cwd=HERE))

    def test_invalid_world_and_hash(self):
        capsule = fixture()
        broken = copy.deepcopy(capsule)
        broken["world"]["cpt"]["0"][0][0] = 0.4
        with self.assertRaises(ValueError):
            restore_episode(broken)
        broken = copy.deepcopy(capsule)
        broken["components_sha256"]["prompt"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_capsule(broken, protocol_module(HERE))
        with self.assertRaises(ValueError):
            canonical({"bad": math.nan})


if __name__ == "__main__":
    unittest.main()
