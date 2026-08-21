from __future__ import annotations

import json
import unittest
from dataclasses import asdict

from cpt_world import (
    CandidateSeedSpec,
    check_seed_legality,
    cladder_ate_is_degenerate,
    load_candidate_seed_manifest,
    seed_triple_is_registered,
    validate_candidate_seed_manifest,
)


class CandidateRealSeedManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seeds = load_candidate_seed_manifest()

    def test_manifest_loads_and_passes_validation(self) -> None:
        self.assertEqual(len(self.seeds), 5)
        validate_candidate_seed_manifest(self.seeds)
        self.assertEqual(
            {seed.seed_id for seed in self.seeds},
            {
                "SEED-CL-CONF-ATE",
                "SEED-CL-DIAMONDCUT-BACKADJ",
                "SEED-BN-CANCER-BESTINT",
                "SEED-BN-ASIA-MEDIATOR",
                "SEED-BN-SURVEY-BESTINT",
            },
        )

    def test_visible_schema_is_anonymous(self) -> None:
        all_labels: set[str] = set()
        for seed in self.seeds:
            labels = seed.visible_labels()
            self.assertEqual(set(labels), set(seed.internal_variable_names()))
            for label in labels.values():
                self.assertEqual(len(label), 3)
                self.assertTrue(label.isalpha())
                self.assertNotIn(label, all_labels)
                all_labels.add(label)
            serialized = json.dumps(seed.visible_variables())
            for name in seed.internal_variable_names():
                if len(name) >= 3:
                    self.assertNotIn(name.lower(), serialized.lower())

    def test_every_seed_is_a_world_query_task_triple(self) -> None:
        expected = {
            "SEED-CL-CONF-ATE": ("confounding", "ate", "target_query"),
            "SEED-CL-DIAMONDCUT-BACKADJ": ("diamondcut", "backadj_minimal_sets", "discovery"),
            "SEED-BN-CANCER-BESTINT": ("bnlearn_bif", "best_intervention", "decision"),
            "SEED-BN-ASIA-MEDIATOR": ("bnlearn_bif", "mediator_set", "discovery"),
            "SEED-BN-SURVEY-BESTINT": ("bnlearn_bif", "best_intervention", "decision"),
        }
        for seed in self.seeds:
            self.assertEqual(seed.seed_triple(), expected[seed.seed_id])

    def test_registered_triples_pass_assembly_legality(self) -> None:
        for seed in self.seeds:
            world, query, task = seed.seed_triple()
            self.assertTrue(seed_triple_is_registered(world, query, task))
            self.assertEqual(check_seed_legality(asdict(seed)), [])

        expected = {
            "SEED-CL-CONF-ATE": ("confounding", "ate", "target_query"),
            "SEED-CL-DIAMONDCUT-BACKADJ": ("diamondcut", "backadj_minimal_sets", "discovery"),
            "SEED-BN-CANCER-BESTINT": ("bnlearn_bif", "best_intervention", "decision"),
            "SEED-BN-ASIA-MEDIATOR": ("bnlearn_bif", "mediator_set", "discovery"),
            "SEED-BN-SURVEY-BESTINT": ("bnlearn_bif", "best_intervention", "decision"),
        }
        for seed in self.seeds:
            self.assertEqual(seed.seed_triple(), expected[seed.seed_id])

    def test_candidate_seed_spec_type_guards(self) -> None:
        with self.assertRaises(ValueError):
            CandidateSeedSpec(
                seed_id="BAD",
                world_source={"type": "unknown"},
                query={"type": "unknown"},
                task_head={"head": "unknown"},
                manipulability={"X": True},
                readable={"X": True},
                visible_schema={"variable_labels": {"X": "ABC"}},
            )

    def test_confounding_ate_seed_is_not_do_obs_degenerate(self) -> None:
        seed = next(seed for seed in self.seeds if seed.seed_id == "SEED-CL-CONF-ATE")
        self.assertFalse(cladder_ate_is_degenerate(seed))

    def test_confounding_ate_seed_keeps_query_endpoints_readonly(self) -> None:
        seed = next(seed for seed in self.seeds if seed.seed_id == "SEED-CL-CONF-ATE")
        self.assertEqual(
            seed.manipulability,
            {"V1": True, "X": False, "Y": False},
        )

    def test_pinned_decisions_separate_probe_and_deployment_targets(self) -> None:
        for seed in self.seeds:
            if seed.query.get("type") != "best_intervention":
                continue
            decision_target = str(seed.query["decision_target"])
            self.assertFalse(seed.manipulability[decision_target])
            self.assertTrue(any(seed.manipulability.values()))


if __name__ == "__main__":
    unittest.main()
