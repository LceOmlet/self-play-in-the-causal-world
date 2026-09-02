from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cpt_world.counterfactual_isolation import compute_counterfactual_truth_isolated
from cpt_world.query_truth import compute_query_truth
from cpt_world.world_space import WorldSpec


def _direct_counterfactual_fixture() -> tuple[WorldSpec, dict[str, object]]:
    world = WorldSpec(
        family="test",
        topology="A->B",
        variables=("A", "B"),
        domains=(2, 2),
        state_names=(("state_0", "state_1"), ("state_0", "state_1")),
        edges=((0, 1),),
        parents={0: (), 1: (0,)},
        cpt={
            0: ((0.5, 0.5),),
            1: ((0.8, 0.2), (0.2, 0.8)),
        },
    )
    seed: dict[str, object] = {
        "seed_id": "TEST-direct-counterfactual",
        "query": {
            "type": "individual_counterfactual_probability",
            "treatment": "A",
            "outcome": "B",
            "factual_value": 0,
            "counterfactual_value": 1,
            "factual_outcome_state": 0,
            "outcome_state": 1,
        },
    }
    return world, seed


class CounterfactualIsolationTests(unittest.TestCase):
    def test_isolated_owner_matches_direct_owner_exactly(self) -> None:
        world, seed = _direct_counterfactual_fixture()
        expected = compute_query_truth(
            world,
            seed,
            counterfactual_endpoint_time_limit_seconds=5.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            actual = compute_counterfactual_truth_isolated(
                world,
                seed,
                endpoint_time_limit_seconds=5.0,
                diagnostic_dir=Path(temporary),
            )
            self.assertEqual(actual, expected)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
