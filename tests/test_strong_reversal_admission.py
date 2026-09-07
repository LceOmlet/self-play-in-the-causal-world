"""Independent rational checks of the unchanged strong-reversal acceptance event."""

import math
import random
import unittest
from dataclasses import replace
from fractions import Fraction as F
from unittest.mock import patch

from cpt_world import WorldSpec, query_truth
from cpt_world import world_space as w


def world_from_rows(decision_rows, outcome_rows):
    size = len(decision_rows[0])
    return WorldSpec(
        family="sampled_dag",
        topology="admission-reference",
        variables=("U", "X", "Y"),
        domains=(2, size, 2),
        state_names=(("0", "1"), tuple(map(str, range(size))), ("0", "1")),
        edges=((0, 1), (0, 2), (1, 2)),
        parents={0: (), 1: (0,), 2: (0, 1)},
        cpt={
            0: ((F(1, 4), F(3, 4)),),
            1: tuple(decision_rows),
            2: tuple((1 - p, p) for p in outcome_rows),
        },
    )


def reference(world, outcome, objective):
    """Enumerate the three-node rational joint law, without production inference."""
    size = world.domains[1]
    causal, observed = [], []
    for action in range(size):
        causal.append(
            sum(world.cpt[0][0][u] * world.cpt[2][u * size + action][outcome] for u in range(2))
        )
        masses = [world.cpt[0][0][u] * world.cpt[1][u][action] for u in range(2)]
        observed.append(
            sum(masses[u] * world.cpt[2][u * size + action][outcome] for u in range(2))
            / sum(masses)
        )
    best = min if objective == "minimize" else max
    causal_best, observed_best = best(causal), best(observed)
    observed_actions = [a for a in range(size) if observed[a] == observed_best]
    discordant = all(causal[a] != causal_best for a in observed_actions)
    regret = abs(causal_best - best(causal[a] for a in observed_actions))
    return discordant, float(regret), tuple(causal)


class StrongReversalAdmissionTests(unittest.TestCase):
    def test_rational_enumeration_and_float_worlds_at_threshold_neighbors(self):
        rng = random.Random(70907)
        for size in (2, 3, 4, 5):
            for case in range(16):
                rows = []
                for _ in range(2):
                    cuts = [0, *sorted(rng.sample(range(1, 16), size - 1)), 16]
                    rows.append(
                        tuple(F(b - a, 16) for a, b in zip(cuts[:-1], cuts[1:], strict=True))
                    )
                world = world_from_rows(rows, [F(rng.randrange(9), 8) for _ in range(2 * size)])
                binary64_world = replace(
                    world,
                    cpt={
                        n: tuple(tuple(map(float, row)) for row in table)
                        for n, table in world.cpt.items()
                    },
                )
                for state in (0, 1):
                    for objective in ("minimize", "maximize"):
                        anchors = dict(
                            decision_target=1, outcome=2, outcome_state=state, objective=objective
                        )
                        discordant, gap, causal = reference(world, state, objective)
                        thresholds = {
                            0.0,
                            w.BEST_INTERVENTION_STRONG_REVERSAL_MIN_GAP,
                            gap,
                            math.nextafter(gap, math.inf),
                        }
                        if gap > 0:
                            thresholds.add(math.nextafter(gap, -math.inf))
                        for actual_world in (world, binary64_world):
                            self.assertEqual(
                                w._best_intervention_causal_values(actual_world, anchors), causal
                            )
                            self.assertEqual(
                                w._best_intervention_observational_relation(actual_world, anchors),
                                (discordant, gap),
                            )
                            for threshold in thresholds:
                                for desired in (False, True):
                                    with self.subTest(
                                        size=size,
                                        case=case,
                                        state=state,
                                        objective=objective,
                                        threshold=threshold,
                                        desired=desired,
                                        exact=actual_world is world,
                                    ):
                                        expected = (
                                            (discordant and gap >= threshold)
                                            if desired
                                            else not discordant
                                        )
                                        with patch.object(
                                            w,
                                            "BEST_INTERVENTION_STRONG_REVERSAL_MIN_GAP",
                                            threshold,
                                        ):
                                            self.assertEqual(
                                                w._best_intervention_proposal_is_admitted(
                                                    actual_world,
                                                    anchors,
                                                    desired_discordant=desired,
                                                ),
                                                expected,
                                            )

    def test_weak_discordance_still_reported_but_observation_not_needed_for_rejection(self):
        world = world_from_rows(
            ((F(7, 8), F(1, 8)), (F(1, 8), F(7, 8))), (F(3, 4), F(25, 32), F(1, 4), F(9, 32))
        )
        anchors = dict(decision_target=1, outcome=2, outcome_state=1, objective="maximize")
        self.assertEqual(
            w._best_intervention_observational_relation(world, anchors), (True, 1 / 32)
        )
        with patch(
            "cpt_world.query_truth.worldspec_projected_interventional_distribution",
            wraps=query_truth.worldspec_projected_interventional_distribution,
        ) as inference:
            self.assertFalse(
                w._best_intervention_proposal_is_admitted(world, anchors, desired_discordant=True)
            )
        self.assertEqual(len(inference.call_args_list), 2)
        self.assertTrue(all(call.args[1] for call in inference.call_args_list))

    def test_observational_tie_and_concordant_slots_are_not_range_rejected(self):
        world = world_from_rows(((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))), (F(1, 2),) * 4)
        anchors = dict(decision_target=1, outcome=2, outcome_state=1, objective="maximize")
        self.assertEqual(reference(world, 1, "maximize")[:2], (False, 0.0))
        with patch.object(w, "BEST_INTERVENTION_STRONG_REVERSAL_MIN_GAP", 1.0):
            self.assertTrue(
                w._best_intervention_proposal_is_admitted(world, anchors, desired_discordant=False)
            )


if __name__ == "__main__":
    unittest.main()
