"""Independent rational references for jointly attainable terminal endpoints."""

from __future__ import annotations

import unittest
from fractions import Fraction as F
from itertools import product

from cpt_world import counterfactual_solver as CANDIDATE
from cpt_world.query_truth import _reference_counterfactual_joint_bounds
from cpt_world.world_space import WorldSpec


def terminal_world(rows, second_stratum=None):
    """X -> M -> Y; optional unaffected S -> Y indexes disjoint contexts."""
    k, d = len(rows), len(rows[0])
    if second_stratum is None:
        domains = (2, k, d)
        parents = {0: (), 1: (0,), 2: (1,)}
        cpt = {0: ((F(1, 2),) * 2,), 1: ((F(1, k),) * k,) * 2, 2: tuple(rows)}
    else:
        domains = (2, 2, k, d)
        parents = {0: (), 1: (), 2: (0,), 3: (1, 2)}
        cpt = {
            0: ((F(1, 2),) * 2,),
            1: ((F(1, 2),) * 2,),
            2: ((F(1, k),) * k,) * 2,
            3: tuple(rows) + tuple(second_stratum),
        }
    return WorldSpec(
        family="test_dag",
        topology="rational-terminal-packing",
        variables=tuple(f"V{i}" for i in range(len(domains))),
        domains=domains,
        state_names=tuple(tuple(map(str, range(d))) for d in domains),
        edges=tuple((a, b) for b, values in parents.items() for a in values),
        parents=parents,
        cpt=cpt,
    )


def layered_world(same_event):
    """Two binary mediators cannot differ across arms with mass above 9/100."""
    row = (F(1, 4), F(3, 4)) if same_event else (F(1, 4), F(1, 2), F(1, 4))
    domains = (2, 2, 2, len(row))
    mediator = ((F(9, 10), F(1, 10)), (F(4, 5), F(1, 5)))
    return WorldSpec(
        family="test_dag",
        topology="rational-two-mediator-diagonal",
        variables=("X", "M1", "M2", "Y"),
        domains=domains,
        state_names=tuple(tuple(map(str, range(d))) for d in domains),
        edges=((0, 1), (1, 2), (2, 3)),
        parents={0: (), 1: (0,), 2: (1,), 3: (2,)},
        cpt={0: ((F(1, 2),) * 2,), 1: mediator, 2: mediator, 3: (row, row)},
    )


def predicate(world, events, endpoint):
    return CANDIDATE._terminal_event_endpoint_is_jointly_attainable(
        world,
        0,
        len(world.domains) - 1,
        events,
        endpoint,
        baseline_value=0,
        treatment_value=1,
    )


def terminal_factor(world, events, endpoint):
    # Inspect the real factor's public numeric callback without building SCIP.
    owner = object.__new__(CANDIDATE._SparseResponseModel)
    owner.world = world
    owner.treatment, owner.outcome = 0, len(world.domains) - 1
    owner.baseline_value, owner.treatment_value = 0, 1
    owner.outcome_state, owner.outcome_events = None, events
    owner.affected = tuple(CANDIDATE._descendants(world, 0))
    owner.terminal_event_endpoint = endpoint
    return owner._terminal_event_factor()


class TerminalPackingTests(unittest.TestCase):
    def test_same_event_small_and_complement_rational_response_vertices(self):
        sparse = {(1, 0, 0): F(1, 4), (0, 1, 0): F(1, 4), (0, 0, 1): F(1, 4), (0, 0, 0): F(1, 4)}
        for complement in (False, True):
            with self.subTest(complement=complement):
                weights = {
                    tuple(1 - v if complement else v for v in response): mass
                    for response, mass in sparse.items()
                }
                p = F(3, 4) if complement else F(1, 4)
                world = terminal_world(((1 - p, p),) * 3)
                self.assertTrue(predicate(world, ((1,), (1,)), "lower"))
                factor = terminal_factor(world, ((1,), (1,)), "lower")
                for i, j in product(range(3), repeat=2):
                    exact = sum(
                        mass
                        for response, mass in weights.items()
                        if response[i] == response[j] == 1
                    )
                    expected = p if i == j else max(F(), 2 * p - 1)
                    self.assertEqual(exact, expected)
                    self.assertEqual(factor.values((i, j)), float(exact))

    def test_disjoint_multivalued_sparse_rational_vertex_recovers_every_cpt_state(self):
        # Six rational atoms retain A, B and two distinct residual outcome states.
        weights = dict.fromkeys(
            [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)],
            F(1, 6),
        )
        row = (F(1, 6), F(1, 2), F(1, 6), F(1, 6))
        world = terminal_world((row,) * 3)
        for events in (((0,), (1,)), ((1,), (0,))):
            self.assertTrue(predicate(world, events, "upper"))
            factor = terminal_factor(world, events, "upper")
            for i, j in product(range(3), repeat=2):
                exact = sum(
                    mass
                    for response, mass in weights.items()
                    if response[i] in events[0] and response[j] in events[1]
                )
                self.assertEqual(exact, F() if i == j else F(1, 6))
                self.assertEqual(factor.values((i, j)), float(exact))
        for i, state in product(range(3), range(4)):
            self.assertEqual(
                sum(mass for response, mass in weights.items() if response[i] == state), row[state]
            )

    def test_three_context_fair_triangle_rejects_false_upper(self):
        world = terminal_world(((F(1, 3),) * 3,) * 3)
        self.assertFalse(predicate(world, ((0,), (1,)), "upper"))
        # Independently certify the sum of six cross-event masses is at most 4/3.
        # Pointwise Frechet would demand 2. The inequality holds for all 27 types.
        for response in product(range(3), repeat=3):
            count_a, count_b = response.count(0), response.count(1)
            self.assertLessEqual(F(count_a * count_b), F(2, 3) * (count_a + count_b))
        self.assertLess(F(4, 3), F(2))

    def test_shared_strata_must_pass_separately(self):
        rare = ((F(9, 10), F(1, 10)),) * 3
        frequent = ((F(1, 10), F(9, 10)),) * 3
        failing = ((F(1, 2), F(1, 2)),) * 3
        self.assertTrue(predicate(terminal_world(rare, frequent), ((1,), (1,)), "lower"))
        self.assertFalse(predicate(terminal_world(rare, failing), ((1,), (1,)), "lower"))

    def test_exact_threshold_and_overlapping_events(self):
        exact = ((F(2, 3), F(1, 3)),) * 3
        above = list(exact)
        epsilon = F(1, 10**12)
        above[0] = (F(2, 3) - epsilon, F(1, 3) + epsilon)
        self.assertTrue(predicate(terminal_world(exact), ((1,), (1,)), "lower"))
        self.assertFalse(predicate(terminal_world(above), ((1,), (1,)), "lower"))
        world = terminal_world(((F(1, 3),) * 3,) * 3)
        for endpoint in ("lower", "upper"):
            self.assertFalse(predicate(world, ((0, 1), (1, 2)), endpoint))

    def test_two_mediator_endpoints_match_independent_full_rational_vertex_oracle(self):
        for same, endpoint, expected in ((False, "upper", F(9, 400)), (True, "lower", F(91, 400))):
            with self.subTest(same_event=same, endpoint=endpoint):
                world = layered_world(same)
                events = ((0,), (0 if same else 1,))
                reference = _reference_counterfactual_joint_bounds(
                    world,
                    0,
                    3,
                    treatment_value=1,
                    baseline_value=0,
                    baseline_outcome_states=events[0],
                    treatment_outcome_states=events[1],
                )
                self.assertIsInstance(reference[0], F)
                self.assertEqual(reference[0 if endpoint == "lower" else 1], expected)
                self.assertTrue(predicate(world, events, endpoint))
                actual = CANDIDATE._two_mediator_joint_bounds(
                    world,
                    0,
                    3,
                    baseline_value=0,
                    treatment_value=1,
                    outcome_events=events,
                    time_limit_seconds=5.0,
                    endpoint_only=endpoint,
                )
                self.assertIsNotNone(actual)
                self.assertAlmostEqual(getattr(actual, endpoint), float(expected), places=10)


if __name__ == "__main__":
    unittest.main()
