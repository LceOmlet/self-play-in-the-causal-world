"""Independent rational coefficients and transport endpoints for numeric elimination."""

import unittest
from contextlib import nullcontext
from fractions import Fraction as F
from itertools import product
from unittest.mock import patch

from cpt_world.counterfactual_solver import _SparseResponseModel
from cpt_world.world_space import WorldSpec


def world_fixture(z_domain, zero_mass):
    z_rows = (
        ((F(4, 5), F(1, 5)), (F(3, 10), F(7, 10)))
        if z_domain == 2
        else ((F(1, 2), F(1, 3), F(1, 6)), (F(1, 5), F(2, 5), F(2, 5)))
    )
    outcome_p = (F(1, 10), F(9, 10), F(2, 5), F(4, 5), F(1, 5), F(3, 5))
    chosen = tuple(outcome_p[3 * m + z] for m in range(2) for z in range(z_domain))
    return WorldSpec(
        family="test_dag",
        topology="X-to-M-to-Y-correlated-U-Z",
        variables=("X", "M", "Y", "U", "Z"),
        domains=(2, 2, 2, 2, z_domain),
        state_names=(("0", "1"),) * 4 + (tuple(map(str, range(z_domain))),),
        edges=((0, 1), (3, 1), (1, 2), (4, 2), (3, 4)),
        parents={0: (), 1: (0, 3), 2: (1, 4), 3: (), 4: (3,)},
        cpt={
            0: ((F(1, 2), F(1, 2)),),
            1: tuple((1 - p, p) for p in (F(1, 10), F(7, 10), F(4, 5), F(3, 10))),
            2: tuple((1 - p, p) for p in chosen),
            3: ((F(0), F(1)) if zero_mass else (F(3, 7), F(4, 7)),),
            4: z_rows,
        },
    )


def exact_cost(world, endpoint, u, left, right):
    result = F(0)
    for z, weight in enumerate(world.cpt[4][u]):
        lp = world.cpt[2][left * world.domains[4] + z][1]
        rp = world.cpt[2][right * world.domains[4] + z][1]
        value = min(lp, rp) if endpoint == "upper" else max(F(0), rp - lp)
        result += weight * value
    return result


def exact_endpoint(world, endpoint):
    """Keep one M coupling per U across every Z; enumerate its two vertices."""
    answer = F(0)
    for u, weight in enumerate(world.cpt[3][0]):
        left_one, right_one = world.cpt[1][u][1], world.cpt[1][2 + u][1]
        values = []
        for both_one in (max(F(0), left_one + right_one - 1), min(left_one, right_one)):
            coupling = (
                (1 - left_one - right_one + both_one, right_one - both_one),
                (left_one - both_one, both_one),
            )
            values.append(
                sum(
                    (
                        coupling[a][b] * exact_cost(world, endpoint, u, a, b)
                        for a, b in product(range(2), repeat=2)
                    ),
                    F(0),
                )
            )
        answer += weight * (min(values) if endpoint == "lower" else max(values))
    return answer


class NumericTerminalCompilationTests(unittest.TestCase):
    def test_coefficients_and_endpoints_preserve_shared_response_couplings(self):
        for z_domain, zero_mass, projected, endpoint in product(
            (2, 3), (False, True), (False, True), ("lower", "upper")
        ):
            with self.subTest(
                z=z_domain, zero_mass=zero_mass, projected=projected, endpoint=endpoint
            ):
                world = world_fixture(z_domain, zero_mass)
                answers = []
                contexts = []
                for compiled in (False, True):
                    context = (
                        nullcontext()
                        if compiled
                        else patch.object(
                            _SparseResponseModel,
                            "_compile_numeric_terminal_factors",
                            return_value=None,
                        )
                    )
                    with context:
                        owner = _SparseResponseModel(
                            world,
                            0,
                            2,
                            baseline_value=0,
                            treatment_value=1,
                            outcome_state=None,
                            outcome_events=((1,), (1,)) if endpoint == "upper" else ((0,), (1,)),
                            sense="maximize" if endpoint == "upper" else "minimize",
                            target_outer_bounds=(0.0, 1.0),
                            terminal_event_endpoint=endpoint,
                            projected_message_bounds=projected,
                        )
                    contexts.append((owner.context_rows, owner.context_components))
                    if compiled:
                        factors, removed = owner._compile_numeric_terminal_factors()
                        self.assertEqual(removed, frozenset({4}))
                        for u, a, b in product(range(2), repeat=3):
                            assignment = {(3, -1): u, (1, 0): a, (1, 1): b}
                            coefficient = 1.0
                            for factor in factors:
                                key = tuple(assignment[token] for token in factor.scope)
                                coefficient *= (
                                    factor.values(key)
                                    if callable(factor.values)
                                    else factor.values[key]
                                )
                            self.assertAlmostEqual(
                                coefficient,
                                float(world.cpt[3][0][u] * exact_cost(world, endpoint, u, a, b)),
                                places=14,
                            )
                    value, _ = owner.optimize(time_limit_seconds=5.0)
                    answers.append(value)
                    self.assertAlmostEqual(value, float(exact_endpoint(world, endpoint)), places=8)
                self.assertEqual(contexts[0], contexts[1])
                self.assertAlmostEqual(answers[0], answers[1], places=8)


if __name__ == "__main__":
    unittest.main()
