"""Check the compact kernel against exact conditional-expectation projections."""

import math
import unittest
from fractions import Fraction
from itertools import combinations, product

from cpt_world.world_space import _parent_interaction_projection


def exact_projection(table, domains, positions):
    """Independent rational inclusion-exclusion; no production marginal helper."""
    assignments = tuple(product(*(range(domain) for domain in domains)))
    output = []
    for target in assignments:
        values = []
        for child in range(len(table[0])):
            value = Fraction(0)
            for size in range(len(positions) + 1):
                for subset in combinations(positions, size):
                    selected = [
                        Fraction(table[row][child])
                        for row, assignment in enumerate(assignments)
                        if all(assignment[axis] == target[axis] for axis in subset)
                    ]
                    value += (-1) ** (len(positions) - size) * sum(selected) / len(selected)
            values.append(value)
        output.append(tuple(values))
    return tuple(output)


class ANOVAAxisProjectionTests(unittest.TestCase):
    def test_every_block_matches_exact_projection_on_mixed_domains(self):
        for domains, children in (
            ((2,), 2),
            ((5,), 5),
            ((2, 3, 2), 3),
            ((3, 2, 4), 2),
            ((2, 2, 2, 2), 4),
        ):
            table = tuple(
                tuple(
                    (row * row * (child + 1) + 3 * row + child) % 31 - 15
                    for child in range(children)
                )
                for row in range(math.prod(domains))
            )
            for size in range(1, len(domains) + 1):
                for positions in combinations(range(len(domains)), size):
                    with self.subTest(domains=domains, positions=positions):
                        expected = exact_projection(table, domains, positions)
                        actual = _parent_interaction_projection(table, domains, positions)
                        for observed, reference in zip(actual, expected, strict=True):
                            for left, right in zip(observed, reference, strict=True):
                                self.assertAlmostEqual(left, float(right), places=12)

    def test_axis_and_state_relabeling_commute_with_projection(self):
        domains = (2, 3, 4)
        assignments = tuple(product(*(range(domain) for domain in domains)))
        table = tuple((float(i % 7), float((i * 11) % 17)) for i in range(len(assignments)))
        original = dict(zip(assignments, table, strict=True))
        expected = dict(
            zip(assignments, _parent_interaction_projection(table, domains, (0, 2)), strict=True)
        )
        order = (2, 0, 1)
        maps = ((1, 0), (2, 0, 1), (3, 1, 0, 2))
        reordered_domains = tuple(domains[axis] for axis in order)
        reordered_assignments = tuple(product(*(range(domain) for domain in reordered_domains)))

        def old_assignment(states):
            original_states = [0] * 3
            for new_axis, old_axis in enumerate(order):
                original_states[old_axis] = maps[old_axis][states[new_axis]]
            return tuple(original_states)

        transformed = tuple(original[old_assignment(states)] for states in reordered_assignments)
        actual = _parent_interaction_projection(transformed, reordered_domains, (0, 1))
        for states, row in zip(reordered_assignments, actual, strict=True):
            for left, right in zip(row, expected[old_assignment(states)], strict=True):
                self.assertAlmostEqual(left, right, places=12)

    def test_invalid_projection_contract_still_rejected(self):
        for positions in ((), (1, 0), (0, 0), (-1,), (2,)):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                _parent_interaction_projection(((1.0, 2.0),) * 6, (2, 3), positions)
        with self.assertRaises(ValueError):
            _parent_interaction_projection(((1.0, 2.0),) * 5, (2, 3), (0,))


if __name__ == "__main__":
    unittest.main()
