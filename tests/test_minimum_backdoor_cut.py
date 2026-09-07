"""Compare the minimum cut with exhaustive active-path back-door semantics."""

import unittest
from itertools import combinations

from cpt_world.world_space import _minimum_backdoor_adjustment_size


def reachable(node_count, edges, source):
    seen = {source}
    queue = [source]
    for node in queue:
        for parent, child in edges:
            if parent == node and child not in seen:
                seen.add(child)
                queue.append(child)
    return seen - {source}


def reference_minimum(node_count, edges, treatment, outcome):
    """Enumerate conditioning sets and simple active paths; no moral graph."""
    allowed = tuple(
        node
        for node in range(node_count)
        if node not in {treatment, outcome} and node not in reachable(node_count, edges, treatment)
    )
    backdoor = frozenset((a, b) for a, b in edges if a != treatment)
    neighbors = {node: set() for node in range(node_count)}
    for a, b in backdoor:
        neighbors[a].add(b)
        neighbors[b].add(a)
    paths = []

    def visit(path):
        if path[-1] == outcome:
            paths.append(path)
            return
        for node in neighbors[path[-1]]:
            if node not in path:
                visit((*path, node))

    visit((treatment,))
    for size in range(len(allowed) + 1):
        for subset in combinations(allowed, size):
            condition = frozenset(subset)
            activated = set(condition)
            for node in range(node_count):
                if reachable(node_count, tuple(backdoor), node) & condition:
                    activated.add(node)

            def active(path, condition, activated):
                for a, node, b in zip(path, path[1:], path[2:], strict=False):
                    collider = (a, node) in backdoor and (b, node) in backdoor
                    if collider and node not in activated:
                        return False
                    if not collider and node in condition:
                        return False
                return True

            if not any(active(path, condition, activated) for path in paths):
                return size
    raise AssertionError("A causal role in a fully observed DAG has a parent adjustment set")


class MinimumBackdoorCutTests(unittest.TestCase):
    def test_all_small_dags_against_independent_active_path_enumeration(self):
        for n in range(2, 6):
            edge_slots = tuple(combinations(range(n), 2))
            for mask in range(1 << len(edge_slots)):
                edges = tuple(edge for i, edge in enumerate(edge_slots) if mask & (1 << i))
                for treatment in range(n):
                    for outcome in sorted(reachable(n, edges, treatment)):
                        with self.subTest(n=n, mask=mask, x=treatment, y=outcome):
                            expected = reference_minimum(n, edges, treatment, outcome)
                            self.assertEqual(
                                _minimum_backdoor_adjustment_size(n, edges, treatment, outcome),
                                expected,
                            )
                            # Node identifiers need not be a topological ordering.
                            relabeled = tuple((n - 1 - a, n - 1 - b) for a, b in edges)
                            self.assertEqual(
                                _minimum_backdoor_adjustment_size(
                                    n, relabeled, n - 1 - treatment, n - 1 - outcome
                                ),
                                expected,
                            )

    def test_forbidden_descendants_stay_in_graph_but_cannot_be_cut(self):
        # All five back-door routes share D, but D is a treatment descendant.
        edges = tuple(edge for z in range(5) for edge in ((z, 5), (z, 6))) + (
            (5, 6),
            (6, 7),
        )
        self.assertEqual(reference_minimum(8, edges, 5, 7), 5)
        self.assertEqual(_minimum_backdoor_adjustment_size(8, edges, 5, 7), 5)
        # Making the shared bottleneck a nondescendant reduces the answer to one.
        nondescendant = tuple(edge for edge in edges if edge != (5, 6)) + ((5, 7),)
        self.assertEqual(reference_minimum(8, nondescendant, 5, 7), 1)
        self.assertEqual(_minimum_backdoor_adjustment_size(8, nondescendant, 5, 7), 1)

    def test_nonancestor_collider_is_not_moralized_into_a_false_path(self):
        edges = ((0, 1), (1, 2), (0, 3), (2, 3))
        self.assertEqual(reference_minimum(4, edges, 1, 2), 0)
        self.assertEqual(_minimum_backdoor_adjustment_size(4, edges, 1, 2), 0)


if __name__ == "__main__":
    unittest.main()
