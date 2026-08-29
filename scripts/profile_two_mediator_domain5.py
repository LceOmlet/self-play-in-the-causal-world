"""Aggregate structural probe for the five-state two-mediator fast-path gap.

The probe never emits task identities.  It enumerates every spanning-tree
basis of the complete bipartite support graph for the upstream transport and
counts the distinct feasible vertices for the complete structural class that
the current layered owner rejects only because its first mediator has five
states.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Iterator
from itertools import product
from typing import Any

from measure_counterfactual_optimizer import (
    DISTRIBUTION_COUNT,
    DISTRIBUTION_START_SEED,
    _query_indices,
)

from cpt_world import WorldGrammar, iter_sampled_seeds, sample_task_world
from cpt_world import counterfactual_solver as solver
from cpt_world.query_truth import _response_coupling_vertices


def _bipartite_tree_edges(size: int) -> Iterator[tuple[tuple[int, int], ...]]:
    """Generate every spanning tree of K(size,size) exactly once.

    The two words are the bipartite Pruefer code.  An entry on the left side
    is recorded when a right leaf is deleted and conversely.  Reconstructing
    the ordinary Pruefer deletion order gives a bijection, so the generator
    has exactly ``size ** (2 * size - 2)`` outputs without filtering graphs.
    """

    left_words = tuple(product(range(size), repeat=size - 1))
    for left_word in left_words:
        left_base = [1] * size
        for value in left_word:
            left_base[value] += 1
        for right_word in product(range(size), repeat=size - 1):
            degrees = [*left_base, *([1] * size)]
            for value in right_word:
                degrees[size + value] += 1
            left_cursor = 0
            right_cursor = 0
            edges: list[tuple[int, int]] = []
            for _ in range(2 * size - 2):
                leaf = next(index for index, degree in enumerate(degrees) if degree == 1)
                if leaf < size:
                    neighbor = size + right_word[right_cursor]
                    right_cursor += 1
                    edge = (leaf, neighbor - size)
                else:
                    neighbor = left_word[left_cursor]
                    left_cursor += 1
                    edge = (neighbor, leaf - size)
                edges.append(edge)
                degrees[leaf] -= 1
                degrees[neighbor] -= 1
            remaining = [index for index, degree in enumerate(degrees) if degree == 1]
            if len(remaining) != 2:
                raise RuntimeError("bipartite Pruefer decoder left invalid degrees")
            left = remaining[0] if remaining[0] < size else remaining[1]
            right = remaining[1] if remaining[1] >= size else remaining[0]
            edges.append((left, right - size))
            yield tuple(edges)


def _transport_vertex(
    edges: tuple[tuple[int, int], ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    tolerance: float = 1e-9,
) -> tuple[float, ...] | None:
    """Solve one tree-supported transport basis by exact leaf elimination."""

    size = len(left)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(2 * size)]
    for edge_index, (left_node, right_node) in enumerate(edges):
        adjacency[left_node].append((size + right_node, edge_index))
        adjacency[size + right_node].append((left_node, edge_index))
    degree = [len(neighbors) for neighbors in adjacency]
    residual = [*left, *(-value for value in right)]
    values = [0.0] * len(edges)
    active = [True] * len(edges)
    leaves = [index for index, value in enumerate(degree) if value == 1]
    while leaves:
        node = leaves.pop()
        if degree[node] != 1:
            continue
        neighbor, edge_index = next(
            pair for pair in adjacency[node] if active[pair[1]]
        )
        value = residual[node] if node < size else -residual[node]
        if value < -tolerance:
            return None
        values[edge_index] = max(0.0, value)
        residual[neighbor] += residual[node]
        residual[node] = 0.0
        active[edge_index] = False
        degree[node] -= 1
        degree[neighbor] -= 1
        if degree[neighbor] == 1:
            leaves.append(neighbor)
    if any(active) or max(abs(value) for value in residual) > 10.0 * tolerance:
        return None
    matrix = [0.0] * (size * size)
    for edge, value in zip(edges, values, strict=True):
        matrix[edge[0] * size + edge[1]] = value
    return tuple(matrix)


def _eligible_rows(world: Any, treatment: int, outcome: int, baseline: int, treated: int):
    ancestors = solver._ancestors(world, outcome) | {outcome}
    affected = tuple(
        node
        for node in solver._topological_order(world)
        if node in (solver._descendants(world, treatment) & ancestors)
    )
    if len(affected) != 3 or affected[-1] != outcome:
        return None
    first, second, _ = affected
    if world.domains[first] != 5:
        return None
    if treatment not in world.parents[first]:
        return None
    if first not in world.parents[second] or second not in world.parents[outcome]:
        return None
    if any(parent in affected for parent in world.parents[first]):
        return None
    if {parent for parent in world.parents[second] if parent in affected} != {first}:
        return None
    if {
        parent for parent in world.parents[outcome] if parent in affected
    } not in ({second}, {first, second}):
        return None

    first_shared = tuple(parent for parent in world.parents[first] if parent != treatment)
    rows: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for assignment in product(*(range(world.domains[parent]) for parent in first_shared)):
        shared = dict(zip(first_shared, assignment, strict=True))
        left_context = tuple(
            baseline if parent == treatment else shared[parent]
            for parent in world.parents[first]
        )
        right_context = tuple(
            treated if parent == treatment else shared[parent]
            for parent in world.parents[first]
        )
        rows.append(
            (
                tuple(
                    float(value)
                    for value in world.cpt[first][solver._row_index(world, first, left_context)]
                ),
                tuple(
                    float(value)
                    for value in world.cpt[first][solver._row_index(world, first, right_context)]
                ),
            )
        )
    return tuple(rows), world.domains[second]


def main() -> None:
    grammar = WorldGrammar()
    row_pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    class_records: list[tuple[int, int]] = []
    for sample_index in range(
        DISTRIBUTION_START_SEED,
        DISTRIBUTION_START_SEED + DISTRIBUTION_COUNT,
    ):
        rendered = iter_sampled_seeds(
            grammar,
            start_seed=sample_index,
            count=1,
            query_types=("individual_counterfactual_probability",),
        )[0]
        world = sample_task_world(
            grammar, sample_index, "individual_counterfactual_probability"
        )
        treatment, outcome, baseline, treated, _, _ = _query_indices(world, rendered)
        eligible = _eligible_rows(world, treatment, outcome, baseline, treated)
        if eligible is None:
            continue
        rows, second_domain = eligible
        row_pairs.extend(rows)
        class_records.append((len(rows), second_domain))

    small_supports = tuple(_bipartite_tree_edges(3))
    if len(small_supports) != 3**4 or len(set(small_supports)) != 3**4:
        raise RuntimeError("bipartite Pruefer generator is not bijective")
    validation_rows = ((0.2, 0.3, 0.5), (0.4, 0.1, 0.5))
    tree_vertices = {
        tuple(round(value, 12) for value in vertex)
        for edges in small_supports
        if (vertex := _transport_vertex(edges, *validation_rows)) is not None
    }
    owner_vertices = set()
    for vertex in _response_coupling_vertices(validation_rows):
        matrix = [0.0] * 9
        for response, mass in vertex:
            matrix[response[0] * 3 + response[1]] = float(mass)
        owner_vertices.add(tuple(round(value, 12) for value in matrix))
    if tree_vertices != owner_vertices:
        raise RuntimeError("tree-basis transport vertices disagree with exact owner")

    expected_supports = 5**8
    vertices_by_pair: list[set[tuple[float, ...]]] = [set() for _ in row_pairs]
    started = time.perf_counter()
    support_count = 0
    for edges in _bipartite_tree_edges(5):
        support_count += 1
        for index, (left, right) in enumerate(row_pairs):
            vertex = _transport_vertex(edges, left, right)
            if vertex is not None:
                vertices_by_pair[index].add(
                    tuple(round(value, 12) for value in vertex)
                )
    enumeration_seconds = time.perf_counter() - started
    if support_count != expected_supports:
        raise RuntimeError("bipartite Pruefer generator emitted the wrong count")
    vertex_counts = [len(vertices) for vertices in vertices_by_pair]
    selection_counts = [
        int(np_product(vertex_counts[offset : offset + context_count]))
        for offset, (context_count, _) in _offset_records(class_records)
    ]

    payload = {
        "cohort_size": DISTRIBUTION_COUNT,
        "eligible_worlds": len(class_records),
        "first_shared_contexts": dict(sorted(Counter(count for count, _ in class_records).items())),
        "second_domains": dict(sorted(Counter(domain for _, domain in class_records).items())),
        "transport_row_pairs": len(row_pairs),
        "complete_bipartite_tree_supports": support_count,
        "vertex_count_min": min(vertex_counts, default=0),
        "vertex_count_median": statistics.median(vertex_counts) if vertex_counts else 0,
        "vertex_count_max": max(vertex_counts, default=0),
        "vertex_enumeration_seconds": enumeration_seconds,
        "upstream_selection_log10_min": min(
            (math.log10(value) for value in selection_counts), default=0.0
        ),
        "upstream_selection_log10_median": (
            statistics.median(math.log10(value) for value in selection_counts)
            if selection_counts
            else 0.0
        ),
        "upstream_selection_log10_max": max(
            (math.log10(value) for value in selection_counts), default=0.0
        ),
    }
    print(json.dumps(payload, sort_keys=True))


def np_product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _offset_records(records: list[tuple[int, int]]) -> Iterator[tuple[int, tuple[int, int]]]:
    offset = 0
    for record in records:
        yield offset, record
        offset += record[0]


if __name__ == "__main__":
    main()
