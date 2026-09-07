"""Regression tests for semantics-preserving probability-kernel compilation."""

from fractions import Fraction
from itertools import product
from unittest.mock import patch

from cpt_world.counterfactual_solver import _two_mediator_joint_bounds
from cpt_world.query_truth import (
    worldspec_interventional_distribution,
    worldspec_projected_interventional_distribution,
)
from cpt_world.world_space import WorldSpec


def test_projected_law_preserves_mixed_radix_and_measured_interventions():
    domains = (2, 3, 2, 3)
    parents = {0: (), 1: (0,), 2: (0, 1), 3: (0,)}
    rows = {}
    for node, domain in enumerate(domains):
        contexts = list(product(*(range(domains[p]) for p in parents[node])))
        rows[node] = tuple(
            tuple(Fraction((i + j) % domain, domain * (domain - 1) // 2) for j in range(domain))
            for i, _ in enumerate(contexts)
        )
    world = WorldSpec(
        family="test_dag",
        topology="mixed-radix",
        variables=("A", "B", "C", "D"),
        domains=domains,
        state_names=tuple(tuple(map(str, range(d))) for d in domains),
        parents=parents,
        edges=((0, 1), (0, 2), (1, 2), (0, 3)),
        cpt=rows,
    )
    for interventions in ({}, {1: 2}, {0: 1, 1: 0}, {2: 1, 3: 2}):
        joint = worldspec_interventional_distribution(world, interventions)
        for measure in ((2,), (2, 1), (3, 2, 0), (3, 2, 1, 0)):
            expected = {
                states: Fraction(0) for states in product(*(range(domains[v]) for v in measure))
            }
            for states, mass in joint:
                expected[tuple(states[v] for v in measure)] += mass
            assert (
                dict(worldspec_projected_interventional_distribution(world, interventions, measure))
                == expected
            )


def test_layered_dimension_guard_precedes_transport_vertex_enumeration():
    # X,Z1,Z2 -> M4 -> L2 -> Y2, with X -> Y. Four independent
    # upstream contexts each have a nine-dimensional transport polytope;
    # at least 10**4 combinations already exceed the unchanged 4096 guard.
    domains = (2, 2, 2, 4, 2, 2)
    parents = {0: (), 1: (), 2: (), 3: (0, 1, 2), 4: (3,), 5: (0, 4)}
    world = WorldSpec(
        family="test_dag",
        topology="layered-positive-contexts",
        variables=tuple("ABCDEF"),
        domains=domains,
        state_names=tuple(tuple(map(str, range(d))) for d in domains),
        parents=parents,
        edges=((0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (0, 5)),
        cpt={
            node: tuple(
                (Fraction(1, d),) * d for _ in product(*(range(domains[p]) for p in parents[node]))
            )
            for node, d in enumerate(domains)
        },
    )
    with patch(
        "cpt_world.query_truth._response_coupling_vertices",
        side_effect=AssertionError("unnecessary vertex enumeration"),
    ):
        assert (
            _two_mediator_joint_bounds(
                world,
                0,
                5,
                baseline_value=0,
                treatment_value=1,
                outcome_events=((0,), (1,)),
                time_limit_seconds=5,
            )
            is None
        )
