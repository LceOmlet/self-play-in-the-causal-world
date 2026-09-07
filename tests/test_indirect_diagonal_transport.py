"""Exact structural reductions checked against independent response vertices."""

from fractions import Fraction as F

import pytest
from cpt_world.counterfactual_solver import (
    _indirect_mediator_joint_bounds,
    sparse_individual_counterfactual_probability_bounds,
)
from cpt_world.query_truth import reference_individual_counterfactual_probability_bounds
from cpt_world.world_space import WorldSpec


def chain(marginals, outcome_rows):
    m, d = len(marginals[0]), len(outcome_rows[0])
    return WorldSpec(
        family="test_dag",
        topology="indirect-diagonal-transport",
        variables=("X", "M", "Y"),
        domains=(2, m, d),
        state_names=tuple(tuple(map(str, range(n))) for n in (2, m, d)),
        parents={0: (), 1: (0,), 2: (1,)},
        edges=((0, 1), (1, 2)),
        cpt={0: ((F(1, 2), F(1, 2)),), 1: marginals, 2: outcome_rows},
    )


@pytest.mark.parametrize("states", (2, 3))
def test_uniform_chain_has_sharp_partition_value_without_nonlinear_search(states):
    world = chain(((F(1, states),) * states,) * 2, ((F(1, 2), F(1, 2)),) * states)
    actual = sparse_individual_counterfactual_probability_bounds(
        world,
        0,
        2,
        factual_value=0,
        counterfactual_value=1,
        factual_outcome_state=0,
        target_outcome_state=1,
        time_limit_seconds=5,
    )
    assert actual.backend == "indirect_mediator_diagonal_transport"
    assert actual.certification == "exact"
    assert abs(actual.lower) < 1e-9
    assert abs(actual.upper - float(F(2 * (states // 2), states))) < 1e-9


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("events", ((0, 0), (0, 1), (1, 0), (1, 1)))
def test_nonuniform_three_state_chain_matches_full_rational_vertices(reverse, events):
    world = chain(
        ((F(1, 2), F(1, 3), F(1, 6)), (F(1, 4), F(1, 2), F(1, 4))),
        ((F(1, 5), F(4, 5)), (F(3, 5), F(2, 5)), (F(1, 2), F(1, 2))),
    )
    kwargs = dict(
        factual_value=int(reverse),
        counterfactual_value=int(not reverse),
        factual_outcome_state=events[0],
        target_outcome_state=events[1],
    )
    expected = reference_individual_counterfactual_probability_bounds(world, 0, 2, **kwargs)
    actual = sparse_individual_counterfactual_probability_bounds(world, 0, 2, **kwargs)
    assert actual.backend == "indirect_mediator_diagonal_transport"
    assert abs(actual.lower - float(expected[0])) < 1e-9
    assert abs(actual.upper - float(expected[1])) < 1e-9


@pytest.mark.parametrize("events", ((0, 0), (0, 1), (1, 2)))
def test_binary_mediator_multistate_outcome_matches_full_rational_vertices(events):
    world = chain(
        ((F(1, 3), F(2, 3)), (F(3, 4), F(1, 4))),
        ((F(1, 2), F(1, 3), F(1, 6)), (F(1, 4), F(1, 2), F(1, 4))),
    )
    kwargs = dict(
        factual_value=0,
        counterfactual_value=1,
        factual_outcome_state=events[0],
        target_outcome_state=events[1],
    )
    expected = reference_individual_counterfactual_probability_bounds(world, 0, 2, **kwargs)
    actual = sparse_individual_counterfactual_probability_bounds(world, 0, 2, **kwargs)
    assert actual.backend == "indirect_mediator_diagonal_transport"
    assert abs(actual.lower - float(expected[0])) < 1e-9
    assert abs(actual.upper - float(expected[1])) < 1e-9


@pytest.mark.parametrize("root", ((F(1, 3), F(2, 3)), (F(1), F(0))))
def test_shared_upstream_and_zero_context_preserve_terminal_mechanism(root):
    world = WorldSpec(
        family="test_dag",
        topology="shared-terminal-response",
        variables=("Z", "X", "M", "Y"),
        domains=(2,) * 4,
        state_names=(("0", "1"),) * 4,
        parents={0: (), 1: (), 2: (0, 1), 3: (2,)},
        edges=((0, 2), (1, 2), (2, 3)),
        cpt={
            0: (root,),
            1: ((F(1, 2), F(1, 2)),),
            2: ((F(1), F(0)), (F(1, 4), F(3, 4)), (F(2, 5), F(3, 5)), (F(4, 5), F(1, 5))),
            3: ((F(1, 5), F(4, 5)), (F(3, 5), F(2, 5))),
        },
    )
    kwargs = dict(
        factual_value=0, counterfactual_value=1, factual_outcome_state=0, target_outcome_state=1
    )
    expected = reference_individual_counterfactual_probability_bounds(world, 1, 3, **kwargs)
    actual = sparse_individual_counterfactual_probability_bounds(world, 1, 3, **kwargs)
    assert actual.backend == "indirect_mediator_diagonal_transport"
    assert abs(actual.lower - float(expected[0])) < 1e-9
    assert abs(actual.upper - float(expected[1])) < 1e-9


def test_structurally_unsupported_cases_retain_original_owner():
    for states, terminal_states, events in (
        (4, 2, ((0,), (1,))),
        (3, 3, ((0,), (1,))),
        (2, 3, ((0, 1), (1, 2))),
    ):
        world = chain(
            ((F(1, states),) * states,) * 2, ((F(1, terminal_states),) * terminal_states,) * states
        )
        assert (
            _indirect_mediator_joint_bounds(
                world,
                0,
                2,
                mediator=1,
                baseline_value=0,
                treatment_value=1,
                outcome_events=events,
                time_limit_seconds=5,
            )
            is None
        )
