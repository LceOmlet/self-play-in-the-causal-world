"""Analytic and independent rational tests for the response transport bound."""

from fractions import Fraction as F

import pytest
from cpt_world.counterfactual_solver import (
    _indirect_response_transport_envelope,
    sparse_individual_counterfactual_probability_bounds,
)
from cpt_world.query_truth import _reference_counterfactual_joint_bounds
from cpt_world.world_space import WorldSpec


@pytest.mark.parametrize("states", (2, 3, 4, 5))
def test_uniform_mediator_matches_sharp_partition_proof(states):
    world = WorldSpec(
        family="test_dag",
        topology="uniform-mediator-partition",
        variables=("X", "M", "Y"),
        domains=(2, states, 2),
        state_names=(("0", "1"), tuple(map(str, range(states))), ("0", "1")),
        parents={0: (), 1: (0,), 2: (1,)},
        edges=((0, 1), (1, 2)),
        cpt={
            0: ((F(1, 2), F(1, 2)),),
            1: ((F(1, states),) * states,) * 2,
            2: ((F(1, 2), F(1, 2)),) * states,
        },
    )
    envelope = _indirect_response_transport_envelope(
        world,
        0,
        2,
        baseline_value=0,
        treatment_value=1,
        outcome_events=((0,), (1,)),
        time_limit_seconds=5,
    )
    expected_joint_upper = F(states // 2, states)
    assert abs(envelope.lower) < 1e-9
    assert abs(envelope.upper - float(expected_joint_upper)) < 1e-9
    result = sparse_individual_counterfactual_probability_bounds(
        world,
        0,
        2,
        factual_value=0,
        counterfactual_value=1,
        factual_outcome_state=0,
        target_outcome_state=1,
        time_limit_seconds=5,
    )
    assert result.certification == "exact"
    assert abs(result.lower) < 1e-8
    assert abs(result.upper - float(2 * expected_joint_upper)) < 3e-8
    if states == 3:
        assert envelope.attained_upper is not None
        assert result.backend == "one_mediator_response_envelope_attainment"


@pytest.mark.parametrize("events", (((0,), (1,)), ((0, 1), (1, 2)), ((1,), (1,))))
def test_multistate_overlapping_events_enclose_exact_reference(events):
    world = WorldSpec(
        family="test_dag",
        topology="overlapping-events",
        variables=("X", "M", "Y"),
        domains=(2, 2, 3),
        state_names=(("0", "1"), ("0", "1"), ("0", "1", "2")),
        parents={0: (), 1: (0,), 2: (1,)},
        edges=((0, 1), (1, 2)),
        cpt={
            0: ((F(1, 2), F(1, 2)),),
            1: ((F(1, 3), F(2, 3)), (F(3, 4), F(1, 4))),
            2: ((F(1, 2), F(1, 3), F(1, 6)), (F(1, 4), F(1, 2), F(1, 4))),
        },
    )
    envelope = _indirect_response_transport_envelope(
        world,
        0,
        2,
        baseline_value=0,
        treatment_value=1,
        outcome_events=events,
    )
    exact = _reference_counterfactual_joint_bounds(
        world,
        0,
        2,
        baseline_value=0,
        treatment_value=1,
        baseline_outcome_states=events[0],
        treatment_outcome_states=events[1],
    )
    assert envelope.lower <= float(exact[0]) + 1e-9
    assert envelope.upper >= float(exact[1]) - 1e-9


@pytest.mark.parametrize("root", ((F(1, 3), F(2, 3)), (F(1), F(0))))
def test_shared_upstream_contexts_preserve_one_terminal_response(root):
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
    envelope = _indirect_response_transport_envelope(
        world,
        1,
        3,
        baseline_value=0,
        treatment_value=1,
        outcome_events=((0,), (1,)),
    )
    exact = _reference_counterfactual_joint_bounds(
        world,
        1,
        3,
        baseline_value=0,
        treatment_value=1,
        baseline_outcome_states=(0,),
        treatment_outcome_states=(1,),
    )
    assert envelope.response_blocks == 1
    assert envelope.lower <= float(exact[0]) + 1e-9
    assert envelope.upper >= float(exact[1]) - 1e-9
