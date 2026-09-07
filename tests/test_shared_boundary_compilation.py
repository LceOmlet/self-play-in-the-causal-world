"""Exact references for marginalizing shared ancestors without splitting SCMs."""

from fractions import Fraction as F
from itertools import product

from cpt_world.counterfactual_solver import (
    _SparseResponseModel,
    sparse_individual_counterfactual_probability_bounds,
)
from cpt_world.query_truth import worldspec_projected_interventional_distribution
from cpt_world.world_space import WorldSpec


def _shared_chain():
    return WorldSpec(
        family="test_dag",
        topology="shared-ancestor-chain",
        variables=tuple("ZWXMY"),
        domains=(2,) * 5,
        state_names=(("0", "1"),) * 5,
        parents={0: (), 1: (0,), 2: (), 3: (2,), 4: (1, 3)},
        edges=((0, 1), (2, 3), (1, 4), (3, 4)),
        cpt={
            0: ((F(2, 5), F(3, 5)),),
            1: ((F(4, 5), F(1, 5)), (F(1, 10), F(9, 10))),
            2: ((F(1, 2), F(1, 2)),),
            3: ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))),
            4: ((F(9, 10), F(1, 10)), (F(3, 5), F(2, 5)), (F(2, 5), F(3, 5)), (F(1, 10), F(9, 10))),
        },
    )


def test_compiled_shared_chain_matches_independent_exact_vertex_reference():
    # Full rational response-vertex enumeration yields exactly 30/103,65/103.
    # In particular the M response is shared across the two W strata.
    result = sparse_individual_counterfactual_probability_bounds(
        _shared_chain(),
        2,
        4,
        factual_value=0,
        counterfactual_value=1,
        factual_outcome_state=0,
        target_outcome_state=1,
    )
    assert result.certification == "exact"
    assert abs(result.lower - float(F(30, 103))) < 1e-9
    assert abs(result.upper - float(F(65, 103))) < 1e-9


def test_compiled_shared_joint_retains_correlation_and_zero_mass_contexts():
    for zero_root in (False, True):
        parents = {0: (), 1: (0,), 2: (0,), 3: (), 4: (3,), 5: (1, 2, 4)}
        cpt = {
            n: tuple((F(1, 2), F(1, 2)) for _ in product(*(range(2) for p in parents[n])))
            for n in range(6)
        }
        cpt[0] = ((F(1), F(0)),) if zero_root else ((F(2, 5), F(3, 5)),)
        cpt[1] = ((F(1), F(0)), (F(0), F(1)))
        cpt[2] = ((F(4, 5), F(1, 5)), (F(1, 10), F(9, 10)))
        world = WorldSpec(
            family="test_dag",
            topology="shared-confounded-boundary",
            variables=tuple("ZWVXMY"),
            domains=(2,) * 6,
            state_names=(("0", "1"),) * 6,
            parents=parents,
            edges=((0, 1), (0, 2), (3, 4), (1, 5), (2, 5), (4, 5)),
            cpt=cpt,
        )
        owner = _SparseResponseModel(
            world,
            3,
            5,
            baseline_value=0,
            treatment_value=1,
            outcome_state=None,
            outcome_events=((0,), (1,)),
            sense="minimize",
            target_outer_bounds=(0, 1),
            terminal_event_endpoint="lower",
        )
        assert owner.shared == (1, 2)
        expected = worldspec_projected_interventional_distribution(world, {}, (1, 2))
        for assignment, mass in expected:
            observed = 1.0
            for i, node in enumerate(owner.shared):
                observed *= owner.shared_factor_overrides[node].values[assignment[: i + 1]]
            assert abs(observed - float(mass)) < 1e-14
        conditional = owner.shared_factor_overrides[2].values
        for prefix in (0, 1):
            assert abs(sum(conditional[(prefix, state)] for state in (0, 1)) - 1) < 1e-14
