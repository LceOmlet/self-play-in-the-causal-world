from __future__ import annotations

import unittest
from fractions import Fraction
from unittest.mock import MagicMock, patch

from pyscipopt import SCIP_RESULT, Model

from cpt_world import (
    WorldSpec,
    reference_counterfactual_transition_bounds,
    reference_individual_counterfactual_probability_bounds,
)
from cpt_world.counterfactual_solver import (
    _coarsen_terminal_event_outcome,
    _eliminate_factor_tokens,
    _exact_pairwise_map,
    _global_bounds_numerically_closed,
    _PairwiseMapOptimization,
    _ResponsePricer,
    _SparseResponseModel,
    _SymbolicFactor,
    sparse_counterfactual_transition_bounds,
    sparse_individual_counterfactual_probability_bounds,
)
from cpt_world.query_truth import (
    interventional_frechet_transition_outer_bounds,
    interventional_probability,
)


def _uniform_multivalued_chain() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="X2-to-M2-to-Y3-uniform",
        variables=("X", "M", "Y"),
        domains=(2, 2, 3),
        state_names=(("0", "1"), ("0", "1"), ("0", "1", "2")),
        edges=((0, 1), (1, 2)),
        parents={0: (), 1: (0,), 2: (1,)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(1, 2), Fraction(1, 2)),) * 2,
            2: ((Fraction(1, 3),) * 3,) * 2,
        },
    )


def _non_direct_terminal_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="non-direct-terminal-endpoint-decomposition",
        variables=("X", "M", "Y"),
        domains=(2, 2, 3),
        state_names=(("0", "1"), ("0", "1"), ("0", "1", "2")),
        edges=((0, 1), (1, 2)),
        parents={0: (), 1: (0,), 2: (1,)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: (
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(3, 10), Fraction(7, 10)),
            ),
            2: (
                (Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)),
                (Fraction(1, 4), Fraction(7, 20), Fraction(2, 5)),
            ),
        },
    )


def _four_state_non_direct_terminal_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="four-state-terminal-event-quotient",
        variables=("X", "M", "Y"),
        domains=(2, 2, 4),
        state_names=(("0", "1"), ("0", "1"), ("0", "1", "2", "3")),
        edges=((0, 1), (1, 2)),
        parents={0: (), 1: (0,), 2: (1,)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: (
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(3, 10), Fraction(7, 10)),
            ),
            2: (
                (
                    Fraction(1, 10),
                    Fraction(2, 10),
                    Fraction(3, 10),
                    Fraction(4, 10),
                ),
                (
                    Fraction(4, 10),
                    Fraction(3, 10),
                    Fraction(2, 10),
                    Fraction(1, 10),
                ),
            ),
        },
    )


def _shared_root_separator_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="shared-root-separates-every-affected-mechanism",
        variables=("X", "S", "M", "Y"),
        domains=(2, 2, 2, 3),
        state_names=(
            ("0", "1"),
            ("0", "1"),
            ("0", "1"),
            ("0", "1", "2"),
        ),
        edges=((0, 2), (1, 2), (1, 3), (2, 3)),
        parents={0: (), 1: (), 2: (0, 1), 3: (1, 2)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(2, 5), Fraction(3, 5)),),
            2: (
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(7, 10), Fraction(3, 10)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
            3: (
                (Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)),
                (Fraction(1, 4), Fraction(7, 20), Fraction(2, 5)),
                (Fraction(3, 10), Fraction(2, 5), Fraction(3, 10)),
                (Fraction(7, 20), Fraction(9, 20), Fraction(1, 5)),
            ),
        },
    )


def _medium_cyclic_context_world() -> WorldSpec:
    """A four-context, three-state response block (81 explicit columns)."""

    return WorldSpec(
        family="test_dag",
        topology="X2-to-M2-and-X2-M2-to-Y3",
        variables=("X", "M", "Y"),
        domains=(2, 2, 3),
        state_names=(("0", "1"), ("0", "1"), ("0", "1", "2")),
        edges=((0, 1), (0, 2), (1, 2)),
        parents={0: (), 1: (0,), 2: (0, 1)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: (
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
            2: (
                (Fraction(7, 10), Fraction(1, 5), Fraction(1, 10)),
                (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5)),
                (Fraction(3, 10), Fraction(2, 5), Fraction(3, 10)),
                (Fraction(1, 10), Fraction(3, 10), Fraction(3, 5)),
            ),
        },
    )


def _tiny_dynamic_pricer_world() -> WorldSpec:
    """A tiny response block whose two endpoints both need missing columns."""

    return WorldSpec(
        family="test_dag",
        topology="tiny-dynamic-pricer",
        variables=("X", "M", "Y"),
        domains=(2, 2, 3),
        state_names=(("0", "1"), ("0", "1"), ("0", "1", "2")),
        edges=((0, 1), (0, 2), (1, 2)),
        parents={0: (), 1: (0,), 2: (0, 1)},
        cpt={
            0: ((Fraction(3, 10), Fraction(7, 10)),),
            1: (
                (Fraction(1, 5), Fraction(4, 5)),
                (Fraction(1, 2), Fraction(1, 2)),
            ),
            2: (
                (Fraction(1, 9), Fraction(4, 9), Fraction(4, 9)),
                (Fraction(8, 19), Fraction(7, 19), Fraction(4, 19)),
                (Fraction(2, 11), Fraction(8, 11), Fraction(1, 11)),
                (Fraction(7, 15), Fraction(7, 15), Fraction(1, 15)),
            ),
        },
    )


def _one_mediator_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="one-mediator-exact-elimination",
        variables=("X", "M", "Y"),
        domains=(2, 2, 2),
        state_names=(("0", "1"),) * 3,
        edges=((0, 1), (0, 2), (1, 2)),
        parents={0: (), 1: (0,), 2: (0, 1)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: (
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(3, 10), Fraction(7, 10)),
            ),
            2: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
        },
    )


def _two_mediator_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="two-mediator-exact-elimination",
        variables=("X", "A", "B", "Y"),
        domains=(2, 2, 2, 2),
        state_names=(("0", "1"),) * 4,
        edges=((0, 1), (0, 2), (1, 2), (0, 3), (2, 3)),
        parents={0: (), 1: (0,), 2: (0, 1), 3: (0, 2)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: (
                (Fraction(7, 10), Fraction(3, 10)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
            2: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
            3: (
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            ),
        },
    )


def _two_mediator_no_direct_terminal_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="two-mediator-no-direct-terminal-edge",
        variables=("X", "A", "B", "Y"),
        domains=(2, 2, 2, 2),
        state_names=(("0", "1"),) * 4,
        edges=((0, 1), (0, 2), (1, 2), (2, 3)),
        parents={0: (), 1: (0,), 2: (0, 1), 3: (2,)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: (
                (Fraction(7, 10), Fraction(3, 10)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
            2: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
            3: (
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
        },
    )


def _two_mediator_shared_parent_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="two-mediator-shared-parent-exact-elimination",
        variables=("X", "Z", "A", "B", "Y"),
        domains=(2, 2, 2, 2, 2),
        state_names=(("0", "1"),) * 5,
        edges=((0, 2), (1, 2), (0, 3), (2, 3), (0, 4), (3, 4)),
        parents={0: (), 1: (), 2: (0, 1), 3: (0, 2), 4: (0, 3)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(2, 5), Fraction(3, 5)),),
            2: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 4), Fraction(1, 4)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
            3: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
            4: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
        },
    )


def _two_mediator_disconnected_shared_context_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="two-mediator-disconnected-shared-contexts",
        variables=("X", "Z", "A", "B", "Y"),
        domains=(2, 2, 2, 2, 2),
        state_names=(("0", "1"),) * 5,
        edges=((0, 2), (1, 2), (1, 3), (2, 3), (0, 4), (3, 4)),
        parents={0: (), 1: (), 2: (0, 1), 3: (1, 2), 4: (0, 3)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(2, 5), Fraction(3, 5)),),
            2: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 4), Fraction(1, 4)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
            3: (
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
            4: (
                (Fraction(7, 10), Fraction(3, 10)),
                (Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
        },
    )


def _two_mediator_terminal_parent_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="two-mediator-terminal-first-and-second-parents",
        variables=("X", "A", "B", "Y"),
        domains=(2, 2, 2, 2),
        state_names=(("0", "1"),) * 4,
        edges=((0, 1), (1, 2), (0, 3), (1, 3), (2, 3)),
        parents={0: (), 1: (0,), 2: (1,), 3: (0, 1, 2)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: (
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
            2: (
                (Fraction(3, 4), Fraction(1, 4)),
                (Fraction(1, 4), Fraction(3, 4)),
            ),
            3: tuple(
                (Fraction(10 - state, 11), Fraction(1 + state, 11))
                for state in range(8)
            ),
        },
    )


def _solve_sparse_pair(
    world: WorldSpec,
    *,
    probability_message_bounds: bool,
    on_demand_response_columns: bool,
) -> tuple[_SparseResponseModel, float, float]:
    outer = tuple(
        map(
            float,
            interventional_frechet_transition_outer_bounds(
                world,
                0,
                2,
                treatment_value=1,
                baseline_value=0,
                outcome_state=0,
            ),
        )
    )
    owner = _SparseResponseModel(
        world,
        0,
        2,
        baseline_value=0,
        treatment_value=1,
        outcome_state=0,
        sense="minimize",
        target_outer_bounds=outer,
        probability_message_bounds=probability_message_bounds,
        on_demand_response_columns=on_demand_response_columns,
    )
    lower, _ = owner.optimize(time_limit_seconds=None)
    owner.restart_with_objective("maximize")
    upper, _ = owner.optimize(time_limit_seconds=None)
    return owner, lower, upper


def _contract_transport_message(
    domain_size: int,
    *,
    probability_message_bounds: bool,
) -> tuple[float, float, float, int, int]:
    """Run one affected-node contraction with the legacy or probability bound."""

    world = WorldSpec(
        family="test_dag",
        topology="isolated-pair-kernel",
        variables=("K", "O"),
        domains=(domain_size, 2),
        state_names=(tuple(str(state) for state in range(domain_size)), ("0", "1")),
        edges=(),
        parents={0: (), 1: ()},
        cpt={
            0: ((Fraction(1, domain_size),) * domain_size,),
            1: ((Fraction(1, 2), Fraction(1, 2)),),
        },
    )
    model = Model(f"transport-message-d{domain_size}")
    model.hideOutput()
    tokens = ((0, 0), (0, 1))
    kernel: dict[tuple[int, int], object] = {}
    downstream: dict[tuple[int, int], object] = {}
    kernel_upper: dict[tuple[int, int], float] = {}
    kernel_initial: dict[tuple[int, int], float] = {}
    downstream_upper: dict[tuple[int, int], float] = {}
    downstream_initial: dict[tuple[int, int], float] = {}
    for left in range(domain_size):
        for right in range(domain_size):
            pair = (left, right)
            kernel[pair] = model.addVar(lb=0.0, ub=1.0 / domain_size)
            downstream[pair] = model.addVar(lb=0.0, ub=1.0)
            kernel_upper[pair] = 1.0 / domain_size
            kernel_initial[pair] = float(left == right) / domain_size
            downstream_upper[pair] = 1.0
            downstream_initial[pair] = 0.5
    for left in range(domain_size):
        model.addCons(
            sum(kernel[(left, right)] for right in range(domain_size)) == 1.0 / domain_size
        )
    for right in range(domain_size):
        model.addCons(
            sum(kernel[(left, right)] for left in range(domain_size)) == 1.0 / domain_size
        )

    local = _SymbolicFactor(tokens, kernel, kernel_upper, kernel_initial)
    child = _SymbolicFactor(tokens, downstream, downstream_upper, downstream_initial)
    auxiliary_values: list[tuple[object, float]] = []
    result = _eliminate_factor_tokens(
        world,
        model,
        [child],
        tokens,
        local,
        outcome=1,
        outcome_state=1,
        auxiliary_values=auxiliary_values,
        probability_message_bounds=probability_message_bounds,
    )
    message = result[0].values[()]
    initial = result[0].initial_values[()]
    model.setObjective(message, "maximize")
    model.optimize()
    if str(model.getStatus()) != "optimal":
        raise AssertionError("local transport contraction did not solve")
    return (
        float(model.getVal(message)),
        float(message.getUbOriginal()),
        float(initial),
        model.getNVars(),
        model.getNConss(),
    )


class CounterfactualSolverOptimizationTests(unittest.TestCase):
    def test_epsilon_sharp_termination_returns_the_safe_dual_endpoint(self) -> None:
        for sense, primal, dual in (
            ("minimize", 0.251, 0.250),
            ("maximize", 0.749, 0.750),
        ):
            with self.subTest(sense=sense):
                owner = object.__new__(_SparseResponseModel)
                owner.model = MagicMock()
                owner.model.getStatus.return_value = "gaplimit"
                owner.model.getPrimalbound.return_value = primal
                owner.model.getDualbound.return_value = dual
                owner.pricer = MagicMock()
                owner.pricer.closed = True
                owner.pricer.timed_out = False
                owner.sense = sense

                endpoint, _ = owner.optimize(
                    time_limit_seconds=1.0,
                    accepted_absolute_gap=0.002,
                )

                self.assertEqual(endpoint, dual)
                self.assertEqual(owner.last_certification, "epsilon_sharp")
                self.assertAlmostEqual(owner.last_endpoint_error, 0.001)

    def test_epsilon_sharp_never_accepts_incomplete_pricing(self) -> None:
        owner = object.__new__(_SparseResponseModel)
        owner.model = MagicMock()
        owner.model.getStatus.return_value = "gaplimit"
        owner.model.getPrimalbound.return_value = 0.501
        owner.model.getDualbound.return_value = 0.5
        owner.pricer = MagicMock()
        owner.pricer.closed = False
        owner.pricer.timed_out = False
        owner.sense = "minimize"

        with self.assertRaisesRegex(RuntimeError, "pricing_closed=False"):
            owner.optimize(
                time_limit_seconds=1.0,
                accepted_absolute_gap=0.002,
            )

    def test_global_bound_bracket_uses_only_numerical_tolerance(self) -> None:
        self.assertTrue(_global_bounds_numerically_closed(0.3, 0.3 + 5e-9))
        self.assertFalse(_global_bounds_numerically_closed(0.3, 0.3 + 5e-8))
        self.assertFalse(_global_bounds_numerically_closed(float("inf"), 0.3))

    def test_terminal_event_quotient_preserves_categorical_marginals(self) -> None:
        world = _four_state_non_direct_terminal_world()
        quotient = _coarsen_terminal_event_outcome(world, 2, ((0,), (1,)))
        self.assertIsNotNone(quotient)
        quotient_world, quotient_events = quotient  # type: ignore[misc]
        self.assertEqual(quotient_world.domains[2], 3)
        self.assertEqual(quotient_events, ((0,), (1,)))
        self.assertEqual(
            quotient_world.cpt[2],
            (
                (Fraction(1, 10), Fraction(2, 10), Fraction(7, 10)),
                (Fraction(4, 10), Fraction(3, 10), Fraction(3, 10)),
            ),
        )

    def test_terminal_event_quotient_matches_full_categorical_reference(self) -> None:
        world = _four_state_non_direct_terminal_world()
        expected = reference_individual_counterfactual_probability_bounds(
            world,
            0,
            2,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            2,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        self.assertAlmostEqual(actual.lower, float(expected[0]), places=8)
        self.assertAlmostEqual(actual.upper, float(expected[1]), places=8)

    def test_shared_root_separator_matches_general_owner(self) -> None:
        world = _shared_root_separator_world()
        factual_probability = float(interventional_probability(world, {0: 0}, 3, 0))
        counterfactual_probability = float(
            interventional_probability(world, {0: 1}, 3, 1)
        )
        outer = (
            max(0.0, factual_probability + counterfactual_probability - 1.0),
            min(factual_probability, counterfactual_probability),
        )
        owner = _SparseResponseModel(
            world,
            0,
            3,
            baseline_value=0,
            treatment_value=1,
            outcome_state=None,
            outcome_events=((0,), (1,)),
            sense="minimize",
            target_outer_bounds=outer,
        )
        expected_lower, _ = owner.optimize(time_limit_seconds=None)
        owner.restart_with_objective("maximize")
        expected_upper, _ = owner.optimize(time_limit_seconds=None)

        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            3,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        self.assertAlmostEqual(
            actual.lower,
            expected_lower / factual_probability,
            places=8,
        )
        self.assertAlmostEqual(
            actual.upper,
            expected_upper / factual_probability,
            places=8,
        )
        self.assertEqual(actual.backend, "shared_root_separator_decomposition")

    def test_disjoint_terminal_lower_decomposition_matches_reference(self) -> None:
        world = _non_direct_terminal_world()
        expected = reference_individual_counterfactual_probability_bounds(
            world,
            0,
            2,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            2,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        self.assertEqual(actual.lower, 0.0)
        self.assertAlmostEqual(actual.lower, float(expected[0]), places=8)
        self.assertAlmostEqual(actual.upper, float(expected[1]), places=8)
        self.assertEqual(actual.backend, "terminal_event_endpoint_decomposition")

    def test_identical_terminal_upper_decomposition_matches_reference(self) -> None:
        world = _non_direct_terminal_world()
        expected = reference_individual_counterfactual_probability_bounds(
            world,
            0,
            2,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=1,
            target_outcome_state=1,
        )
        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            2,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=1,
            target_outcome_state=1,
        )
        self.assertAlmostEqual(actual.lower, float(expected[0]), places=8)
        self.assertAlmostEqual(actual.upper, float(expected[1]), places=8)
        self.assertEqual(actual.backend, "terminal_event_endpoint_decomposition")

    def test_uncertified_terminal_endpoint_is_rejected(self) -> None:
        world = _non_direct_terminal_world()
        with self.assertRaisesRegex(ValueError, "no joint response certificate"):
            _SparseResponseModel(
                world,
                0,
                2,
                baseline_value=0,
                treatment_value=1,
                outcome_state=None,
                outcome_events=((0,), (1,)),
                sense="maximize",
                target_outer_bounds=(0.0, 1.0),
                terminal_event_endpoint="upper",
            )

    def test_one_mediator_elimination_matches_full_response_enumeration(self) -> None:
        world = _one_mediator_world()
        expected = reference_individual_counterfactual_probability_bounds(
            world,
            0,
            2,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            2,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        self.assertAlmostEqual(actual.lower, float(expected[0]), places=8)
        self.assertAlmostEqual(actual.upper, float(expected[1]), places=8)
        self.assertEqual(actual.affected_nodes, 2)
        self.assertEqual(actual.auxiliary_variables, 0)

    def test_two_mediator_elimination_matches_full_response_enumeration(self) -> None:
        world = _two_mediator_world()
        expected = reference_individual_counterfactual_probability_bounds(
            world,
            0,
            3,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            3,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        self.assertAlmostEqual(actual.lower, float(expected[0]), places=8)
        self.assertAlmostEqual(actual.upper, float(expected[1]), places=8)
        self.assertEqual(actual.affected_nodes, 3)
        self.assertEqual(actual.auxiliary_variables, 0)

    def test_two_mediator_terminal_endpoint_matches_full_response_owner(self) -> None:
        world = _two_mediator_no_direct_terminal_world()
        expected = reference_individual_counterfactual_probability_bounds(
            world,
            0,
            3,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=1,
            target_outcome_state=1,
        )
        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            3,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=1,
            target_outcome_state=1,
        )
        self.assertAlmostEqual(actual.lower, float(expected[0]), places=8)
        self.assertAlmostEqual(actual.upper, float(expected[1]), places=8)
        self.assertIn("layered_endpoint", actual.backend)

    def test_two_mediator_shared_component_matches_general_exact_owner(self) -> None:
        world = _two_mediator_shared_parent_world()
        factual_probability = float(interventional_probability(world, {0: 0}, 4, 0))
        counterfactual_probability = float(
            interventional_probability(world, {0: 1}, 4, 1)
        )
        owner = _SparseResponseModel(
            world,
            0,
            4,
            baseline_value=0,
            treatment_value=1,
            outcome_state=None,
            outcome_events=((0,), (1,)),
            sense="minimize",
            target_outer_bounds=(
                max(0.0, factual_probability + counterfactual_probability - 1.0),
                min(factual_probability, counterfactual_probability),
            ),
        )
        expected_lower, _ = owner.optimize(time_limit_seconds=None)
        owner.restart_with_objective("maximize")
        expected_upper, _ = owner.optimize(time_limit_seconds=None)
        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            4,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        self.assertAlmostEqual(
            actual.lower,
            expected_lower / factual_probability,
            places=8,
        )
        self.assertAlmostEqual(
            actual.upper,
            expected_upper / factual_probability,
            places=8,
        )

    def test_terminal_first_and_second_parent_elimination_matches_general_owner(
        self,
    ) -> None:
        world = _two_mediator_terminal_parent_world()
        factual_probability = float(interventional_probability(world, {0: 0}, 3, 0))
        counterfactual_probability = float(
            interventional_probability(world, {0: 1}, 3, 1)
        )
        outer = (
            max(0.0, factual_probability + counterfactual_probability - 1.0),
            min(factual_probability, counterfactual_probability),
        )
        owner = _SparseResponseModel(
            world,
            0,
            3,
            baseline_value=0,
            treatment_value=1,
            outcome_state=None,
            outcome_events=((0,), (1,)),
            sense="minimize",
            target_outer_bounds=outer,
        )
        expected_lower, _ = owner.optimize(time_limit_seconds=None)
        owner.restart_with_objective("maximize")
        expected_upper, _ = owner.optimize(time_limit_seconds=None)

        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            3,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        self.assertAlmostEqual(
            actual.lower, expected_lower / factual_probability, places=8
        )
        self.assertAlmostEqual(
            actual.upper, expected_upper / factual_probability, places=8
        )

    def test_two_mediator_disconnected_contexts_match_general_exact_owner(self) -> None:
        world = _two_mediator_disconnected_shared_context_world()
        factual_probability = float(interventional_probability(world, {0: 0}, 4, 0))
        counterfactual_probability = float(
            interventional_probability(world, {0: 1}, 4, 1)
        )
        owner = _SparseResponseModel(
            world,
            0,
            4,
            baseline_value=0,
            treatment_value=1,
            outcome_state=None,
            outcome_events=((0,), (1,)),
            sense="minimize",
            target_outer_bounds=(
                max(0.0, factual_probability + counterfactual_probability - 1.0),
                min(factual_probability, counterfactual_probability),
            ),
        )
        expected_lower, _ = owner.optimize(time_limit_seconds=None)
        owner.restart_with_objective("maximize")
        expected_upper, _ = owner.optimize(time_limit_seconds=None)
        actual = sparse_individual_counterfactual_probability_bounds(
            world,
            0,
            4,
            factual_value=0,
            counterfactual_value=1,
            factual_outcome_state=0,
            target_outcome_state=1,
        )
        with patch(
            "cpt_world.counterfactual_solver._MAX_LAYERED_RESPONSE_COLUMNS",
            1,
        ):
            priced = sparse_individual_counterfactual_probability_bounds(
                world,
                0,
                4,
                factual_value=0,
                counterfactual_value=1,
                factual_outcome_state=0,
                target_outcome_state=1,
            )
        self.assertAlmostEqual(
            actual.lower, expected_lower / factual_probability, places=8
        )
        self.assertAlmostEqual(
            actual.upper, expected_upper / factual_probability, places=8
        )
        self.assertAlmostEqual(priced.lower, actual.lower, places=8)
        self.assertAlmostEqual(priced.upper, actual.upper, places=8)
        self.assertEqual(actual.backend, "two_mediator_layered_elimination")

    def test_priced_layered_response_matches_general_owner(self) -> None:
        world = _two_mediator_terminal_parent_world()
        factual_probability = float(interventional_probability(world, {0: 0}, 3, 0))
        counterfactual_probability = float(
            interventional_probability(world, {0: 1}, 3, 1)
        )
        outer = (
            max(0.0, factual_probability + counterfactual_probability - 1.0),
            min(factual_probability, counterfactual_probability),
        )
        owner = _SparseResponseModel(
            world,
            0,
            3,
            baseline_value=0,
            treatment_value=1,
            outcome_state=None,
            outcome_events=((0,), (1,)),
            sense="minimize",
            target_outer_bounds=outer,
        )
        expected_lower, _ = owner.optimize(time_limit_seconds=None)
        owner.restart_with_objective("maximize")
        expected_upper, _ = owner.optimize(time_limit_seconds=None)

        with patch(
            "cpt_world.counterfactual_solver._MAX_LAYERED_RESPONSE_COLUMNS",
            1,
        ):
            actual = sparse_individual_counterfactual_probability_bounds(
                world,
                0,
                3,
                factual_value=0,
                counterfactual_value=1,
                factual_outcome_state=0,
                target_outcome_state=1,
            )
        with (
            patch(
                "cpt_world.counterfactual_solver._MAX_LAYERED_RESPONSE_COLUMNS",
                1,
            ),
            patch(
                "cpt_world.counterfactual_solver._ExactPricedResponseLP."
                "transformed_upper_bound",
                return_value=(float("inf"), 0.0),
            ),
        ):
            unpruned = sparse_individual_counterfactual_probability_bounds(
                world,
                0,
                3,
                factual_value=0,
                counterfactual_value=1,
                factual_outcome_state=0,
                target_outcome_state=1,
            )
        self.assertAlmostEqual(
            actual.lower, expected_lower / factual_probability, places=8
        )
        self.assertAlmostEqual(
            actual.upper, expected_upper / factual_probability, places=8
        )
        self.assertAlmostEqual(actual.lower, unpruned.lower, places=8)
        self.assertAlmostEqual(actual.upper, unpruned.upper, places=8)
        self.assertEqual(actual.backend, "two_mediator_layered_elimination")
        self.assertEqual(actual.dynamic_response_blocks, 1)

    def test_farkas_pricing_does_not_erase_a_reduced_cost_certificate(self) -> None:
        pricer = _ResponsePricer([], {})
        pricer.begin_solve(deadline=None)
        self.assertTrue(pricer.closed)
        result = pricer._price(farkas=True)
        self.assertEqual(result["result"], SCIP_RESULT.SUCCESS)
        self.assertTrue(pricer.closed)

    def test_pairwise_pricing_can_exactly_exclude_existing_responses(self) -> None:
        unrestricted = _exact_pairwise_map(
            {0: (0.0, 1.0), 1: (0.0, 2.0)},
            {},
            domain_size=2,
            constant=0.0,
        )
        self.assertEqual(unrestricted.status, "optimal")
        self.assertEqual(unrestricted.response, (0, 0))
        excluded = _exact_pairwise_map(
            {0: (0.0, 1.0), 1: (0.0, 2.0)},
            {},
            domain_size=2,
            constant=0.0,
            forbidden_responses=frozenset({(0, 0)}),
        )
        self.assertEqual(excluded.status, "optimal")
        self.assertEqual(excluded.response, (1, 0))
        self.assertAlmostEqual(excluded.value, 1.0)

    def test_probability_bound_preserves_local_transport_contraction(self) -> None:
        for domain_size in (3, 5):
            with self.subTest(domain_size=domain_size):
                legacy = _contract_transport_message(
                    domain_size,
                    probability_message_bounds=False,
                )
                tightened = _contract_transport_message(
                    domain_size,
                    probability_message_bounds=True,
                )
                self.assertAlmostEqual(legacy[0], 1.0, delta=5e-8)
                self.assertAlmostEqual(tightened[0], legacy[0], delta=5e-8)
                self.assertAlmostEqual(tightened[2], legacy[2], places=12)
                self.assertEqual(tightened[3:], legacy[3:])
                self.assertAlmostEqual(legacy[1], float(domain_size), places=9)
                self.assertAlmostEqual(tightened[1], 1.0, places=9)

    def test_legacy_tightened_and_reference_endpoints_match(self) -> None:
        world = _uniform_multivalued_chain()
        expected = reference_counterfactual_transition_bounds(
            world,
            0,
            2,
            treatment_value=1,
            baseline_value=0,
            outcome_state=0,
        )
        legacy, legacy_lower, legacy_upper = _solve_sparse_pair(
            world,
            probability_message_bounds=False,
            on_demand_response_columns=False,
        )
        tightened, tightened_lower, tightened_upper = _solve_sparse_pair(
            world,
            probability_message_bounds=True,
            on_demand_response_columns=True,
        )

        self.assertAlmostEqual(legacy_lower, float(expected[0]), places=8)
        self.assertAlmostEqual(legacy_upper, float(expected[1]), places=8)
        self.assertAlmostEqual(tightened_lower, legacy_lower, places=8)
        self.assertAlmostEqual(tightened_upper, legacy_upper, places=8)
        self.assertEqual(len(tightened.auxiliary_values), len(legacy.auxiliary_values))
        self.assertEqual(tightened.model.getNVars(), legacy.model.getNVars())
        self.assertEqual(tightened.model.getNConss(), legacy.model.getNConss())
        self.assertAlmostEqual(tightened.initial_target, legacy.initial_target, places=12)
        self.assertGreater(
            max(variable.getUbOriginal() for variable, _ in legacy.auxiliary_values),
            1.0,
        )
        self.assertLessEqual(
            max(variable.getUbOriginal() for variable, _ in tightened.auxiliary_values),
            1.0,
        )

    def test_on_demand_columns_match_legacy_explicit_medium_block(self) -> None:
        world = _medium_cyclic_context_world()
        explicit, explicit_lower, explicit_upper = _solve_sparse_pair(
            world,
            probability_message_bounds=True,
            on_demand_response_columns=False,
        )
        on_demand, on_demand_lower, on_demand_upper = _solve_sparse_pair(
            world,
            probability_message_bounds=True,
            on_demand_response_columns=True,
        )

        self.assertEqual(len(explicit.dynamic_pricing_blocks), 0)
        self.assertGreater(len(on_demand.dynamic_pricing_blocks), 0)
        self.assertGreater(on_demand.pricer.generated_columns, 0)
        self.assertAlmostEqual(on_demand_lower, explicit_lower, places=8)
        self.assertAlmostEqual(on_demand_upper, explicit_upper, places=8)
        self.assertAlmostEqual(on_demand.initial_target, explicit.initial_target, places=12)

    def test_real_pricer_closes_missing_columns_for_both_endpoints(self) -> None:
        world = _tiny_dynamic_pricer_world()
        _, expected_lower, expected_upper = _solve_sparse_pair(
            world,
            probability_message_bounds=True,
            on_demand_response_columns=False,
        )
        outer = tuple(
            map(
                float,
                interventional_frechet_transition_outer_bounds(
                    world,
                    0,
                    2,
                    treatment_value=1,
                    baseline_value=0,
                    outcome_state=0,
                ),
            )
        )
        owner = _SparseResponseModel(
            world,
            0,
            2,
            baseline_value=0,
            treatment_value=1,
            outcome_state=0,
            sense="minimize",
            target_outer_bounds=outer,
        )

        lower, _ = owner.optimize(time_limit_seconds=None)
        lower_columns = owner.pricer.generated_columns
        self.assertTrue(owner.pricer.closed)
        self.assertFalse(owner.pricer.timed_out)
        self.assertTrue(all(round_.completed for round_ in owner.pricer.rounds))
        self.assertFalse(owner.pricer.rounds[-1].farkas)
        self.assertEqual(owner.pricer.rounds[-1].generated_columns, 0)
        self.assertGreater(lower_columns, 0)
        self.assertAlmostEqual(lower, expected_lower, delta=1e-8)

        owner.restart_with_objective("maximize")
        upper, _ = owner.optimize(time_limit_seconds=None)
        self.assertTrue(owner.pricer.closed)
        self.assertFalse(owner.pricer.timed_out)
        self.assertTrue(all(round_.completed for round_ in owner.pricer.rounds))
        self.assertFalse(owner.pricer.rounds[-1].farkas)
        self.assertEqual(owner.pricer.rounds[-1].generated_columns, 0)
        self.assertGreater(owner.pricer.generated_columns - lower_columns, 0)
        self.assertAlmostEqual(upper, expected_upper, places=8)

    def test_pricing_timeout_fails_closed(self) -> None:
        world = _tiny_dynamic_pricer_world()
        outer = tuple(
            map(
                float,
                interventional_frechet_transition_outer_bounds(
                    world,
                    0,
                    2,
                    treatment_value=1,
                    baseline_value=0,
                    outcome_state=0,
                ),
            )
        )
        timeout = _PairwiseMapOptimization(
            value=float("inf"),
            response=None,
            status="timelimit",
            solve_seconds=0.0,
            variables=0,
            constraints=0,
            backend="simulated_timeout",
        )
        owner = _SparseResponseModel(
            world,
            0,
            2,
            baseline_value=0,
            treatment_value=1,
            outcome_state=0,
            sense="minimize",
            target_outer_bounds=outer,
        )
        with (
            patch(
                "cpt_world.counterfactual_solver._exact_pairwise_map",
                return_value=timeout,
            ),
            self.assertRaisesRegex(RuntimeError, "pricing_closed=False"),
        ):
            owner.optimize(time_limit_seconds=1.0)
        self.assertTrue(owner.pricer.timed_out)
        self.assertFalse(owner.pricer.closed)
        self.assertTrue(any(not round_.completed for round_ in owner.pricer.rounds))

        with (
            patch(
                "cpt_world.counterfactual_solver._exact_pairwise_map",
                return_value=timeout,
            ),
            patch(
                "cpt_world.counterfactual_solver._one_mediator_joint_bounds",
                return_value=None,
            ),
            patch(
                "cpt_world.counterfactual_solver._direct_treatment_terminal_bounds",
                return_value=None,
            ),
            patch(
                "cpt_world.counterfactual_solver._partially_attainable_terminal_bounds",
                return_value=None,
            ),
            patch(
                "cpt_world.counterfactual_solver._coarsen_terminal_event_outcome",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "pricing_closed=False"),
        ):
            sparse_counterfactual_transition_bounds(
                world,
                0,
                2,
                treatment_value=1,
                baseline_value=0,
                outcome_state=0,
                time_limit_seconds=1.0,
            )


if __name__ == "__main__":
    unittest.main()
