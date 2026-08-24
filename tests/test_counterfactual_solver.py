from __future__ import annotations

import unittest
from fractions import Fraction
from unittest.mock import patch

from pyscipopt import SCIP_RESULT, Model

from cpt_world import (
    WorldSpec,
    reference_counterfactual_transition_bounds,
)
from cpt_world.counterfactual_solver import (
    _eliminate_factor_tokens,
    _exact_pairwise_map,
    _PairwiseMapOptimization,
    _ResponsePricer,
    _SparseResponseModel,
    _SymbolicFactor,
    sparse_counterfactual_transition_bounds,
)
from cpt_world.query_truth import interventional_frechet_transition_outer_bounds


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
        self.assertAlmostEqual(lower, expected_lower, places=8)

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
