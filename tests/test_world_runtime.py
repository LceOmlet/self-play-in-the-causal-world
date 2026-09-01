from __future__ import annotations

import json
import unittest
from fractions import Fraction

from cpt_world import (
    HIDING_MODES,
    Budget,
    OutcomeTape,
    WorldGrammar,
    WorldIntervention,
    WorldInterventionCommand,
    WorldObservationCommand,
    WorldSpec,
    WorldSpecEpisode,
    assemble_seed,
    backdoor_adjustment_sets,
    compute_query_truth,
    legal_query_anchors,
    sample_task_world,
    sample_worldspec_batch,
    worldspec_interventional_distribution,
)


def _sampled_task(
    query_type: str,
    *,
    preferred_seed: int | None = None,
    preferred_anchor: int = 0,
):
    grammar = WorldGrammar(node_counts=(2, 3, 4))
    candidate_seeds = (preferred_seed,) if preferred_seed is not None else tuple(range(200))
    for seed_number in candidate_seeds:
        # The task-conditioned world is the world-first owner.  Looking up
        # roles on an independently sampled, unconditioned world can select a
        # different DAG for the same integer seed.
        world = sample_task_world(grammar, seed_number, query_type)
        anchors_list = legal_query_anchors(world, query_type)
        if len(anchors_list) <= preferred_anchor:
            continue
        anchors = anchors_list[preferred_anchor]
        task_head = (
            "target_query"
            if query_type in {"ate", "individual_counterfactual_probability"}
            else "decision"
        )
        seed = assemble_seed(
            world,
            tuple(sorted(HIDING_MODES)),
            query_type,
            task_head,
            anchors=anchors,
            seed_id=f"RUNTIME-{seed_number}-{query_type}-a{preferred_anchor}",
        )
        return seed, world
    raise AssertionError(f"no sampled {query_type} task found")


def _visible_label(seed, internal_name: str) -> str:
    return str(seed["visible_schema"]["variable_labels"][internal_name])


def _batch_count_map(batch) -> dict[tuple[int, ...], int]:
    return dict(zip(batch.assignments, batch.counts, strict=True))


def _combined_count_map(*batches) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = {}
    for batch in batches:
        for assignment, count in _batch_count_map(batch).items():
            result[assignment] = result.get(assignment, 0) + count
    return result


def _deterministic_multivalue_chain() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="T3-to-M2-to-Y3",
        variables=("T", "M", "Y"),
        domains=(3, 2, 3),
        state_names=(("t0", "t1", "t2"), ("m0", "m1"), ("y0", "y1", "y2")),
        edges=((0, 1), (1, 2)),
        parents={0: (), 1: (0,), 2: (1,)},
        cpt={
            0: ((Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)),),
            1: ((Fraction(1), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
            2: ((Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(0), Fraction(1))),
        },
    )


def _ternary_fork_world() -> WorldSpec:
    copy_rows = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    return WorldSpec(
        family="test_dag",
        topology="C3-to-X3-and-Y3",
        variables=("C", "X", "Y"),
        domains=(3, 3, 3),
        state_names=(("0", "1", "2"),) * 3,
        edges=((0, 1), (0, 2)),
        parents={0: (), 1: (0,), 2: (0,)},
        cpt={
            0: ((Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)),),
            1: copy_rows,
            2: copy_rows,
        },
    )


def _backdoor_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="Z-to-X-Z-to-Y-X-to-Y",
        variables=("Z", "X", "Y"),
        domains=(2, 2, 2),
        state_names=(("z0", "z1"), ("x0", "x1"), ("y0", "y1")),
        edges=((0, 1), (0, 2), (1, 2)),
        parents={0: (), 1: (0,), 2: (0, 1)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(3, 4), Fraction(1, 4)), (Fraction(1, 4), Fraction(3, 4))),
            2: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
        },
    )


def _first_command(seed, world, *, batch_size: int = 8) -> str:
    target_name = next(name for name, allowed in seed["manipulability"].items() if allowed)
    measure_names = [
        name for name, readable in seed["readable"].items() if readable and name != target_name
    ][:2]
    return json.dumps(
        {
            "type": "intervene",
            "target": _visible_label(seed, target_name),
            "value": (
                "state_1" if world.domains[world.variables.index(target_name)] > 1 else "state_0"
            ),
            "measure": [_visible_label(seed, name) for name in measure_names],
            "batch_size": batch_size,
        },
        separators=(",", ":"),
    )


class WorldSpecRuntimeTests(unittest.TestCase):
    def test_interventional_distribution_replaces_the_target_mechanism(self) -> None:
        _, world = _sampled_task("ate", preferred_seed=64)
        target = 2
        state = 1
        distribution = worldspec_interventional_distribution(world, {target: state})
        self.assertEqual(sum(probability for _, probability in distribution), 1)
        self.assertEqual(
            sum(probability for values, probability in distribution if values[target] == state),
            1,
        )
        self.assertEqual(
            sum(probability for values, probability in distribution if values[target] != state),
            0,
        )

    def test_hard_do_is_not_conditioning_on_the_intervened_value(self) -> None:
        world = _ternary_fork_world()
        distribution = {
            assignment: probability
            for assignment, probability in worldspec_interventional_distribution(world, {1: 2})
            if probability
        }
        self.assertEqual(
            distribution,
            {
                (0, 2, 0): Fraction(1, 3),
                (1, 2, 1): Fraction(1, 3),
                (2, 2, 2): Fraction(1, 3),
            },
        )

    def test_parser_is_strict_visible_multivalued_and_budget_aware(self) -> None:
        seed, world = _sampled_task("ate", preferred_seed=64)
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-parser"),
            budget=Budget(max_observations=16),
            measure_max=2,
        )
        raw = _first_command(seed, world)
        command = episode.parse_intervention(raw)
        self.assertEqual(command.batch_size, 8)
        self.assertEqual(len(command.measure), 2)
        self.assertNotIn(command.intervention.target, command.measure)

        visible = json.loads(raw)
        invalid = dict(visible)
        invalid["target"] = world.variables[command.intervention.target]
        with self.assertRaises(ValueError):
            episode.parse_intervention(json.dumps(invalid))
        invalid = dict(visible)
        invalid["measure"] = [visible["target"]]
        with self.assertRaises(ValueError):
            episode.parse_intervention(json.dumps(invalid))
        invalid = dict(visible)
        invalid["value"] = "state_999"
        with self.assertRaises(ValueError):
            episode.parse_intervention(json.dumps(invalid))
        invalid = dict(visible)
        invalid["measure"] = visible["measure"] + [visible["measure"][0]]
        with self.assertRaises(ValueError):
            episode.parse_intervention(json.dumps(invalid))
        with self.assertRaises(ValueError):
            episode.parse_intervention(raw[:-1] + ',"target":"' + visible["target"] + '"}')
        for noncanonical_state in ("state_01", "state_١"):
            invalid = dict(visible)
            invalid["value"] = noncanonical_state
            with self.assertRaises(ValueError):
                episode.parse_intervention(json.dumps(invalid))
        with self.assertRaises(ValueError):
            episode.parse_intervention(raw.replace('"batch_size":8', '"batch_size":NaN'))

    def test_arbitrary_positive_batches_and_more_than_four_queries_are_legal(self) -> None:
        seed, world = _sampled_task("ate", preferred_seed=64)
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-unbounded-query-count"),
            budget=Budget(max_observations=17),
            measure_max=1,
        )
        command = json.loads(_first_command(seed, world))
        command["measure"] = command["measure"][:1]
        command["batch_size"] = 3
        episode.step(json.dumps(command))
        command["batch_size"] = 2
        for _ in range(7):
            episode.step(json.dumps(command))

        self.assertEqual(episode.queries_used, 8)
        self.assertEqual(episode.sample_rows_used, 17)
        self.assertEqual(episode.observations_used, 17)
        self.assertEqual(episode.remaining_budget, 0)
        command["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "terminal answer is required"):
            episode.parse_intervention(json.dumps(command))

    def test_query_cost_uses_batch_size_times_actual_measure_width(self) -> None:
        seed, world = _sampled_task("ate", preferred_seed=64)
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-observation-cost"),
            budget=Budget(max_observations=10),
            measure_max=2,
        )
        command = json.loads(_first_command(seed, world))
        command["batch_size"] = 5
        episode.step(json.dumps(command))
        self.assertEqual(episode.observations_used, 10)
        self.assertEqual(episode.remaining_budget, 0)

    def test_deterministic_multivalue_hard_do_and_selected_feedback(self) -> None:
        world = _deterministic_multivalue_chain()
        seed = assemble_seed(
            world,
            tuple(sorted(HIDING_MODES)),
            "mediator_set",
            "discovery",
            anchors={"treatment": 0, "outcome": 2},
            seed_id="RUNTIME-MULTIVALUE-CHAIN",
            # This pinned runtime fixture exercises a hard intervention on T;
            # main-pipeline tasks still draw K through the shared sampler.
            manipulability={"T": True, "M": False, "Y": False},
        )
        labels = seed["visible_schema"]["variable_labels"]
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-deterministic-chain"),
            budget=Budget(max_observations=8),
            measure_max=2,
        )
        command = json.dumps(
            {
                "type": "intervene",
                "target": labels["T"],
                "value": "state_2",
                "measure": [labels["M"], labels["Y"]],
                "batch_size": 4,
            }
        )
        step = episode.step(command)
        self.assertEqual(step.batch.assignments, ((1, 2),))
        self.assertEqual(step.batch.counts, (4,))
        self.assertEqual(step.batch.count((0, 0)), 0)
        payload = json.loads(str(step.message).splitlines()[0])
        histogram = payload["batch"]["joint_histogram"]
        self.assertEqual(histogram["columns"], [labels["M"], labels["Y"]])
        self.assertEqual(histogram["rows"], [[[1, 2], 4]])
        self.assertEqual(payload["remaining_budget"], 0)
        self.assertIn("Return a terminal answer now", str(step.message))

        truth = compute_query_truth(world, seed)
        answer = json.dumps(
            {
                "type": "answer",
                "mediators": [labels[name] for name in truth["mediators"]],
                "order": [[labels[left], labels[right]] for left, right in truth["order"]],
            }
        )
        terminal = episode.step(answer)
        self.assertEqual(terminal.score["mediator_f1"], 1)
        self.assertEqual(terminal.score["order_f1"], 1)
        self.assertEqual(terminal.reward, 1)

    def test_action_keyed_stream_is_split_interleave_and_measure_invariant(self) -> None:
        seed, world = _sampled_task("ate", preferred_seed=64)
        target_name = next(name for name, allowed in seed["manipulability"].items() if allowed)
        target = world.variables.index(target_name)
        other_nodes = tuple(node for node in range(len(world.variables)) if node != target)
        full_command = WorldInterventionCommand(
            WorldIntervention(target, 1),
            other_nodes,
            16,
        )
        half_command = WorldInterventionCommand(
            WorldIntervention(target, 1),
            other_nodes,
            8,
        )
        tape = OutcomeTape("runtime-pairing")
        full = sample_worldspec_batch(world, tape, full_command, start_index=0)
        first = sample_worldspec_batch(world, tape, half_command, start_index=0)
        second = sample_worldspec_batch(world, tape, half_command, start_index=8)
        self.assertEqual(_batch_count_map(full), _combined_count_map(first, second))

        selected_command = WorldInterventionCommand(
            WorldIntervention(target, 1),
            (other_nodes[0],),
            16,
        )
        selected = sample_worldspec_batch(world, tape, selected_command, start_index=0)
        marginalized: dict[tuple[int, ...], int] = {}
        for selected_state in range(world.domains[other_nodes[0]]):
            count = sum(
                count
                for assignment, count in _batch_count_map(full).items()
                if assignment[0] == selected_state
            )
            if count:
                marginalized[(selected_state,)] = count
        self.assertEqual(_batch_count_map(selected), marginalized)

        episode_a = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-order"),
            budget=Budget(max_observations=128),
        )
        episode_b = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-order"),
            budget=Budget(max_observations=128),
        )
        primary = half_command
        alternate = WorldInterventionCommand(
            WorldIntervention(target, 0),
            other_nodes,
            8,
        )
        a_primary = episode_a.intervene(primary)
        episode_a.intervene(alternate)
        episode_b.intervene(alternate)
        b_primary = episode_b.intervene(primary)
        self.assertEqual(_batch_count_map(a_primary), _batch_count_map(b_primary))

    def test_passive_observation_uses_natural_law_and_its_own_split_stable_stream(self) -> None:
        world = _ternary_fork_world()
        tape = OutcomeTape("runtime-natural-observation")
        full_command = WorldObservationCommand(measure=(0, 1, 2), batch_size=16)
        half_command = WorldObservationCommand(measure=(0, 1, 2), batch_size=8)
        full = sample_worldspec_batch(world, tape, full_command, start_index=0)
        first = sample_worldspec_batch(world, tape, half_command, start_index=0)
        second = sample_worldspec_batch(world, tape, half_command, start_index=8)

        self.assertIsNone(full.intervention)
        self.assertEqual(_batch_count_map(full), _combined_count_map(first, second))
        for assignment, count in zip(full.assignments, full.counts, strict=True):
            if count:
                self.assertEqual(assignment[0], assignment[1])
                self.assertEqual(assignment[0], assignment[2])

    def test_seed_owned_observation_bandwidth_and_observe_protocol(self) -> None:
        world = _deterministic_multivalue_chain()
        seed = assemble_seed(
            world,
            tuple(sorted(HIDING_MODES)),
            "mediator_set",
            "discovery",
            anchors={"treatment": 0, "outcome": 2},
            seed_id="RUNTIME-OBSERVE",
            observation_bandwidth=2,
        )
        labels = seed["visible_schema"]["variable_labels"]
        budget = Budget(max_observations=16)
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-observe-episode"),
            budget=budget,
        )
        prompt = episode.initial_messages()[1]["content"]
        self.assertIn('{"type":"observe"', prompt)
        self.assertIn("natural distribution", prompt)
        self.assertIn("At most 2 variables", prompt)
        with self.assertRaisesRegex(ValueError, "cannot override"):
            WorldSpecEpisode(
                world,
                seed,
                OutcomeTape("runtime-observe-override"),
                budget=budget,
                measure_max=1,
            )

        command = {
            "type": "observe",
            "measure": [labels["T"], labels["Y"]],
            "batch_size": 4,
        }
        parsed = episode.parse_observation(json.dumps(command))
        self.assertEqual(parsed.measure, (0, 2))
        invalid = {**command, "target": labels["M"]}
        with self.assertRaises(ValueError):
            episode.parse_observation(json.dumps(invalid))
        invalid = {**command, "measure": [labels["T"], labels["M"], labels["Y"]]}
        with self.assertRaises(ValueError):
            episode.parse_observation(json.dumps(invalid))

        step = episode.step(json.dumps(command))
        self.assertEqual(step.kind, "batch")
        self.assertIsNone(step.batch.intervention)
        payload = json.loads(str(step.message).splitlines()[0])
        self.assertEqual(payload["experiment"], {"type": "observe"})
        self.assertNotIn("intervention", payload)
        histogram = payload["batch"]["joint_histogram"]
        self.assertEqual(histogram["columns"], [labels["T"], labels["Y"]])
        self.assertEqual(sum(row[1] for row in histogram["rows"]), 4)
        self.assertEqual(episode.queries_used, 1)
        self.assertEqual(episode.sample_rows_used, 4)
        self.assertEqual(episode.observations_used, 8)

    def test_seed_owned_power_of_two_budget_is_used_by_default(self) -> None:
        world = _deterministic_multivalue_chain()
        seed = assemble_seed(
            world,
            tuple(sorted(HIDING_MODES)),
            "mediator_set",
            "discovery",
            anchors={"treatment": 0, "outcome": 2},
            seed_id="RUNTIME-POWER-OF-TWO-BUDGET",
            observation_bandwidth=2,
            observation_budget_exponent=13,
        )
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-power-of-two-budget"),
        )

        self.assertEqual(episode.budget.max_observations, 2 * 2**13)
        prompt = episode.initial_messages()[1]["content"]
        self.assertIn("2 x 2^13 = 2 x 8192", prompt)

    def test_legacy_seed_budget_defaults_to_two_to_the_eleventh(self) -> None:
        world = _deterministic_multivalue_chain()
        seed = assemble_seed(
            world,
            tuple(sorted(HIDING_MODES)),
            "mediator_set",
            "discovery",
            anchors={"treatment": 0, "outcome": 2},
            seed_id="RUNTIME-LEGACY-BUDGET",
            observation_bandwidth=2,
        )
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-legacy-budget"),
        )

        self.assertEqual(episode.budget.max_observations, 2 * 2**11)

    def test_feedback_contains_only_selected_visible_joint_counts(self) -> None:
        seed, world = _sampled_task("ate", preferred_seed=64)
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-feedback"),
            budget=Budget(max_observations=16),
            measure_max=1,
        )
        raw = json.loads(_first_command(seed, world))
        raw["measure"] = raw["measure"][:1]
        step = episode.step(json.dumps(raw))
        self.assertEqual(step.kind, "batch")
        self.assertIsNotNone(step.message)
        payload = json.loads(str(step.message).splitlines()[0])
        histogram = payload["batch"]["joint_histogram"]
        self.assertEqual(histogram["columns"], raw["measure"])
        self.assertEqual(payload["batch"]["n"], 8)
        self.assertEqual(sum(row[1] for row in histogram["rows"]), 8)
        unselected = {
            label
            for label in seed["visible_schema"]["variable_labels"].values()
            if label not in raw["measure"]
        }
        self.assertTrue(all(label not in histogram["columns"] for label in unselected))
        for internal_name in world.variables:
            self.assertNotIn(internal_name, step.message)
        self.assertEqual(payload["remaining_budget"], 8)

    def test_feedback_cell_ceiling_rejects_before_sampling_or_budget_use(self) -> None:
        world = sample_task_world(
            WorldGrammar(node_counts=(8,), max_domain_size=2),
            0,
            "ate",
        )
        seed = assemble_seed(
            world,
            tuple(sorted(HIDING_MODES)),
            "ate",
            "target_query",
            anchors=legal_query_anchors(world, "ate")[0],
            seed_id="RUNTIME-FEEDBACK-CELL-CEILING",
        )
        labels = seed["visible_schema"]["variable_labels"]
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-feedback-cell-ceiling"),
            budget=Budget(max_observations=2048),
        )
        command = {
            "type": "observe",
            "measure": [labels[name] for name in world.variables],
            "batch_size": 129,
        }

        with self.assertRaisesRegex(ValueError, "feedback cell bound exceeds 128"):
            episode.step(json.dumps(command))

        self.assertEqual(episode.queries_used, 0)
        self.assertEqual(episode.sample_rows_used, 0)
        self.assertEqual(episode.observations_used, 0)
        self.assertEqual(episode.remaining_budget, 2048)
        self.assertEqual(episode.history, ())

    def test_numeric_and_decision_tasks_run_from_prompt_to_terminal_score(self) -> None:
        for query_type in ("ate", "individual_counterfactual_probability", "best_intervention"):
            seed, world = _sampled_task(query_type)
            episode = WorldSpecEpisode(
                world,
                seed,
                OutcomeTape(f"runtime-mode-{query_type}"),
                budget=Budget(max_observations=16),
                measure_max=2,
            )
            initial = episode.initial_messages()
            self.assertIn("DOLENS HIDDEN-MECHANISM TASK", initial[1]["content"])
            if query_type == "best_intervention":
                self.assertIn(
                    '{"type":"answer","value":"state_i"}',
                    initial[1]["content"],
                )
                self.assertNotIn('"values"', initial[1]["content"])
                decision_label = seed["query"]["decision_target"]
                measure_label = next(
                    label
                    for internal, label in seed["visible_schema"]["variable_labels"].items()
                    if seed["readable"][internal] and label != decision_label
                )
                with self.assertRaisesRegex(ValueError, "not manipulable"):
                    episode.parse_intervention(
                        json.dumps(
                            {
                                "type": "intervene",
                                "target": decision_label,
                                "value": "state_0",
                                "measure": [measure_label],
                                "batch_size": 4,
                            }
                        )
                    )
            batch_step = episode.step(_first_command(seed, world))
            self.assertEqual(batch_step.kind, "batch")
            self.assertEqual(episode.queries_used, 1)
            truth = compute_query_truth(world, seed)
            if query_type == "best_intervention":
                raw_answer = json.dumps(
                    {
                        "type": "answer",
                        "value": f"state_{truth['value']}",
                    }
                )
            elif query_type == "ate":
                raw_answer = json.dumps(
                    {
                        "type": "answer",
                        "effect": {
                            f"state_{state}": float(component)
                            for state, component in enumerate(truth["effect"])
                        },
                    }
                )
            else:
                raw_answer = json.dumps(
                    {
                        "type": "answer",
                        "lower": float(truth["lower"]),
                        "upper": float(truth["upper"]),
                    }
                )
            terminal_step = episode.step(raw_answer)
            self.assertEqual(terminal_step.kind, "terminal")
            self.assertTrue(episode.completed)
            from cpt_world import terminal_quality_reward

            self.assertEqual(terminal_step.reward, terminal_quality_reward(terminal_step.score))
            self.assertEqual(episode.terminal_reward, terminal_step.reward)
            if query_type == "best_intervention":
                self.assertEqual(terminal_step.score["regret"], 0)
            elif query_type == "ate":
                self.assertLess(terminal_step.score["l1_error"], Fraction(1, 10**12))
            else:
                self.assertLess(
                    terminal_step.score["mean_absolute_endpoint_error"],
                    Fraction(1, 10**12),
                )
            with self.assertRaises(ValueError):
                episode.step(raw_answer)

    def test_individual_counterfactual_probability_runs_to_terminal_score(self) -> None:
        seed, world = _sampled_task("individual_counterfactual_probability")
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-individual-counterfactual-probability"),
            budget=Budget(max_observations=16),
            measure_max=2,
        )
        self.assertIn(
            "q has a sharp identified interval [lower, upper]",
            episode.initial_messages()[1]["content"],
        )
        truth = compute_query_truth(world, seed)
        terminal = episode.step(
            json.dumps(
                {
                    "type": "answer",
                    "lower": float(truth["lower"]),
                    "upper": float(truth["upper"]),
                }
            )
        )
        self.assertEqual(terminal.kind, "terminal")
        self.assertLess(terminal.score["mean_absolute_endpoint_error"], Fraction(1, 10**12))

    def test_backdoor_discovery_runs_from_intervention_to_exact_terminal_score(self) -> None:
        world = _backdoor_world()
        seed = assemble_seed(
            world,
            tuple(sorted(HIDING_MODES)),
            "backadj_minimal_sets",
            "discovery",
            anchors={"treatment": 1, "outcome": 2},
            seed_id="RUNTIME-BACKDOOR",
        )
        labels = seed["visible_schema"]["variable_labels"]
        episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape("runtime-backdoor"),
            budget=Budget(max_observations=8),
            measure_max=2,
        )
        step = episode.step(
            json.dumps(
                {
                    "type": "intervene",
                    "target": labels["Z"],
                    "value": "state_1",
                    "measure": [labels["X"], labels["Y"]],
                    "batch_size": 4,
                }
            )
        )
        self.assertEqual(step.kind, "batch")
        treatment = world.variables.index("X")
        outcome = world.variables.index("Y")
        adjustment_set = backdoor_adjustment_sets(world, treatment, outcome)[0]
        terminal = episode.step(
            json.dumps(
                {
                    "type": "answer",
                    "adjustment_set": [labels[name] for name in adjustment_set],
                }
            )
        )
        self.assertEqual(terminal.score["edit_distance"], 0)
        self.assertEqual(terminal.reward, 1)

    def test_episode_rejects_a_reward_graph_bound_smaller_than_its_world(self) -> None:
        seed, world = _sampled_task("ate")
        with self.assertRaisesRegex(ValueError, "covering the episode world"):
            WorldSpecEpisode(
                world,
                seed,
                OutcomeTape("runtime-invalid-reward-graph-bound"),
                max_graph_nodes=len(world.variables) - 1,
            )

    def test_episode_rejects_an_action_surface_with_no_target_measure_pair(self) -> None:
        seed, world = _sampled_task("ate", preferred_seed=64)
        only_name = world.variables[0]
        bad_seed = {
            **seed,
            "manipulability": {name: name == only_name for name in world.variables},
            "readable": {name: name == only_name for name in world.variables},
        }
        with self.assertRaisesRegex(ValueError, "no legal target/measure pair"):
            WorldSpecEpisode(
                world,
                bad_seed,
                OutcomeTape("runtime-impossible-actions"),
                budget=Budget(max_observations=8),
            )


if __name__ == "__main__":
    unittest.main()
