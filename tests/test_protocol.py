from __future__ import annotations

import json
import unittest

from cpt_world import (
    Budget,
    Direction,
    EffectVector,
    EpisodeSampler,
    HardIntervention,
    InterventionCommand,
    OutcomeTape,
    TerminalAnswer,
    Variable,
    VisibleLayout,
    VisibleTask,
    build_candidate_episodes,
    compute_terminal_diagnostics,
    factorial_layouts,
    opaque_labels,
    parse_command,
    render_batch,
    render_batch_message,
    render_initial_messages,
    render_task_prompt,
    seed_by_id,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = factorial_layouts()[0]
        self.task = VisibleTask(self.layout)

    def test_opaque_labels_are_stable_and_avoid_ordinal_tokens(self) -> None:
        labels = opaque_labels(20260818)
        self.assertEqual(labels, opaque_labels(20260818))
        self.assertNotEqual(labels, opaque_labels(20260819))
        self.assertEqual(len(set("".join(labels))), 9)
        self.assertTrue(all(len(label) == 3 for label in labels))
        self.assertFalse(set("".join(labels)) & set("ABCXYZ"))

    def test_surface_factors_are_crossed_independently(self) -> None:
        layouts = factorial_layouts()
        self.assertEqual(len(layouts), 6 * 6 * 2)
        self.assertEqual(len({layout.layout_id for layout in layouts}), len(layouts))
        for role_labels in {
            (layout.first_label, layout.second_label, layout.isolated_label) for layout in layouts
        }:
            group = [
                layout
                for layout in layouts
                if (layout.first_label, layout.second_label, layout.isolated_label) == role_labels
            ]
            self.assertEqual(len({layout.target_order for layout in group}), 6)
            self.assertEqual({layout.reverse_effect_order for layout in group}, {False, True})

    def test_layout_owner_rejects_ordered_or_reused_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "opaque three-letter"):
            VisibleLayout(
                "bad-ordered",
                "ABC",
                "DEF",
                "GHI",
                tuple(Variable),
                False,
            )
        with self.assertRaisesRegex(ValueError, "must not reuse"):
            VisibleLayout(
                "bad-reused",
                "DDD",
                "EFG",
                "HIJ",
                tuple(Variable),
                False,
            )

    def test_prompt_hides_world_truth_and_cpt_values(self) -> None:
        prompt = render_task_prompt(self.task)
        for hidden_token in (
            "QN-EASY",
            "QN-MEDIUM",
            "QN-HARD",
            "0.9",
            "0.7",
            "0.55",
            "FORWARD",
            "REVERSE",
        ):
            self.assertNotIn(hidden_token, prompt)
        self.assertIn("Both fields are mandatory", prompt)
        self.assertIn("not confidence values", prompt)

    def test_complete_initial_messages_are_identical_across_hidden_truths(self) -> None:
        pair = build_candidate_episodes(layouts=(self.layout,))[:2]
        self.assertEqual({episode.world.direction for episode in pair}, set(Direction))
        messages = [render_initial_messages(episode.task) for episode in pair]
        self.assertEqual(messages[0], messages[1])
        rendered = json.dumps(messages[0])
        for episode in pair:
            self.assertNotIn(episode.episode_id, rendered)
            self.assertNotIn(episode.seed.seed_id, rendered)
            self.assertNotIn(episode.world.direction.value, rendered.lower())

    def test_terminal_answer_requires_and_maps_both_visible_fields(self) -> None:
        first = self.layout.first_label.lower()
        second = self.layout.second_label.lower()
        raw = json.dumps(
            {
                "type": "answer",
                f"effect_{first}_to_{second}": 0.4,
                f"effect_{second}_to_{first}": -0.1,
            }
        )
        command = parse_command(raw, self.task, remaining_rounds=1, remaining_samples=4)
        self.assertIsInstance(command, TerminalAnswer)
        self.assertEqual(command.effects, EffectVector(0.4, -0.1))

        missing = json.dumps({"type": "answer", f"effect_{first}_to_{second}": 0.4})
        with self.assertRaisesRegex(ValueError, "both effect fields"):
            parse_command(missing, self.task, remaining_rounds=1, remaining_samples=4)

    def test_intervention_maps_visible_target_to_canonical_role(self) -> None:
        visible = self.layout.labels[Variable.SECOND]
        raw = json.dumps({"type": "intervene", "target": visible, "value": 0, "batch_size": 8})
        command = parse_command(raw, self.task, remaining_rounds=2, remaining_samples=12)
        self.assertEqual(
            command,
            InterventionCommand(HardIntervention(Variable.SECOND, 0), 8),
        )

    def test_decoder_rejects_duplicates_bounds_and_illegal_budget(self) -> None:
        duplicate = '{"type":"answer","type":"answer"}'
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_command(duplicate, self.task, remaining_rounds=1, remaining_samples=4)

        first = self.layout.first_label.lower()
        second = self.layout.second_label.lower()
        out_of_range = json.dumps(
            {
                "type": "answer",
                f"effect_{first}_to_{second}": 1.1,
                f"effect_{second}_to_{first}": 0,
            }
        )
        with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
            parse_command(out_of_range, self.task, remaining_rounds=1, remaining_samples=4)

        intervention = json.dumps(
            {
                "type": "intervene",
                "target": self.layout.first_label,
                "value": 1,
                "batch_size": 4,
            }
        )
        with self.assertRaisesRegex(ValueError, "terminal answer"):
            parse_command(intervention, self.task, remaining_rounds=0, remaining_samples=4)

    def test_visible_batch_preserves_all_counts_and_requested_order(self) -> None:
        world = seed_by_id("QN-EASY").world(Direction.FORWARD)
        sampler = EpisodeSampler(world, OutcomeTape("visible"), Budget(1, 4, (4,)))
        batch = sampler.intervene(InterventionCommand(HardIntervention(Variable.FIRST, 1), 4))
        visible = render_batch(batch, self.layout)
        self.assertEqual(visible["n"], 4)
        self.assertEqual(sum(visible["joint_counts"].values()), 4)
        expected_prefix = f"{self.layout.labels[self.layout.target_order[0]]}="
        self.assertTrue(all(key.startswith(expected_prefix) for key in visible["joint_counts"]))

    def test_batch_message_binds_intervention_and_remaining_budget(self) -> None:
        world = seed_by_id("QN-EASY").world(Direction.FORWARD)
        sampler = EpisodeSampler(world, OutcomeTape("visible-message"), Budget(1, 4, (4,)))
        command = InterventionCommand(HardIntervention(Variable.FIRST, 1), 4)
        batch = sampler.intervene(command)
        message = render_batch_message(
            batch,
            self.task,
            remaining_rounds=sampler.remaining_rounds,
            remaining_samples=sampler.remaining_samples,
        )
        rendered, instruction = message.split("\n", 1)
        payload = json.loads(rendered)
        self.assertEqual(payload["type"], "batch_result")
        self.assertEqual(payload["intervention"]["target"], self.layout.first_label)
        self.assertEqual(payload["intervention"]["value"], 1)
        self.assertEqual(payload["batch"]["n"], 4)
        self.assertEqual(payload["remaining_rounds"], 0)
        self.assertEqual(payload["remaining_samples"], 0)
        self.assertIn("terminal answer", instruction)

    def test_batch_message_rejects_invalid_remaining_budget(self) -> None:
        world = seed_by_id("QN-EASY").world(Direction.FORWARD)
        batch = EpisodeSampler(world, OutcomeTape("invalid-budget")).intervene(
            InterventionCommand(HardIntervention(Variable.FIRST, 1), 4)
        )
        with self.assertRaisesRegex(ValueError, "remaining_rounds"):
            render_batch_message(
                batch,
                self.task,
                remaining_rounds=-1,
                remaining_samples=60,
            )
        forced_answer = render_batch_message(
            batch,
            self.task,
            remaining_rounds=1,
            remaining_samples=2,
        )
        self.assertIn("No intervention remains legal", forced_answer)

    def test_surface_decoder_connects_to_canonical_metrics_without_reinferring_roles(self) -> None:
        layout = factorial_layouts()[1]
        task = VisibleTask(layout)
        world = seed_by_id("QN-MEDIUM").world(Direction.REVERSE)
        first = layout.first_label.lower()
        second = layout.second_label.lower()
        raw = json.dumps(
            {
                "type": "answer",
                f"effect_{first}_to_{second}": 0.0,
                f"effect_{second}_to_{first}": 0.4,
            }
        )
        answer = parse_command(raw, task, remaining_rounds=0, remaining_samples=0)
        self.assertIsInstance(answer, TerminalAnswer)
        metrics = compute_terminal_diagnostics(
            {"episode": world.truth},
            {"episode": answer.effects},
        )
        self.assertEqual(metrics.vector_rmse, 0.0)
        self.assertEqual(metrics.active_mae, 0.0)
        self.assertEqual(metrics.inactive_mae, 0.0)


if __name__ == "__main__":
    unittest.main()
