from __future__ import annotations

import unittest
from fractions import Fraction

from cpt_world import (
    ASSIGNMENTS,
    Budget,
    Direction,
    EpisodeSampler,
    HardIntervention,
    InterventionCommand,
    OutcomeTape,
    Variable,
    build_candidate_episodes,
    factorial_layouts,
    interventional_distribution,
    sample_batch,
    seed_by_id,
)


def marginal_one(distribution, component: int) -> Fraction:
    return sum(
        (mass for assignment, mass in distribution if assignment[component] == 1),
        start=Fraction(0),
    )


class ExactWorldTests(unittest.TestCase):
    def test_seed_parameters_and_truths_are_exact(self) -> None:
        expected = {
            "QN-EASY": (Fraction(1, 10), Fraction(9, 10), Fraction(4, 5)),
            "QN-MEDIUM": (Fraction(3, 10), Fraction(7, 10), Fraction(2, 5)),
            "QN-HARD": (Fraction(9, 20), Fraction(11, 20), Fraction(1, 10)),
        }
        for seed_id, (low, high, effect) in expected.items():
            seed = seed_by_id(seed_id)
            self.assertEqual(Fraction(1, 2) - seed.effect, low)
            self.assertEqual(Fraction(1, 2) + seed.effect, high)
            self.assertEqual(seed.active_effect, effect)
            self.assertEqual(
                seed.world(Direction.FORWARD).truth.effects.first_to_second,
                float(effect),
            )
            self.assertEqual(
                seed.world(Direction.REVERSE).truth.effects.second_to_first,
                float(effect),
            )

    def test_forward_and_reverse_are_observationally_indistinguishable(self) -> None:
        for seed_id in ("QN-EASY", "QN-MEDIUM", "QN-HARD"):
            seed = seed_by_id(seed_id)
            forward = interventional_distribution(seed.world(Direction.FORWARD))
            reverse = interventional_distribution(seed.world(Direction.REVERSE))
            self.assertEqual(forward, reverse)
            self.assertEqual(sum((mass for _, mass in forward), Fraction(0)), 1)

    def test_hard_do_replaces_the_target_mechanism(self) -> None:
        role_index = {
            Variable.FIRST: 0,
            Variable.SECOND: 1,
            Variable.ISOLATED: 2,
        }
        for seed_id in ("QN-EASY", "QN-MEDIUM", "QN-HARD"):
            seed = seed_by_id(seed_id)
            for direction in Direction:
                world = seed.world(direction)
                active_source, active_outcome = (
                    (Variable.FIRST, 1) if direction is Direction.FORWARD else (Variable.SECOND, 0)
                )
                inactive_source, inactive_outcome = (
                    (Variable.SECOND, 0) if direction is Direction.FORWARD else (Variable.FIRST, 1)
                )

                active_zero = interventional_distribution(world, HardIntervention(active_source, 0))
                active_one = interventional_distribution(world, HardIntervention(active_source, 1))
                self.assertEqual(
                    marginal_one(active_one, active_outcome)
                    - marginal_one(active_zero, active_outcome),
                    seed.active_effect,
                )

                inactive_zero = interventional_distribution(
                    world, HardIntervention(inactive_source, 0)
                )
                inactive_one = interventional_distribution(
                    world, HardIntervention(inactive_source, 1)
                )
                self.assertEqual(
                    marginal_one(inactive_one, inactive_outcome)
                    - marginal_one(inactive_zero, inactive_outcome),
                    0,
                )

                for target in Variable:
                    for value in (0, 1):
                        distribution = interventional_distribution(
                            world, HardIntervention(target, value)
                        )
                        self.assertEqual(
                            sum((mass for _, mass in distribution), Fraction(0)),
                            1,
                        )
                        for assignment, mass in distribution:
                            if assignment[role_index[target]] != value:
                                self.assertEqual(mass, 0)

    def test_action_keyed_tape_is_batch_split_invariant(self) -> None:
        world = seed_by_id("QN-EASY").world(Direction.FORWARD)
        tape = OutcomeTape("paired-episode")
        intervention = HardIntervention(Variable.FIRST, 1)
        whole = sample_batch(world, tape, intervention, start_index=0, sample_count=8)
        left = sample_batch(world, tape, intervention, start_index=0, sample_count=4)
        right = sample_batch(world, tape, intervention, start_index=4, sample_count=4)
        self.assertEqual(
            whole.counts,
            tuple(a + b for a, b in zip(left.counts, right.counts, strict=True)),
        )

    def test_arm_streams_do_not_depend_on_action_order(self) -> None:
        world = seed_by_id("QN-HARD").world(Direction.REVERSE)
        tape = OutcomeTape("action-order")
        first_arm = HardIntervention(Variable.FIRST, 1)
        second_arm = HardIntervention(Variable.SECOND, 0)
        direct = sample_batch(world, tape, first_arm, start_index=0, sample_count=8)
        sample_batch(world, tape, second_arm, start_index=0, sample_count=8)
        after_other_arm = sample_batch(world, tape, first_arm, start_index=0, sample_count=8)
        self.assertEqual(direct, after_other_arm)

    def test_episode_sampler_owns_arm_offsets_and_budget(self) -> None:
        world = seed_by_id("QN-MEDIUM").world(Direction.FORWARD)
        sampler = EpisodeSampler(world, OutcomeTape("state"), Budget(2, 8, (4,)))
        command = InterventionCommand(HardIntervention(Variable.FIRST, 1), 4)
        first = sampler.intervene(command)
        second = sampler.intervene(command)
        self.assertEqual(first.start_index, 0)
        self.assertEqual(second.start_index, 4)
        self.assertEqual(sampler.samples_used, 8)
        self.assertEqual(sampler.rounds_used, 2)
        self.assertEqual(len(sampler.history), 2)
        with self.assertRaisesRegex(ValueError, "budget is exhausted"):
            sampler.intervene(command)

    def test_candidate_builder_pairs_truths_on_the_same_tape(self) -> None:
        layouts = factorial_layouts()[:2]
        episodes = build_candidate_episodes(layouts=layouts)
        self.assertEqual(len(episodes), 3 * 2 * 2)
        grouped: dict[tuple[str, str], list] = {}
        for episode in episodes:
            key = (episode.seed.seed_id, episode.task.layout.layout_id)
            grouped.setdefault(key, []).append(episode)
        for pair in grouped.values():
            self.assertEqual({item.world.direction for item in pair}, set(Direction))
            self.assertEqual(len({item.tape_key for item in pair}), 1)
        self.assertEqual(len({item.tape_key for item in episodes}), 1)
        self.assertEqual(len({item.episode_id for item in episodes}), len(episodes))
        for episode in episodes:
            self.assertTrue(episode.episode_id.startswith("episode-"))
            self.assertNotIn(episode.seed.seed_id, episode.episode_id)
            self.assertNotIn(episode.world.direction.value, episode.episode_id)

    def test_surface_variants_share_canonical_outcomes(self) -> None:
        episodes = build_candidate_episodes(layouts=factorial_layouts()[:2])
        pair = [
            episode
            for episode in episodes
            if episode.seed.seed_id == "QN-EASY" and episode.world.direction is Direction.FORWARD
        ]
        self.assertEqual(len(pair), 2)
        command = InterventionCommand(HardIntervention(Variable.FIRST, 1), 4)
        batches = [EpisodeSampler(item.world, item.tape).intervene(command) for item in pair]
        self.assertEqual(batches[0].counts, batches[1].counts)

        next_replicate = build_candidate_episodes(
            layouts=factorial_layouts()[:1],
            replicate_id="r1",
        )
        self.assertNotEqual(pair[0].tape_key, next_replicate[0].tape_key)

    def test_assignment_order_is_complete_and_stable(self) -> None:
        self.assertEqual(len(ASSIGNMENTS), 8)
        self.assertEqual(ASSIGNMENTS[0], (0, 0, 0))
        self.assertEqual(ASSIGNMENTS[-1], (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
