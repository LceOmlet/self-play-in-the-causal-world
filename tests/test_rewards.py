from __future__ import annotations

import itertools
import unittest
from fractions import Fraction

from cpt_world import (
    TERMINAL_QUALITY_REWARD_VERSION,
    UNFINISHED_TERMINAL_QUALITY,
    soft_adjustment_family_f1,
    terminal_quality_reward,
)


class TerminalQualityRewardTests(unittest.TestCase):
    def test_reward_contract_version_and_unfinished_value_are_frozen(self) -> None:
        self.assertEqual(TERMINAL_QUALITY_REWARD_VERSION, "terminal-quality-v4")
        self.assertEqual(UNFINISHED_TERMINAL_QUALITY, 0)

    def test_numeric_and_decision_rewards_use_owner_diagnostics(self) -> None:
        self.assertEqual(
            terminal_quality_reward({"kind": "target_query", "l1_error": Fraction(1, 2)}),
            Fraction(7, 8),
        )
        self.assertEqual(
            terminal_quality_reward(
                {
                    "kind": "counterfactual_roi",
                    "mean_absolute_endpoint_error": Fraction(1, 5),
                }
            ),
            Fraction(4, 5),
        )
        self.assertEqual(
            terminal_quality_reward(
                {
                    "kind": "decision",
                    "regret": Fraction(7, 10),
                    "normalized_regret": Fraction(2, 5),
                }
            ),
            Fraction(3, 5),
        )
        self.assertEqual(
            terminal_quality_reward(
                {"kind": "decision", "regret": Fraction(0), "normalized_regret": 0}
            ),
            1,
        )
        self.assertEqual(
            terminal_quality_reward(
                {"kind": "decision", "regret": Fraction(1, 1000), "normalized_regret": 1}
            ),
            0,
        )

    def test_binary_vector_reward_equals_the_previous_scalar_reward(self) -> None:
        scalar_absolute_error = Fraction(3, 10)
        vector_l1_error = 2 * scalar_absolute_error
        self.assertEqual(
            terminal_quality_reward({"kind": "target_query", "l1_error": vector_l1_error}),
            1 - scalar_absolute_error / 2,
        )

    def test_mediator_reward_preserves_both_partial_f1_signals(self) -> None:
        score = {
            "kind": "mediator",
            "mediator_f1": Fraction(2, 3),
            "order_f1": Fraction(1, 2),
        }
        self.assertEqual(terminal_quality_reward(score), Fraction(7, 12))

    def test_soft_backdoor_reward_preserves_within_set_overlap(self) -> None:
        predicted = (("A",),)
        truth = (("A", "B"),)
        self.assertEqual(soft_adjustment_family_f1(predicted, truth), Fraction(2, 3))
        self.assertEqual(
            terminal_quality_reward({"kind": "backadj", "prediction": predicted, "truth": truth}),
            Fraction(2, 3),
        )

    def test_soft_backdoor_reward_uses_one_to_one_family_matching(self) -> None:
        predicted = (("A",),)
        truth = (("A",), ("A", "B"))
        self.assertEqual(soft_adjustment_family_f1(predicted, truth), Fraction(2, 3))

    def test_soft_backdoor_reward_is_one_exactly_for_equal_families(self) -> None:
        subsets = ((), ("A",), ("B",), ("A", "B"))
        families = tuple(
            family for size in range(3) for family in itertools.combinations(subsets, size)
        )
        for predicted in families:
            for truth in families:
                quality = soft_adjustment_family_f1(predicted, truth)
                self.assertGreaterEqual(quality, 0)
                self.assertLessEqual(quality, 1)
                self.assertEqual(quality == 1, set(predicted) == set(truth))

    def test_soft_backdoor_matching_matches_brute_force_optimum(self) -> None:
        def dice(left: tuple[str, ...], right: tuple[str, ...]) -> Fraction:
            left_set = set(left)
            right_set = set(right)
            if not left_set and not right_set:
                return Fraction(1)
            return Fraction(
                2 * len(left_set & right_set),
                len(left_set) + len(right_set),
            )

        subsets = (
            (),
            ("A",),
            ("B",),
            ("C",),
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
        )
        families = tuple(
            family for size in range(1, 4) for family in itertools.combinations(subsets, size)
        )
        for predicted in families:
            for truth in families:
                if len(predicted) <= len(truth):
                    rows = predicted
                    columns = truth
                else:
                    rows = truth
                    columns = predicted
                optimum = max(
                    sum(
                        (
                            dice(rows[index], columns[column])
                            for index, column in enumerate(assignment)
                        ),
                        start=Fraction(0),
                    )
                    for assignment in itertools.permutations(range(len(columns)), len(rows))
                )
                expected = 2 * optimum / (len(predicted) + len(truth))
                self.assertEqual(
                    soft_adjustment_family_f1(predicted, truth),
                    expected,
                    (predicted, truth),
                )

    def test_invalid_owner_diagnostics_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported terminal diagnostic kind"):
            terminal_quality_reward({"kind": "unknown"})
        with self.assertRaisesRegex(ValueError, "l1_error"):
            terminal_quality_reward({"kind": "target_query", "l1_error": 5})
        with self.assertRaisesRegex(ValueError, "duplicate adjustment set"):
            soft_adjustment_family_f1((("A",), ("A",)), (("A",),))


if __name__ == "__main__":
    unittest.main()
