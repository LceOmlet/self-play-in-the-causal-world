from __future__ import annotations

import unittest
from fractions import Fraction

from cpt_world import (
    DEFAULT_REWARD_MAX_GRAPH_NODES,
    TERMINAL_QUALITY_REWARD_VERSION,
    TERMINAL_SAMPLING_RESOLUTION,
    UNFINISHED_TERMINAL_QUALITY,
    terminal_quality_reward,
)


class TerminalQualityRewardTests(unittest.TestCase):
    def test_reward_contract_version_and_unfinished_value_are_frozen(self) -> None:
        self.assertEqual(TERMINAL_QUALITY_REWARD_VERSION, "terminal-quality-v10")
        self.assertEqual(DEFAULT_REWARD_MAX_GRAPH_NODES, 16)
        self.assertEqual(UNFINISHED_TERMINAL_QUALITY, 0)

    def test_numeric_shortcuts_use_the_fixed_budget_sampling_resolution(self) -> None:
        resolution = TERMINAL_SAMPLING_RESOLUTION
        self.assertEqual(
            terminal_quality_reward(
                {
                    "kind": "target_query",
                    "total_variation_error": Fraction(1, 4),
                    "observational_shortcut_error": Fraction(1, 4),
                }
            ),
            (resolution + Fraction(1, 4)) / (resolution + Fraction(1, 2)),
        )
        self.assertEqual(
            terminal_quality_reward(
                {
                    "kind": "counterfactual_roi",
                    "mean_absolute_endpoint_error": Fraction(1, 5),
                    "observational_shortcut_error": Fraction(1, 5),
                }
            ),
            (resolution + Fraction(1, 5)) / (resolution + Fraction(2, 5)),
        )
        self.assertEqual(
            terminal_quality_reward(
                {
                    "kind": "decision",
                    "normalized_regret": Fraction(2, 5),
                    "observational_shortcut_normalized_regret": Fraction(2, 5),
                }
            ),
            (resolution + Fraction(2, 5)) / (resolution + Fraction(4, 5)),
        )

    def test_shortcut_calibration_is_exact_at_one_and_continuous_below_it(self) -> None:
        self.assertEqual(
            terminal_quality_reward(
                {
                    "kind": "decision",
                    "normalized_regret": 0,
                    "observational_shortcut_normalized_regret": Fraction(1, 10),
                }
            ),
            1,
        )
        self.assertEqual(
            terminal_quality_reward(
                {
                    "kind": "decision",
                    "normalized_regret": Fraction(3, 10),
                    "observational_shortcut_normalized_regret": Fraction(1, 10),
                }
            ),
            (TERMINAL_SAMPLING_RESOLUTION + Fraction(1, 10))
            / (TERMINAL_SAMPLING_RESOLUTION + Fraction(2, 5)),
        )

    def test_zero_and_near_zero_shortcut_errors_are_continuous(self) -> None:
        error = Fraction(1, 1000)
        zero = terminal_quality_reward(
            {
                "kind": "target_query",
                "total_variation_error": error,
                "observational_shortcut_error": 0,
            }
        )
        near_zero = terminal_quality_reward(
            {
                "kind": "target_query",
                "total_variation_error": error,
                "observational_shortcut_error": Fraction(1, 10**13),
            }
        )
        self.assertEqual(
            zero,
            TERMINAL_SAMPLING_RESOLUTION / (TERMINAL_SAMPLING_RESOLUTION + error),
        )
        self.assertLess(abs(float(near_zero - zero)), 1e-10)

    def test_unavailable_shortcut_keeps_continuous_absolute_quality(self) -> None:
        self.assertEqual(
            terminal_quality_reward(
                {
                    "kind": "counterfactual_roi",
                    "mean_absolute_endpoint_error": Fraction(1, 5),
                    "observational_shortcut_error": None,
                }
            ),
            Fraction(4, 5),
        )

    def test_mediator_reward_preserves_both_partial_f1_signals(self) -> None:
        score = {
            "kind": "mediator",
            "mediator_f1": Fraction(2, 3),
            "order_f1": Fraction(1, 2),
        }
        self.assertEqual(terminal_quality_reward(score), Fraction(7, 12))

    def test_backdoor_reward_is_linear_in_nearest_valid_set_distance(self) -> None:
        self.assertEqual(
            terminal_quality_reward({"kind": "backadj", "edit_distance": 0}),
            1,
        )
        self.assertEqual(
            terminal_quality_reward({"kind": "backadj", "edit_distance": 1}),
            Fraction(13, 14),
        )
        self.assertEqual(
            terminal_quality_reward({"kind": "backadj", "edit_distance": 2}),
            Fraction(6, 7),
        )
        self.assertEqual(
            terminal_quality_reward({"kind": "backadj", "edit_distance": 14}),
            0,
        )

    def test_backdoor_reward_uses_the_configured_maximum_graph_size(self) -> None:
        rewards = [
            terminal_quality_reward(
                {"kind": "backadj", "edit_distance": distance},
                max_graph_nodes=6,
            )
            for distance in range(5)
        ]
        self.assertEqual(rewards, [1, Fraction(3, 4), Fraction(1, 2), Fraction(1, 4), 0])
        self.assertEqual(
            [rewards[index] - rewards[index + 1] for index in range(4)],
            [Fraction(1, 4)] * 4,
        )

    def test_invalid_owner_diagnostics_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported terminal diagnostic kind"):
            terminal_quality_reward({"kind": "unknown"})
        with self.assertRaisesRegex(ValueError, "total_variation_error"):
            terminal_quality_reward(
                {
                    "kind": "target_query",
                    "total_variation_error": 5,
                    "observational_shortcut_error": 1,
                }
            )
        with self.assertRaisesRegex(ValueError, "edit_distance"):
            terminal_quality_reward({"kind": "backadj", "edit_distance": Fraction(1, 2)})
        with self.assertRaisesRegex(ValueError, "edit_distance"):
            terminal_quality_reward({"kind": "backadj", "edit_distance": -1})
        with self.assertRaisesRegex(ValueError, "configured graph support"):
            terminal_quality_reward(
                {"kind": "backadj", "edit_distance": 5},
                max_graph_nodes=6,
            )
        with self.assertRaisesRegex(ValueError, "max_graph_nodes"):
            terminal_quality_reward(
                {"kind": "backadj", "edit_distance": 0},
                max_graph_nodes=2,
            )


if __name__ == "__main__":
    unittest.main()
