from __future__ import annotations

import math
import unittest

from cpt_world import (
    METRICS_SCHEMA_VERSION,
    Direction,
    EffectVector,
    EpisodeTruth,
    TerminalDiagnostics,
    compute_terminal_diagnostics,
)


class TerminalDiagnosticsTests(unittest.TestCase):
    def test_perfect_predictions_are_zero(self) -> None:
        truths = {
            "forward": EpisodeTruth(EffectVector(0.8, 0.0), Direction.FORWARD),
            "reverse": EpisodeTruth(EffectVector(0.0, 0.4), Direction.REVERSE),
        }
        result = compute_terminal_diagnostics(
            truths,
            {episode_id: truth.effects for episode_id, truth in truths.items()},
        )
        self.assertEqual(result.vector_rmse, 0.0)
        self.assertEqual(result.active_mae, 0.0)
        self.assertEqual(result.inactive_mae, 0.0)
        self.assertEqual(result.n_episodes, 2)
        self.assertEqual(result.schema_version, METRICS_SCHEMA_VERSION)

    def test_all_three_formulas_use_continuous_component_errors(self) -> None:
        truths = {
            "a": EpisodeTruth(EffectVector(0.8, 0.0), Direction.FORWARD),
            "b": EpisodeTruth(EffectVector(0.0, 0.4), Direction.REVERSE),
        }
        predictions = {
            "a": EffectVector(0.7, 0.3),
            "b": EffectVector(0.1, 0.3),
        }
        result = compute_terminal_diagnostics(truths, predictions)
        self.assertAlmostEqual(result.vector_rmse, math.sqrt(0.03))
        self.assertAlmostEqual(result.active_mae, 0.1)
        self.assertAlmostEqual(result.inactive_mae, 0.2)

    def test_active_role_is_certificate_driven_without_threshold(self) -> None:
        tiny_truth = EpisodeTruth(EffectVector(0.0001, 0.0), Direction.FORWARD)
        result = compute_terminal_diagnostics(
            {"tiny": tiny_truth},
            {"tiny": EffectVector(0.0, 0.25)},
        )
        self.assertAlmostEqual(result.active_mae, 0.0001)
        self.assertAlmostEqual(result.inactive_mae, 0.25)

    def test_input_order_does_not_change_aggregate(self) -> None:
        first = EpisodeTruth(EffectVector(0.6, 0.0), Direction.FORWARD)
        second = EpisodeTruth(EffectVector(0.0, -0.4), Direction.REVERSE)
        result_a = compute_terminal_diagnostics(
            {"1": first, "2": second},
            {"1": EffectVector(0.5, -0.1), "2": EffectVector(0.2, -0.3)},
        )
        result_b = compute_terminal_diagnostics(
            {"2": second, "1": first},
            {"2": EffectVector(0.2, -0.3), "1": EffectVector(0.5, -0.1)},
        )
        self.assertEqual(result_a, result_b)

    def test_missing_or_extra_episode_is_rejected(self) -> None:
        truth = EpisodeTruth(EffectVector(0.8, 0.0), Direction.FORWARD)
        with self.assertRaisesRegex(ValueError, "episode IDs differ"):
            compute_terminal_diagnostics({"expected": truth}, {})
        with self.assertRaisesRegex(ValueError, "episode IDs differ"):
            compute_terminal_diagnostics(
                {"expected": truth},
                {"expected": truth.effects, "extra": truth.effects},
            )

    def test_episode_ids_must_be_nonempty_strings(self) -> None:
        truth = EpisodeTruth(EffectVector(0.8, 0.0), Direction.FORWARD)
        with self.assertRaisesRegex(TypeError, "truth episode IDs"):
            compute_terminal_diagnostics({1: truth}, {1: truth.effects})  # type: ignore[dict-item]
        with self.assertRaisesRegex(TypeError, "prediction episode IDs"):
            compute_terminal_diagnostics({"expected": truth}, {1: truth.effects})  # type: ignore[dict-item]

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one episode"):
            compute_terminal_diagnostics({}, {})

    def test_effect_vector_rejects_non_numeric_nonfinite_and_out_of_range(self) -> None:
        invalid_values: tuple[object, ...] = (True, "0.2", math.nan, math.inf, 1.01)
        for invalid in invalid_values:
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                EffectVector(invalid, 0.0)  # type: ignore[arg-type]

    def test_truth_certificate_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "active effect"):
            EpisodeTruth(EffectVector(0.0, 0.0), Direction.FORWARD)
        with self.assertRaisesRegex(ValueError, "inactive effect"):
            EpisodeTruth(EffectVector(0.8, 0.1), Direction.FORWARD)
        with self.assertRaises(TypeError):
            EpisodeTruth(EffectVector(0.8, 0.0), "forward")  # type: ignore[arg-type]

    def test_diagnostics_result_rejects_invalid_manual_construction(self) -> None:
        for invalid in (-0.1, math.nan, math.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TerminalDiagnostics(invalid, 0.0, 0.0, 1)
        with self.assertRaises(ValueError):
            TerminalDiagnostics(0.0, 0.0, 0.0, 0)


if __name__ == "__main__":
    unittest.main()
