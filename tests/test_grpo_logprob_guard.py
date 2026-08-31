from __future__ import annotations

import unittest

from scripts.grpo_logprob_guard import require_finite_sampling_logprobs


class GRPOLogprobGuardTests(unittest.TestCase):
    def test_accepts_finite_sampling_logprobs(self) -> None:
        require_finite_sampling_logprobs(
            [[-0.25, -1.5], [], [-3.0]],
            global_step=7,
        )

    def test_rejects_nan_with_exact_location(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"global_step=22, sequence_index=1, token_index=1, value=nan",
        ):
            require_finite_sampling_logprobs(
                [[-0.25], [-1.5, float("nan")]],
                global_step=22,
            )

    def test_rejects_none_created_by_trl_nan_sanitization(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"sequence_index=0, token_index=1, value=None",
        ):
            require_finite_sampling_logprobs(
                [[-0.25, None]],
                global_step=22,
            )

    def test_rejects_missing_sampling_logprobs(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"returned no sampling log-probs"):
            require_finite_sampling_logprobs(None, global_step=22)


if __name__ == "__main__":
    unittest.main()
