from __future__ import annotations

import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from itertools import islice
from pathlib import Path
from unittest.mock import patch

from cpt_world import trl_environment as owner
from cpt_world.registry import TASK_FAMILY_QUERY_TYPES
from cpt_world.world_space import WorldGrammar

TRUTH = {"type": "individual_counterfactual_probability", "lower": 0.25, "upper": 0.75}


def scripted_row(grammar, sample_index, query_type, *, best_intervention_balance_slot=None):
    del grammar
    # The best-intervention proposal identity intentionally differs from its
    # outer source cursor, as it does in the real conditional generator.
    proposal = sample_index + (1_000_000 if query_type == "best_intervention" else 0)
    seed = {"seed_id": f"SAMPLED-{proposal}-{query_type}", "outer_index": sample_index}
    row = {
        "sample_index": proposal,
        "query_type": query_type,
        "balance_slot": best_intervention_balance_slot,
        "terminal_truth_json": "",
    }
    return row, object(), seed


def scripted_truth(world, seed, **kwargs):
    del world, kwargs
    if seed["outer_index"] == 101:
        raise owner.CounterfactualResourceLimitError("scripted hard timeout")
    if seed["outer_index"] == 102:
        raise RuntimeError("scripted uncertified endpoint")
    return dict(TRUTH)


class BalancedTrainingRowStreamTests(unittest.TestCase):
    @patch.object(owner, "_training_row", side_effect=scripted_row)
    @patch.object(owner, "compute_counterfactual_truth_isolated", side_effect=scripted_truth)
    def test_rejections_advance_only_candidate_cursor(self, solve, build):
        stream = owner.BalancedTrainingRowStream(
            start_seed=100, counterfactual_endpoint_time_limit_seconds=2.5
        )
        self.assertIs(iter(stream), stream)
        self.assertEqual(build.call_count, 0)
        self.assertEqual(next(stream)["sample_index"], 100)
        self.assertEqual(solve.call_count, 0)
        with patch.dict(owner.os.environ, {"CPT_WORLD_RESOURCE_DIAGNOSTIC_DIR": "trace-dir"}):
            with redirect_stdout(io.StringIO()) as output:
                row = next(stream)
        self.assertEqual(row["sample_index"], 103)
        self.assertEqual(json.loads(row["terminal_truth_json"]), TRUTH)
        self.assertEqual(
            stream.state_dict(),
            {
                "version": 1,
                "next_sample_index": 104,
                "next_family_offset": 2,
                "best_intervention_balance_slot": 0,
                "accepted_rows": 2,
            },
        )
        self.assertEqual([r["sample_index"] for r in stream.last_attempts], [101, 102, 103])
        self.assertEqual(
            [r["status"] for r in stream.last_attempts], ["rejected"] * 2 + ["accepted"]
        )
        self.assertEqual(
            [r.get("error_type") for r in stream.last_attempts],
            ["CounterfactualResourceLimitError", "RuntimeError", None],
        )
        self.assertEqual(output.getvalue().count("CPT_WORLD_COUNTERFACTUAL_SKIP="), 1)
        for call in solve.call_args_list:
            self.assertEqual(call.kwargs["endpoint_time_limit_seconds"], 2.5)
            self.assertEqual(call.kwargs["diagnostic_dir"], Path("trace-dir"))
        previous_attempts = stream.last_attempts
        self.assertEqual(next(stream)["query_type"], "backadj_minimal_sets")
        self.assertEqual(len(stream.last_attempts), 1)
        self.assertEqual(len(previous_attempts), 3)

    @patch.object(owner, "_training_row", side_effect=scripted_row)
    @patch.object(owner, "compute_counterfactual_truth_isolated", side_effect=scripted_truth)
    def test_every_family_boundary_resumes_same_rows_and_best_balance(self, _solve, _build):
        with redirect_stdout(io.StringIO()):
            reference = list(islice(owner.BalancedTrainingRowStream(start_seed=100), 12))
            for boundary in range(1, 11):
                with self.subTest(boundary=boundary):
                    initial = owner.BalancedTrainingRowStream(start_seed=100)
                    prefix = list(islice(initial, boundary))
                    saved = json.loads(json.dumps(initial.state_dict()))
                    resumed = owner.BalancedTrainingRowStream(start_seed=999)
                    resumed.load_state_dict(saved)
                    tail = list(islice(resumed, 12 - boundary))
                    self.assertEqual(prefix + tail, reference)
                    self.assertEqual(resumed.accepted_rows, 12)
                    self.assertEqual(resumed.next_sample_index, 114)
                    self.assertEqual(resumed.next_family_offset, 2)
                    self.assertEqual(resumed.best_intervention_balance_slot, 2)
        self.assertEqual(
            [r["query_type"] for r in reference],
            list(TASK_FAMILY_QUERY_TYPES) * 2 + list(TASK_FAMILY_QUERY_TYPES[:2]),
        )
        self.assertEqual(
            [r["balance_slot"] for r in reference if r["query_type"] == "best_intervention"],
            [0, 1],
        )

    @patch.object(owner, "_training_row", side_effect=scripted_row)
    @patch.object(owner, "compute_counterfactual_truth_isolated", side_effect=scripted_truth)
    def test_restore_after_accepted_cf_does_not_reopen_its_solver(self, solve, _build):
        initial = owner.BalancedTrainingRowStream(start_seed=100)
        with redirect_stdout(io.StringIO()):
            next(initial)
            accepted_row = next(initial)
        state = initial.state_dict()
        self.assertEqual(solve.call_count, 3)
        resumed = owner.BalancedTrainingRowStream()
        resumed.load_state_dict(state)
        self.assertEqual(solve.call_count, 3)
        self.assertEqual(next(resumed)["query_type"], "backadj_minimal_sets")
        self.assertEqual(solve.call_count, 3)
        self.assertEqual(json.loads(accepted_row["terminal_truth_json"]), TRUTH)

    def test_cursor_validation_is_atomic_and_snapshot_is_detached(self):
        stream = owner.BalancedTrainingRowStream(start_seed=100)
        original = stream.state_dict()
        invalid = [
            {**original, "version": 2},
            {**original, "accepted_rows": True},
            {**original, "next_sample_index": -1},
            {**original, "next_family_offset": 1},
            {**original, "best_intervention_balance_slot": 1},
            {
                **original,
                "next_sample_index": 1,
                "accepted_rows": 5,
                "best_intervention_balance_slot": 1,
            },
            {key: value for key, value in original.items() if key != "accepted_rows"},
        ]
        for state in invalid:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    stream.load_state_dict(state)
                self.assertEqual(stream.state_dict(), original)
        detached = stream.state_dict()
        detached["next_sample_index"] = 1000
        self.assertEqual(stream.next_sample_index, 100)

    @patch.object(owner, "_MAX_COUNTERFACTUAL_RESAMPLES", 2)
    @patch.object(owner, "_training_row", side_effect=scripted_row)
    @patch.object(
        owner, "compute_counterfactual_truth_isolated", side_effect=RuntimeError("no certificate")
    )
    def test_original_retry_limit_raises_without_delivering_uncertified_row(self, solve, _build):
        stream = owner.BalancedTrainingRowStream(start_seed=100)
        next(stream)
        with self.assertRaisesRegex(RuntimeError, "could not sample a counterfactual task"):
            next(stream)
        self.assertEqual(solve.call_count, 2)
        self.assertEqual(stream.accepted_rows, 1)
        self.assertEqual(stream.next_family_offset, 1)
        self.assertEqual(stream.next_sample_index, 103)
        self.assertEqual([r["status"] for r in stream.last_attempts], ["rejected", "rejected"])

    @patch.object(owner, "compute_counterfactual_truth_isolated", return_value=TRUTH)
    def test_real_owner_rows_match_prefix_captured_before_refactor(self, _solve):
        # Captured from the unmodified production _iter_random_balanced_training_rows
        # on 2026-09-08, with the same fixed CF result. These are complete row
        # hashes, including prompt, proposal/anchor, tape and cached terminal truth.
        expected = [
            "38cf48432590ab58a12703d424846e91583d58bd63b47346bfd7fa8d05c51c95",
            "22804310f4576945a27f9fcac90dd2ba5102cd660b41b3f3523e6a67d2dad418",
            "e8df406a74c8f3a832cc52fd3e4ee79941a7e1719faae27fbdb5ff837c922b07",
            "30fcbb5032a706ab6eabde3e73b7f6e66634da15127651d0a9f4a1eb20a08e92",
            "14f7f71c30f1d792058c0d450c29089f117037f0782e3381d02301b2cbacee30",
            "846e6e053913721f1bff514deedebc6ca8228b6e14b4199b90c668a2f5aa6d32",
            "c6f5d52651981060ba9d724834633dc0f79219ba7b8578ce0dc80415cc9a4f89",
            "53eaa1b0801ac7ba5ed5a30ba3d6cf1b23148f87f4e877c605526caaae646306",
            "a2be13000ca16aad2dc6d9e36dc4287b16ce648a63948c6716a16f1af8737123",
            "42cf1dae7ef11b8edfbd5e31a847f9e213c58a82f6405a0246052099e043ab5b",
        ]
        stream = owner._iter_random_balanced_training_rows(
            start_seed=17, grammar=WorldGrammar(node_counts=(8,))
        )
        rows = list(islice(stream, len(expected)))
        observed = [
            hashlib.sha256(
                json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            for row in rows
        ]
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
