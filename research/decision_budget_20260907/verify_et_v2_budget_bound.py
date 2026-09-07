"""Verify the analytic binary sole-parent ET-V2 legal-experiment bound."""

import hashlib
import json
import math
import random
import sys
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cpt_world import world_space as w  # noqa: E402
from cpt_world.rewards import TERMINAL_SAMPLING_RESOLUTION  # noqa: E402

mp.mp.dps = 70
BUDGET = 8 * 2**14
bad = TERMINAL_SAMPLING_RESOLUTION / (1 + TERMINAL_SAMPLING_RESOLUTION)
bad_mp = mp.mpf(bad.numerator) / bad.denominator
row_checks = 0
maximum_delta = 0.0
kl_checks = 0
for base_numerator in range(1, 16):
    b = mp.mpf(base_numerator) / 16
    eta = mp.log(b / (1 - b))
    for strength in (F(1, 10000), F(1, 100), F(1, 5), F(1)):
        t = mp.mpf(strength.numerator) / strength.denominator
        low = 1 / (1 + mp.exp(-eta + 2 * t))
        high = 1 / (1 + mp.exp(-eta - 2 * t))
        for p, q in ((low, high), (high, low)):
            kl = p * mp.log(p / q) + (1 - p) * mp.log((1 - p) / (1 - q))
            assert 0 <= kl <= 2 * t**2
            kl_checks += 1
        for seed in range(8):
            direction = w._sample_parent_subset_balanced_effect((2,), 2, random.Random(seed))
            rms = math.sqrt(sum(v * v for row in direction for v in row) / 4)
            sign = 1 if direction[0][0] > 0 else -1
            expected_direction = ((sign, -sign), (-sign, sign))
            assert (
                max(
                    abs(value / rms - expected)
                    for row, reference in zip(direction, expected_direction, strict=True)
                    for value, expected in zip(row, reference, strict=True)
                )
                < 1e-14
            )
            rows = w._exponential_tilt_rows(
                (float(1 - b), float(b)),
                direction,
                float(t),
                score_scale=w._single_parent_pairwise_score_scale(2),
            )
            expected = (low, high) if sign > 0 else (high, low)
            error = max(abs(rows[x][1] - float(expected[x])) for x in (0, 1))
            assert error < 4e-16
            maximum_delta = max(maximum_delta, error)
            row_checks += 2

targets = []
for quality_target in (F(3, 4), F(9, 10), F(99, 100)):
    target = mp.mpf(quality_target.numerator) / quality_target.denominator
    a = 1 - 2 * (1 - target) / (1 - bad_mp)
    cutoff = max(mp.mpf(0), a) / mp.sqrt(BUDGET)
    targets.append(
        {
            "target_quality": str(quality_target),
            "strength_cutoff": str(cutoff),
            "conditional_uniform_strength_mass": str(cutoff / mp.sqrt(3)),
            "meaning": (
                "Conditional on a binary decision and binary sole-parent outcome in a "
                "concordant slot, sign-pairs below this amplitude have oracle-aided "
                "equal-prior optimal mean quality strictly below the target under the IID "
                "model. This is not the whole-task-distribution prevalence."
            ),
        }
    )
report = {
    "passed": True,
    "budget": BUDGET,
    "real_et_v2_row_checks": row_checks,
    "maximum_cpt_error": maximum_delta,
    "high_precision_kl_checks": kl_checks,
    "kl_per_full_sample_upper": "2*t**2",
    "all_policy_pair_error_lower": "max(0,(1-sqrt(B)*t)/2)",
    "conditional_strength_results": targets,
    "assumptions": [
        "Declared continuous ET-V2 law and IID sample model",
        "binary X,Y; Pa(Y)={X}",
        "The rest of the legal DAG/CPT is arbitrary and fixed between signs",
        "Uniform amplitude and equally likely one-dimensional Gaussian direction signs",
        "All nuisance parameters may be revealed to the decision rule; the sign remains hidden",
    ],
    "world_space_sha256": hashlib.sha256(
        (PROJECT / "src/cpt_world/world_space.py").read_bytes()
    ).hexdigest(),
}
(ROOT / "et-v2-budget-bound.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
