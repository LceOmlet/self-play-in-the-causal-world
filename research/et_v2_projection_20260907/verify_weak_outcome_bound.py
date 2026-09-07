"""Check the weak-outcome bound through the actual CPT, query and reward owners."""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cpt_world import world_space as w  # noqa: E402
from cpt_world.identification import INTERACTION_SURFACE_VERSION  # noqa: E402
from cpt_world.rewards import TERMINAL_SAMPLING_RESOLUTION, terminal_quality_reward  # noqa: E402
from cpt_world.task_scoring import score_terminal_answer  # noqa: E402

records = []
for seed in range(8):
    rng = random.Random(9901 + seed)
    direction = w._sample_parent_subset_balanced_effect((2, 3), 3, rng)
    scale = w._contextual_parent_pair_score_scale(direction, (2, 3))
    base = w._simplex_uniform(3, rng)
    unit_scores = [[scale * math.sqrt(18) * value for value in row] for row in direction]
    diameter = max(
        max(a - b for a, b in zip(left, right, strict=True))
        - min(a - b for a, b in zip(left, right, strict=True))
        for left in unit_scores
        for right in unit_scores
    )
    resolution = float(TERMINAL_SAMPLING_RESOLUTION)
    desired_quality = 0.95
    sufficient_amplitude = (
        4 * math.atanh(resolution * (1 - desired_quality) / desired_quality) / diameter
    )
    for amplitude in (1e-6, 1e-4, 0.01, 0.1, 1.0, sufficient_amplitude * 0.99):
        domains = (2, 2, 3, 3, 2, 2, 2, 2)
        parents = {0: (), 1: (0,), 2: (1,), 3: (0, 2), 4: (), 5: (), 6: (), 7: ()}
        cpt = {node: ((0.5, 0.5),) for node in range(8)}
        cpt[1] = ((0.95, 0.05), (0.05, 0.95))
        cpt[2] = ((0.7, 0.2, 0.1), (0.1, 0.2, 0.7))
        cpt[3] = w._exponential_tilt_rows(base, direction, amplitude, score_scale=scale)
        world = w.WorldSpec(
            family="sampled_dag",
            topology="weak-outcome-proof",
            variables=("Z", "X", "M", "Y", "A", "B", "C", "D"),
            domains=domains,
            state_names=tuple(tuple(str(i) for i in range(domain)) for domain in domains),
            edges=((0, 1), (0, 3), (1, 2), (2, 3)),
            parents=parents,
            cpt=cpt,
        )
        seed_task = w.assemble_seed(
            world,
            "mechanism_hidden",
            "ate",
            "target_query",
            seed_id=f"WEAK-OUTCOME-{seed}-{amplitude}",
            anchors={"treatment": 1, "outcome": 3, "baseline_value": 0, "treatment_value": 1},
            observation_bandwidth=8,
            observation_budget_exponent=14,
            observation_budget=131072,
            interaction_surface_version=INTERACTION_SURFACE_VERSION,
        )
        score = score_terminal_answer(
            json.dumps({"type": "answer", "effect": {"state_0": 0, "state_1": 0, "state_2": 0}}),
            seed_task,
            world,
        )
        error = float(score["total_variation_error"])
        quality = float(terminal_quality_reward(score))
        bound = math.tanh(amplitude * diameter / 4)
        quality_lower = resolution / (resolution + bound)
        assert error <= bound + 1e-14
        assert quality >= quality_lower - 1e-14
        if amplitude < sufficient_amplitude:
            assert quality >= desired_quality
        records.append(
            {
                "seed": seed,
                "amplitude": amplitude,
                "score_diameter": diameter,
                "effect_tv": error,
                "effect_tv_upper": bound,
                "zero_quality": quality,
                "zero_quality_lower": quality_lower,
                "sufficient_amplitude_for_q95": sufficient_amplitude,
                "conditional_uniform_amplitude_mass_lower_q95": min(
                    1.0, sufficient_amplitude / math.sqrt(3)
                ),
            }
        )
report = {
    "passed": True,
    "actual_protocol_and_reward_cases": len(records),
    "records": records,
    "scope": (
        "Sufficient weak-outcome condition; not the prevalence of easy ATE tasks. "
        "Strong-reversal filtering applies only to best_intervention."
    ),
}
(ROOT / "weak-outcome-bound-acceptance.json").write_text(json.dumps(report, indent=2))
print(
    json.dumps(
        {
            "passed": True,
            "cases": len(records),
            "min_quality_at_amplitude_1e_minus6": min(
                row["zero_quality"] for row in records if row["amplitude"] == 1e-6
            ),
        },
        indent=2,
    )
)
