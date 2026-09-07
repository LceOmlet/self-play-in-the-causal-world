"""Exact decision/reward witness, legal-experiment optimum, and finite-tape certificate.

This is an offline audit. It does not replace a sampler, score, or trainer.
"""

import hashlib
import json
import math
import sys
import time
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cpt_world import OutcomeTape, WorldSpecEpisode  # noqa: E402
from cpt_world import world_space as w  # noqa: E402
from cpt_world.identification import INTERACTION_SURFACE_VERSION  # noqa: E402
from cpt_world.query_truth import worldspec_interventional_distribution  # noqa: E402
from cpt_world.rewards import TERMINAL_SAMPLING_RESOLUTION, terminal_quality_reward  # noqa: E402
from cpt_world.task_scoring import score_terminal_answer  # noqa: E402

BUDGET = 8 * 2**14
TAPE_KEY = "decision-budget-all-legal-streams-20260907"
ANCHORS = dict(decision_target=0, outcome=1, outcome_state=1, objective="maximize")
ACTIONS = [None] + [(node, state) for node in range(2, 8) for state in (0, 1)]
SOURCE_FILES = [
    "world.py",
    "world_runtime.py",
    "world_space.py",
    "query_truth.py",
    "task_scoring.py",
    "rewards.py",
    "identification.py",
    "episode.py",
]


def make_world(delta, sign, *, binary64=False):
    names = ("X", "Y", "U2", "U3", "U4", "U5", "U6", "U7")
    cpt = {node: ((F(1, 2), F(1, 2)),) for node in range(8)}
    cpt[1] = tuple((1 - p, p) for p in (F(1, 2) - sign * delta, F(1, 2) + sign * delta))
    if binary64:
        cpt = {node: tuple(tuple(map(float, row)) for row in rows) for node, rows in cpt.items()}
    return w.WorldSpec(
        family="sampled_dag",
        topology="single-direct-binary-decision",
        variables=names,
        domains=(2,) * 8,
        state_names=(("0", "1"),) * 8,
        edges=((0, 1),),
        parents={node: (0,) if node == 1 else () for node in range(8)},
        cpt=cpt,
    )


def make_seed(world):
    return w.assemble_seed(
        world,
        "mechanism_hidden",
        "best_intervention",
        "decision",
        seed_id="DECISION-BUDGET-SAME-PUBLIC-PROMPT",
        anchors=ANCHORS,
        manipulability={name: node not in (0, 1) for node, name in enumerate(world.variables)},
        readable={name: True for name in world.variables},
        observation_bandwidth=8,
        observation_budget_exponent=14,
        observation_budget=BUDGET,
        interaction_surface_version=INTERACTION_SURFACE_VERSION,
    )


def score_pair(delta, *, binary64=False):
    episodes, records = [], []
    for sign in (-1, 1):
        world = make_world(delta, sign, binary64=binary64)
        assert w.legal_world(world)
        assert {"decision_target": 0, "outcome": 1} in w.legal_query_anchors(
            world, "best_intervention"
        )
        assert w._minimum_backdoor_adjustment_size(8, world.edges, 0, 1) == 0
        assert w._best_intervention_observational_relation(world, ANCHORS) == (False, 0.0)
        assert w._best_intervention_proposal_is_admitted(world, ANCHORS, desired_discordant=False)
        seed = make_seed(world)
        episodes.append(WorldSpecEpisode(world, seed, OutcomeTape(TAPE_KEY)))
        for action in (0, 1):
            score = score_terminal_answer(
                json.dumps({"type": "answer", "value": f"state_{action}"}), seed, world
            )
            wrong = action != int(sign > 0)
            assert score["regret"] == (2 * delta if wrong else 0)
            assert score["probability_span"] == 2 * delta
            assert score["normalized_regret"] == int(wrong)
            assert score["observational_shortcut_normalized_regret"] == 0
            quality = terminal_quality_reward(score)
            expected = (
                TERMINAL_SAMPLING_RESOLUTION / (1 + TERMINAL_SAMPLING_RESOLUTION) if wrong else F(1)
            )
            assert quality == expected
            records.append(
                {
                    "sign": sign,
                    "action": action,
                    "absolute_regret": str(F(score["regret"])),
                    "normalized_regret": str(score["normalized_regret"]),
                    "quality_exact": str(quality),
                    "quality": float(quality),
                }
            )
    assert episodes[0].initial_messages() == episodes[1].initial_messages()
    return episodes, records


def verify_legal_laws(delta):
    worlds = [make_world(delta, sign) for sign in (-1, 1)]
    singles, ratios = 0, 0
    for action in ACTIONS:
        laws = [
            dict(
                worldspec_interventional_distribution(
                    world, {} if action is None else {action[0]: action[1]}
                )
            )
            for world in worlds
        ]
        for node in range(8):
            marginal = [
                tuple(
                    sum(p for values, p in law.items() if values[node] == state) for state in (0, 1)
                )
                for law in laws
            ]
            assert marginal[0] == marginal[1]
            singles += 1
        for values, probability in laws[0].items():
            if not probability:
                assert laws[1][values] == 0
                continue
            multiplier = (F(1, 2) + delta) / (F(1, 2) - delta)
            assert laws[1][values] / probability == (
                multiplier if values[0] == values[1] else 1 / multiplier
            )
            ratios += 1
        for sign, law in zip((-1, 1), laws, strict=True):
            assert (
                sum(p for values, p in law.items() if values[0] == values[1])
                == F(1, 2) + sign * delta
            )
    return {
        "legal_actions": len(ACTIONS),
        "identical_single_marginals": singles,
        "full_assignment_likelihood_ratios": ratios,
    }


def binomial_risk(n, delta):
    """Symmetric likelihood-ratio risk, with fair randomization at an even-n tie."""
    if n == 0:
        return mp.mpf(1) / 2
    p = mp.mpf(1) / 2 + mp.mpf(delta.numerator) / delta.denominator
    k = n // 2
    mass = mp.exp(
        mp.loggamma(n + 1)
        - mp.loggamma(k + 1)
        - mp.loggamma(n - k + 1)
        + k * mp.log(p)
        + (n - k) * mp.log1p(-p)
    )
    total = mass / 2 if n % 2 == 0 else mass
    for j in range(k, 0, -1):
        mass *= mp.mpf(j) / (n - j + 1) * (1 - p) / p
        total += mass
    return total


def verify_small_risks():
    count = 0
    for n in range(17):
        for delta in (F(1, 64), F(1, 8), F(1, 4)):
            p = F(1, 2) + delta
            exact = sum(
                (
                    F(math.comb(n, k)) * p**k * (1 - p) ** (n - k) * (F(1, 2) if 2 * k == n else 1)
                    for k in range(n + 1)
                    if 2 * k <= n
                ),
                F(0),
            )
            assert abs(
                binomial_risk(n, delta) - mp.mpf(exact.numerator) / exact.denominator
            ) < mp.mpf("1e-55")
            count += 1
    return count


def scan_tape():
    """Visit every potentially reachable Y draw for every legal arm, using the real tape owner."""
    tape = OutcomeTape(TAPE_KEY)
    records = []
    minimum = F(1)
    minimum_float = 1.0
    for action in ACTIONS:
        started = time.perf_counter()
        arm_min = F(1)
        arm_float_min = 1.0
        digest = hashlib.sha256()
        location = None
        for sample in range(BUDGET):
            u = (
                tape.worldspec_observation_node_uniform(sample, 1)
                if action is None
                else tape.worldspec_node_uniform(*action, sample, 1)
            )
            distance = abs(u - F(1, 2))
            float_distance = abs(float(u) - 0.5)
            if distance < arm_min:
                arm_min, location = distance, sample
            arm_float_min = min(arm_float_min, float_distance)
            numerator = u.numerator * ((2**256) // u.denominator)
            digest.update(numerator.to_bytes(32, "big"))
        minimum, minimum_float = min(minimum, arm_min), min(minimum_float, arm_float_min)
        row = {
            "action": action,
            "draws": BUDGET,
            "minimum_exact_distance": str(arm_min),
            "minimum_float_distance": arm_float_min,
            "minimum_index": location,
            "stream_sha256": digest.hexdigest(),
            "seconds": time.perf_counter() - started,
        }
        records.append(row)
        print(json.dumps({"tape_scan": row}), flush=True)
    margin = min(minimum, F(minimum_float))
    delta = F(1, 2)
    while 4 * delta >= margin:
        delta /= 2
    assert float(F(1, 2) + delta) != 0.5 and float(F(1, 2) - delta) != 0.5
    return delta, {
        "tape_key": TAPE_KEY,
        "records": records,
        "total_draws": len(ACTIONS) * BUDGET,
        "minimum_exact_distance": str(minimum),
        "minimum_float_distance": minimum_float,
        "chosen_delta_exact": str(delta),
        "margin_over_delta": float(margin / delta),
    }


def execute_pair(episodes, batch_size):
    records = []
    for episode in episodes:
        labels = episode.seed["visible_schema"]["variable_labels"]
        command = {
            "type": "observe",
            "measure": [labels["X"], labels["Y"]],
            "batch_size": batch_size,
        }
        started = time.perf_counter()
        step = episode.step(json.dumps(command))
        assert step.kind == "batch" and episode.observations_used == 2 * batch_size
        count_equal = sum(
            count
            for states, count in zip(step.batch.assignments, step.batch.counts, strict=True)
            if states[0] == states[1]
        )
        choice = int(2 * count_equal > batch_size)  # A fixed tie is allowed for this one tape.
        terminal = episode.step(json.dumps({"type": "answer", "value": f"state_{choice}"}))
        assert terminal.kind == "terminal"
        records.append(
            {
                "count_equal": count_equal,
                "action": choice,
                "quality": float(terminal.reward),
                "cost": episode.observations_used,
                "queries": episode.queries_used,
                "feedback_sha256": hashlib.sha256(step.message.encode()).hexdigest(),
                "seconds": time.perf_counter() - started,
            }
        )
    return records


def main():
    mp.mp.dps = 70
    fingerprints = {
        name: hashlib.sha256((PROJECT / "src/cpt_world" / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
    report = {
        "passed": False,
        "source_sha256": fingerprints,
        "budget": BUDGET,
        "small_rational_risk_checks": verify_small_risks(),
        "iid_model_cases": [],
    }
    bad = TERMINAL_SAMPLING_RESOLUTION / (1 + TERMINAL_SAMPLING_RESOLUTION)
    bad_mp = mp.mpf(bad.numerator) / bad.denominator
    for delta in (F(1, 128), F(1, 16384), F(1, 1048576)):
        episodes, scores = score_pair(delta)
        _, float_scores = score_pair(delta, binary64=True)
        assert scores == float_scores
        strength = 0.5 * math.log(float((F(1, 2) + delta) / (F(1, 2) - delta)))
        rows = w._exponential_tilt_rows(
            (0.5, 0.5),
            ((1.0, -1.0), (-1.0, 1.0)),
            strength,
            score_scale=w._single_parent_pairwise_score_scale(2),
        )
        assert (
            max(
                abs(float(p) - q)
                for original, generated in zip(make_world(delta, 1).cpt[1], rows, strict=True)
                for p, q in zip(original, generated, strict=True)
            )
            < 2e-16
        )
        risk = binomial_risk(BUDGET // 2, delta)
        item = {
            "delta": str(delta),
            "causal_span": str(2 * delta),
            "et_v2_strength": strength,
            "scores": scores,
            "legal_laws": verify_legal_laws(delta),
            "optimal_error_iid": str(risk),
            "optimal_mean_quality_iid": str(1 - (1 - bad_mp) * risk),
        }
        report["iid_model_cases"].append(item)
        print(json.dumps({"iid_case": item}), flush=True)
    delta, scan = scan_tape()
    report["finite_tape_certificate"] = scan
    episodes, records = score_pair(delta)
    report["finite_tape_scores"] = records
    replay = execute_pair(episodes, BUDGET // 2)
    assert replay[0]["feedback_sha256"] == replay[1]["feedback_sha256"]
    assert replay[0]["action"] == replay[1]["action"]
    assert sorted(r["quality"] for r in replay) == [float(bad), 1.0]
    report["finite_tape_full_budget_replay"] = replay
    report["finite_tape_optimal_equal_prior_quality_exact"] = str((1 + bad) / 2)
    report["finite_tape_optimal_equal_prior_quality"] = float((1 + bad) / 2)
    report["scope"] = (
        "IID risk is for the declared statistical model; the separately exhaustive "
        "fixed-tape witness is conditional on this recorded tape, not an estimate "
        "of generated-task prevalence. No reward or generator changes."
    )
    assert fingerprints == {
        name: hashlib.sha256((PROJECT / "src/cpt_world" / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
    report["passed"] = True
    (ROOT / "decision-budget-acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"passed": True, "finite_tape_delta": str(delta), "runtime_replay": replay}),
        flush=True,
    )


if __name__ == "__main__":
    main()
