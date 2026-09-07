"""Public-prompt/feedback-only causal decision diagnostic; never used by training."""

import hashlib
import json
import math
import re
from collections import defaultdict

import numpy as np


def parse_prompt(prompt):
    variables = prompt.split("Variables and finite states:\n", 1)[1].split("\n\n", 1)[0]
    domains = {
        line.split(": ", 1)[0]: len(line.split(": ", 1)[1].split(" / "))
        for line in variables.splitlines()
    }
    match = re.search(
        r"deployment state of (\w+) that (maximizes|minimizes) "
        r"P\((\w+)=state_(\d+) \| do\(\1=d\)\)",
        prompt,
    )
    if match is None:
        raise ValueError("Unsupported public decision query")
    x, objective, y, target = match.groups()
    targets = json.loads(re.search(r"Legal experimental do targets: (\[[^\n]+\])", prompt)[1])
    budget = int(re.search(r"total observation budget is (\d+) scalar", prompt)[1])
    assert x not in targets and y not in targets
    return {
        "domains": domains,
        "x": x,
        "y": y,
        "target": int(target),
        "maximize": objective == "maximizes",
        "targets": targets,
        "budget": budget,
    }


def adjusted_values(counts, x_states, y_states, target, alpha=0.5):
    """Empirical backdoor standardization, with explicit missing-cell mass."""
    strata = {}
    for values, count in counts.items():
        table = strata.setdefault(values[2:], np.zeros((x_states, y_states)))
        table[values[0], values[1]] += count
    total = sum(table.sum() for table in strata.values())
    if total <= 0:
        raise ValueError("No observations")
    estimate = np.zeros(x_states)
    missing = np.zeros(x_states)
    sparse = np.zeros(x_states)
    passive_counts = np.zeros((x_states, y_states))
    for table in strata.values():
        weight = table.sum() / total
        n = table.sum(1)
        estimate += weight * (table[:, target] + alpha) / (n + y_states * alpha)
        missing += weight * (n == 0)
        sparse += weight * (n < 5)
        passive_counts += table
    passive = (passive_counts[:, target] + alpha) / (passive_counts.sum(1) + y_states * alpha)
    return {
        "values": estimate.tolist(),
        "unobserved_stratum_mass": missing.tolist(),
        "below_five_samples_stratum_mass": sparse.tolist(),
        "passive_values_same_samples": passive.tolist(),
        "observed_strata": len(strata),
    }


def solve(prompt, act):
    """The sole inputs are public prompt text and legal batch feedback."""
    p = parse_prompt(prompt)
    x, y, domains = p["x"], p["y"], p["domains"]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8]))
    # Predeclared allocation, reusing the earlier ancestor-screen baseline's
    # 3,999-draw pooled multinomial bootstrap and familywise threshold.
    # This is a diagnostic heuristic, not an exact finite-sample guarantee.
    arms = sum(domains[z] for z in p["targets"])
    n = p["budget"] // (4 * arms)
    if n < 1:
        raise ValueError("The fixed diagnostic screen does not fit this task budget")
    selected, screening = [], []
    remaining = p["budget"]
    for z in p["targets"]:
        table = np.zeros((domains[z], domains[x]))
        for state in range(domains[z]):
            feedback = act(
                {
                    "type": "intervene",
                    "target": z,
                    "value": f"state_{state}",
                    "measure": [x],
                    "batch_size": n,
                }
            )
            assert feedback["type"] == "batch_result"
            histogram = feedback["batch"]["joint_histogram"]
            assert histogram["columns"] == [x]
            assert feedback["batch"]["n"] == n
            for values, count in histogram["rows"]:
                table[state, values[0]] += count
            remaining = feedback["remaining_budget"]
        pooled = table.sum(0) / table.sum()
        expected = n * pooled
        statistic = np.sum((table - expected) ** 2 / (expected + 1e-15))
        draws = rng.multinomial(n, pooled, size=(3999, domains[z]))
        expected_draws = draws.sum(1)[:, None, :] / domains[z]
        simulated = ((draws - expected_draws) ** 2 / (expected_draws + 1e-15)).sum((1, 2))
        pvalue = float((1 + np.sum(simulated >= statistic)) / 4000)
        if pvalue <= 0.05 / len(p["targets"]):
            selected.append(z)
        screening.append({"target": z, "counts": table.astype(int).tolist(), "pvalue": pvalue})
    measure = [x, y, *selected]
    possible_rows = math.prod(domains[v] for v in measure)
    counts = defaultdict(int)
    while remaining >= len(measure):
        batch = remaining // len(measure)
        if possible_rows > 128:
            batch = min(batch, 128)
        feedback = act({"type": "observe", "measure": measure, "batch_size": batch})
        assert feedback["type"] == "batch_result"
        histogram = feedback["batch"]["joint_histogram"]
        assert histogram["columns"] == measure
        for values, count in histogram["rows"]:
            counts[tuple(values)] += count
        remaining = feedback["remaining_budget"]
    estimates = adjusted_values(counts, domains[x], domains[y], p["target"])
    choose = np.argmax if p["maximize"] else np.argmin
    answer = {"type": "answer", "value": f"state_{int(choose(estimates['values']))}"}
    return {
        "answer": answer,
        "selected": selected,
        "screening": screening,
        "samples_per_screen_arm": n,
        "observation_samples": sum(counts.values()),
        "observation_width": len(measure),
        "remaining_budget": remaining,
        "same_samples_passive_answer": {
            "type": "answer",
            "value": f"state_{int(choose(estimates['passive_values_same_samples']))}",
        },
        **estimates,
    }
