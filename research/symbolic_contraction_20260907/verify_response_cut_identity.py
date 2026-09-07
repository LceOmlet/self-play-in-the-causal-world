"""Check a fixed-marginal response polytope against exact weighted cuts."""

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from cpt_world.counterfactual_solver import _ExactResponseLP  # noqa: E402

records = []
for n in range(3, 8):
    for variant in range(3):
        edges = [
            (i, j, 1 if variant == 0 else ((i + 1) * (j + 3) + variant * n) % 5)
            for i in range(n)
            for j in range(i + 1, n)
        ]
        owner = _ExactResponseLP(((0.5, 0.5),) * n)
        costs = {
            response: sum(weight for i, j, weight in edges if response[i] != response[j])
            for response in owner.responses
        }
        best = max(costs, key=costs.get)
        complement = tuple(1 - state for state in best)
        assert costs[complement] == costs[best]
        assert all((left + right) == 1 for left, right in zip(best, complement, strict=True))
        actual, _ = owner.optimize(costs, sense="maximize", time_limit_seconds=5.0)
        assert abs(actual - costs[best]) < 1e-8
        records.append(
            {
                "contexts": n,
                "variant": variant,
                "exact_cut_value": costs[best],
                "response_lp_value": actual,
                "independent_pair_upper": sum(w for _, _, w in edges),
                "half_complement_witness": [best, complement],
            }
        )
report = {
    "passed": True,
    "cases": len(records),
    "records": records,
    "scope": "Response-polytope identity; not a claim about the exact difficulty of frozen CPTs.",
}
(Path(__file__).resolve().parent / "response-cut-identity.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps({"passed": True, "cases": len(records), "triangle": records[0]}))
