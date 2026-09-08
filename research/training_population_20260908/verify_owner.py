"""Replay saved diagnostic records through the installed production reward owner.

Run in the project environment. No model, solver, environment episode, or
training process is constructed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from read_population import NUMERIC_FIELDS

from cpt_world import rewards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    differences = []
    completed = 0
    references = 0
    for group in data["groups"]:
        for trace in group["trajectories"]:
            if not trace["completed"]:
                continue
            completed += 1
            score = trace["terminal_score"]
            differences.append(
                abs(float(rewards.terminal_quality_reward(score)) - trace["quality"])
            )
            comparison = trace["reference_comparison"]
            if comparison is None:
                continue
            field, _ = NUMERIC_FIELDS[group["query_type"]]
            for prefix in ("observational", "observational_second"):
                if prefix + "_error" in comparison:
                    reference_score = {**score, field: comparison[prefix + "_error"]}
                    actual = float(rewards.terminal_quality_reward(reference_score))
                    differences.append(abs(actual - comparison[prefix + "_score"]))
                    references += 1
    assert all(math.isfinite(d) and d < 1e-12 for d in differences)
    source = Path(rewards.__file__)
    result = {
        "completed_model_scores": completed,
        "reference_scores": references,
        "max_absolute_difference": max(differences),
        "production_owner": str(source),
        "production_owner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": True,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
