"""Expand the candidate manifest with explicit generic calibration seeds."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from cpt_world import (
    QUERY_TYPES,
    WorldGrammar,
    candidate_seed_manifest_path,
    iter_sampled_seeds,
    load_candidate_seed_manifest,
)


def main() -> None:
    manifest_path = candidate_seed_manifest_path()
    existing = [asdict(seed) for seed in load_candidate_seed_manifest(manifest_path)]
    sampled = iter_sampled_seeds(
        WorldGrammar(),
        count=20,
        query_types=tuple(QUERY_TYPES),
    )
    existing_ids = {seed["seed_id"] for seed in existing}
    merged = existing + [seed for seed in sampled if seed["seed_id"] not in existing_ids]
    out_path = Path(manifest_path).with_name("candidate-v1-expanded.json")
    out_path.write_text(
        json.dumps({"schema": "cpt-world-seed-manifest-v1", "seeds": merged}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path} with {len(merged)} seeds ({len(sampled)} sampled)")


if __name__ == "__main__":
    main()
