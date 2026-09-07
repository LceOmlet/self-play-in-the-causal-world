#!/usr/bin/env python3
"""Export the existing balanced, certified task stream to verl's parquet schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import Dataset

from cpt_world import CPTWorldEnvironment, iter_random_balanced_training_rows
from cpt_world.identification import INTERACTION_SURFACE_VERSION


def convert_row(row, index):
    if row.get("environment_version") != INTERACTION_SURFACE_VERSION:
        raise ValueError("Task row belongs to an obsolete environment contract")
    environment = CPTWorldEnvironment()
    instruction = environment.reset(**row)
    prompt = [dict(message) for message in row["prompt"]]
    prompt[-1]["content"] += instruction
    return {
        "data_source": f"cpt_world/{row['query_type']}",
        "prompt": prompt,
        "agent_name": "tool_agent",
        "reward_model": {"style": "rule", "ground_truth": "environment_owned"},
        "extra_info": {
            "index": index,
            "query_type": row["query_type"],
            "tape_key": row["tape_key"],
            "environment_version": row["environment_version"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"act": {"create_kwargs": {"row_json": json.dumps(row)}}},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()
    if args.count <= 0 or args.count % 5:
        raise ValueError("Use a positive multiple of five to preserve task-family balance")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = iter_random_balanced_training_rows(start_seed=args.start_seed)
    rows = [convert_row(row, i) for i, row in enumerate(islice(source, args.count))]
    source.close()
    Dataset.from_list(rows).to_parquet(str(args.output))
    report = {
        "count": len(rows),
        "start_seed": args.start_seed,
        "environment_version": INTERACTION_SURFACE_VERSION,
        "families": dict(Counter(row["extra_info"]["query_type"] for row in rows)),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
