"""Create the first anonymous candidate seed manifest from real world sources.

Sources:
- CLadder meta-models subset: data/worlds/cladder/meta-models-subset.json
- bnlearn BIF files: data/worlds/bnlearn/*.bif

The manifest keeps the true world for the owner, but the visible_schema only
contains opaque variable labels and state labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
WORLDS = DATA / "worlds"
OUT = DATA / "seeds" / "candidate-v1.json"

_LETTER_POOL = "DEFGHIJKLMNOPQRSTUVW"


def _labels(n: int, seed_id: str) -> list[str]:
    """Deterministic opaque 3-letter labels with no ordered single letters."""
    labels: list[str] = []
    for index in range(n):
        while True:
            digest = hashlib.sha256(
                f"cpt-world-seed-labels-v1\0{seed_id}\0{index}\0{len(labels)}".encode()
            ).digest()
            token = "".join(_LETTER_POOL[byte % len(_LETTER_POOL)] for byte in digest[:3])
            if token not in labels:
                labels.append(token)
                break
    return labels


def _cladder_seed(
    model_id: int, seed_id: str, query: dict, task: dict, manipulable: list[str]
) -> dict:
    models = json.loads(
        (WORLDS / "cladder" / "meta-models-subset.json").read_text(encoding="utf-8")
    )
    model = next(m for m in models if m["model_id"] == model_id)
    variable_names = []
    for token in model["structure"].replace(" ", "").split(","):
        for name in token.split("->"):
            if name not in variable_names:
                variable_names.append(name)
    labels = _labels(len(variable_names), seed_id)
    mapping = dict(zip(variable_names, labels, strict=True))
    return {
        "seed_id": seed_id,
        "world_source": {
            "type": "cladder_meta_model",
            "file": "data/worlds/cladder/meta-models-subset.json",
            "model_id": model_id,
            "graph_id": model["graph_id"],
            "story_id": model["story_id"],
            "structure": model["structure"],
            "params": model["params"],
            "groundtruth": model["groundtruth"],
        },
        "manipulability": {name: (name in manipulable) for name in variable_names},
        "readable": {name: True for name in variable_names},
        "visible_schema": {
            "variable_labels": mapping,
            "variables": [
                {"label": mapping[name], "states": ["state_0", "state_1"]}
                for name in variable_names
            ],
        },
        "query": query,
        "task_head": task,
    }


def _bif_seed(
    seed_id: str,
    bif_name: str,
    query: dict,
    task: dict,
    manipulable: list[str],
    readonly: list[str],
) -> dict:
    bif = (WORLDS / "bnlearn" / bif_name).read_text(encoding="utf-8")
    # Parse variable declarations in file order.
    import re

    variable_names = re.findall(r"variable\s+(\w+)\s*\{", bif)
    variable_states = {
        name: re.findall(r"\{([^}]*)\}", block)[0].replace(",", " ").split()
        for name, block in re.findall(r"variable\s+(\w+)\s*\{(.*?)\n\}", bif, re.S)
    }
    labels = _labels(len(variable_names), seed_id)
    mapping = dict(zip(variable_names, labels, strict=True))
    return {
        "seed_id": seed_id,
        "world_source": {
            "type": "bnlearn_bif",
            "file": f"data/worlds/bnlearn/{bif_name}",
            "bif": bif,
        },
        "manipulability": {name: (name in manipulable) for name in variable_names},
        "readable": {name: True for name in variable_names},
        "visible_schema": {
            "variable_labels": mapping,
            "variables": [
                {
                    "label": mapping[name],
                    "states": [f"state_{i}" for i in range(len(variable_states[name]))],
                }
                for name in variable_names
            ],
        },
        "query": query,
        "task_head": task,
    }


def main() -> None:
    seeds = [
        _cladder_seed(
            810,
            "SEED-CL-CONF-ATE",
            {"type": "ate", "treatment": "X", "outcome": "Y", "outcome_state": 1},
            {"head": "target_query", "answer": "continuous_effect"},
            ["V1"],
        ),
        _cladder_seed(
            60,
            "SEED-CL-DIAMONDCUT-BACKADJ",
            {"type": "backadj_minimal_sets", "treatment": "X", "outcome": "Y"},
            {"head": "discovery", "answer": "adjustment_sets"},
            ["X"],
        ),
        _bif_seed(
            "SEED-BN-CANCER-BESTINT",
            "cancer.bif",
            {
                "type": "best_intervention",
                "decision_target": "Cancer",
                "outcome": "Dyspnoea",
                "target_state": "True",
                "objective": "minimize",
            },
            {"head": "decision", "answer": "single_intervention"},
            ["Pollution", "Smoker"],
            ["Xray", "Dyspnoea"],
        ),
        _bif_seed(
            "SEED-BN-ASIA-MEDIATOR",
            "asia.bif",
            {
                "type": "mediator_set",
                "treatment": "smoke",
                "outcome": "dysp",
                "target_state": "yes",
            },
            {"head": "discovery", "answer": "mediators_with_order"},
            ["asia", "tub", "smoke", "lung", "bronc", "either"],
            ["xray", "dysp"],
        ),
        _bif_seed(
            "SEED-BN-SURVEY-BESTINT",
            "survey.bif",
            {
                "type": "best_intervention",
                "decision_target": "R",
                "outcome": "T",
                "target_state": "car",
                "objective": "maximize",
            },
            {"head": "decision", "answer": "single_intervention"},
            ["E", "O"],
            ["A", "S", "T"],
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"schema": "cpt-world-seed-manifest-v1", "seeds": seeds}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT} with {len(seeds)} seeds")


if __name__ == "__main__":
    main()
