"""CPU preparation against a previously retained checkpoint; never launch GPU work."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    old_run = Path("/home/chen/runs/heldout-learning-20260908/resume-no-val-01")
    source = old_run / "preserved/global_step_50"
    hold = args.root / "cpu-preflight-hold"
    hold.mkdir(parents=True, exist_ok=False)
    control.write(
        hold / "preserved.json",
        {
            "checkpoint": str(source),
            "step": 50,
            "files": {
                str(p.relative_to(source)): control.sha(p) for p in source.rglob("*") if p.is_file()
            },
        },
    )
    control.prepare(
        SimpleNamespace(
            project=args.project,
            old_run=old_run,
            hold=hold,
            run=args.root / "cpu-preflight",
            start_seed=4000000,
        )
    )


if __name__ == "__main__":
    main()
