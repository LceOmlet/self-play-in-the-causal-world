Verified official DAPO training submission, 2026-09-07

This directory now records historical evidence. The fixed 500 training worlds
were an unintended change made by the project integration, not a user request
or an official DAPO requirement. Current training uses a persistent stream of
fresh tasks; see `docs/continuous-training-restoration-20260908.md` and
`research/long_run_10000_20260908/control.py` for the accepted long-run controller.

The actual deployed training command is the official launcher
`scripts/run_official_dapo.sh`, with `trainer.total_training_steps=10000` and
`trainer.resume_mode=resume_path`. The preserved deployment command is recorded
in `research/long_run_prepare_20260908/deployment/prepared.json`; its matching
`resolved.yaml` binds the run's `stream.json` to `CPTWorldStreamingDataset`.
Use that accepted prepared configuration for continuation.

Repository integration, 2026-09-08: the private evidence checkout lacked the
already tracked and deployed long-run controller. Its `control.py` and `README.md`
were restored byte-for-byte from the public worktree after checking the controller
SHA-256 against the deployment receipt:
`baaea5240e4255f81b4d0fdf8417db7d4e7ea3d2f34c4e02f3982d601b57ddb4`.
This source synchronization did not deploy code or restart training.

Direct execution of `prepare.py` and every `control.py` action is disabled.
`--help` explains the replacement; default invocation and old action arguments
exit with an error before preparing data or controlling any process. The old
controller's importable functions, including `environment()`, remain unchanged
because the continuous controller reuses that helper and overrides the training
data path with its stream descriptor. No validation-dataset support, generator
family rotation, or training algorithm was changed. These research entry points
are outside the live stream's environment/converter source fingerprint list.

The user authorized actual training after inspecting a few real trajectories,
task value, computability and the RL execution path. See
`docs/training-submission-20260907.md` for results and their limits.

`prepare.py` calls the existing full task generator and official data exporter.
Its CF observer calls the original isolated solver once and returns its result
or re-raises its error unchanged. It records all accepted and rejected attempts.
It is neither an alternative generator nor a solver implementation.

`control.py` checks data/source hashes and invokes the pinned official launcher.
It does not implement an RL algorithm. The submitted job starts the original
Qwen3.5-9B with fresh LoRA, 500 training worlds and 25 disjoint validation worlds.
It uses the already verified one-prompt/four-trajectory update configuration,
a 500-update limit and a 24-hour wall limit. Limits do not establish learnability.
The expanded validation-only pilot was stopped at the user's direction. Its
partial records are retained, not treated as a completed baseline evaluation.

`analyze_case.py` reproduces the completed pilot model answer's numerical match
to its first passive histogram. `task-value-witnesses.json` identifies two
selected real, solvable strong-reversal tasks and their public trace hashes;
the full earlier cohort, including failures, remains archived separately.

The first actual update and partial subsequent generation are frozen separately
in `first-update.tar.gz`, with hash `first-update.sha256.json`. `capture_progress.py`
reads existing logs only and joins official dumps to tool events by complete
command sequences when the runtime omits request IDs. `compare_speed.py` compares
the first update with the archived legacy log; see `speed-comparison.json` and
`docs/first-update-and-speed-20260907.md` for metrics and causal limits.

`freeze_evidence.py` produces these immutable archives on the training host:

- `certified-data.tar.gz`: all 525 serialized worlds/truths, official parquet
  files, acceptance/hash records, preparation output and all CF rejections.
- `training-start.tar.gz`: the executed controller, trajectory analysis,
  preflight checks, source fingerprints, launch/process identity and a stable
  prefix of the training log. It is explicitly not the eventual training result.
- `../base_signal_diagnostic_20260907/stopped-evidence.tar.gz`: original pilot
  tool events, world truths, control/analysis code, stop and terminal records.

`evidence-manifest.json` records archive byte counts and SHA-256. The pilot has
its own `stopped-evidence.sha256.json`. Recompute hashes before extracting any
serialized evidence, and only load trusted pickle files from this archive.

Archived original run (no longer the current training run):
`/home/chen/runs/training-submission-20260907/run-01`.
`launch.json` identifies source HEAD `3f83d21`, the exact command and controller
hash; `process.json` records PID and creation time. The supervisor writes
`exit.json` only on actual termination. Inspect that identity and raw events
with `../base_signal_diagnostic_20260907/status.py --run <run>`; do not restart
an invocation merely because a remote status request times out.
