Original-base signal diagnostic, 2026-09-07

This directory contains the controller for an actual **validation-only** run of
the pinned official `dapo.main_dapo`. It does not implement or replace the RL
algorithm. Production long training remains stopped; no historical checkpoint
is loaded. `trainer.val_before_train=true` and `trainer.val_only=true` return
from the official trainer immediately after initial validation.

The selected pilot is the first 25 rows, in their existing order, of the frozen
250-world diagnostic dataset. It has five worlds per task family and one
concordant/four strongly discordant decision tasks. Every world is evaluated
four times at the existing training sampling temperature/top-p. This pilot
measures completion, protocol use, within-world answer/reward variation and
task errors. It cannot establish population accuracy, generalization, or
learnability. It is not a replacement for the withdrawn ten-update proposal.

Before launching, all 25 current world truths were recomputed and oracle
terminal scoring was checked at quality one, including cached CF endpoint
consistency. The official source verifier and 20 interface/provenance/tool
termination tests passed (13.57 seconds). No global Python observer is enabled.
Preflight tool events use a separate directory from actual validation events.

Authoritative output directory:
`/home/chen/runs/base-signal-diagnostic-20260907/run-01`.
The controller was copied to the parent directory there before execution.
`launch.json` records the exact command, controller hash, source commit and
sampling counts. `process.json` records PID **and creation time**; `exit.json`
is the supervisor's terminal result. The run has a 5,400-second wall limit,
which is a resource bound, never an evidence sufficiency criterion. On expiry,
the supervisor terminates only this invocation's process group/descendants.
Never infer completion from missing output or restart on observation timeout.

At this commit the evaluation is running; no aggregate result is claimed.
Read `status.py` against the live run to inspect its exact state. Raw tool events
are in `environment/`, official final outputs in `validation/`, and recomputed
worlds/truths in `truth/`. Results and provenance will be archived after the run
becomes terminal, including failures or partial completion if applicable.
