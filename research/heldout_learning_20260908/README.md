Fixed-task interpretation and matched official evaluation preparation.

The full official update25 validation has now finished and passed the complete
reader path:25 uniquely matched tasks,11 submitted answers,14 unfinished outputs.
All14 unfinished trajectories hit the response limit:11 reach exactly30,720
response attention tokens;3 stop before appending feedback at30,699/30,657/30,665.
Exact negative upstream length penalties recover these lengths. Full output,
owner events and the inspected fixed source establish the stop paths; this is
not an approximate retokenization or newly added runtime instrumentation.

`inspect_validation_stops.py`, `update25-validation.json` and `update25-stops.json`
preserve the actual checks and per-task results. All three discrete task families
have strict success0/5. No before/after learning effect is claimed yet.

After full validation was logged, native update25 was preserved and the training
process deliberately paused. Official original-base `val_only` is now live as
PID467771 with step0 validation actually interacting with the original tasks.
Its full result and the subsequent native GPU training resume remain pending.

`update25-validation-and-base-launch-evidence.tar.gz` contains96 members,
893,900 bytes, SHA-256
`6b688cb8cf3c51a9a7479adcc496c26ce91a347d4ab76e75ab2d8eba327a52d1`.
See `docs/update25-validation-stops-20260908.md`. Earlier preparation/partial
states below describe the preceding stages, not the latest execution state.

`control_matched_evaluation.py` reuses the accepted native process/checkpoint
controller to prepare recovery, wait for fully logged validation before pausing,
run the official original-base `val_only` entry point, and resume native training.
Only CPU preparation/configuration and the pending-validation guard have been
executed so far. No base GPU result or GPU restore is claimed by this preparation.

`read_validation.py` joins the full public user prompt and exact executed command
sequence. It requires all25 outputs plus finished official metrics before reporting
aggregate performance, and retains actual task-error coverage. The partial branch
has been exercised on frozen snapshots; the complete branch awaits real outputs.
Snapshot03 contains20 started task traces and19 ended identifiable owner scores:
7 submitted answers,12 no terminal answer. These are partial counts, not a full
validation failure rate or an explanation of why generation ended.

`matched-evaluation-preparation-evidence.tar.gz` freezes64 members (1,028,587 bytes),
SHA-256 `dcdc2bec0079bcdf4916e44d487660e9e305e541d738365c47e6fa76608fd9fc`.
It includes controller/reader versions, actual preparation logs and configuration,
native data/progress state and checkpoint acceptance, and two partial snapshots.
The archive's scope precedes any pause or base GPU launch. Large native weights
remain in the preserved server directory and are identified by acceptance hashes.

Update25 has now been preserved and verified:496 finite LoRA tensors,496 Adam
states at25,992 finite moments, scheduler25, generation/data cursor29. The
actual official data-load method restores the next8 rows identically on CPU;
only the GPU weight RPC is mocked in this separate check. The live TaskRunner
has entered greedy validation at global step25; this is not a completed pass.
`preserved-update25-acceptance.json` records native file hashes and the retained
server path. `update25-validation-started-evidence.tar.gz` preserves21 members
(468,190 bytes), SHA-256
`31a6ad4a0cc25402adb00de2c23001eac2f3706758874055360d2c5dafc9c301`.

`audit_fixed_tasks.py` uses all 25 original validation tasks, verifies the dataset,
admission record and frozen truth hashes, and submits constant and population
observation answers to the production environment/scorer. Population observation
uses hidden probabilities; it is not model performance or a finite-budget solver.
`fixed-task-baselines.json` contains every answer and task metric.

`check_backdoor_advantages.py` invokes the actual pinned upstream GRPO function
on four real retained backdoor groups. For completed, unpenalized answers,
`q=1-d/14` gives `A=-(d-mean(d))/(std(d)+14*epsilon)`. Its apparent score compression
does not compress normalized advantages by14. FP32 comparison with negative
distance changes advantages by at most4.02e-5 in these groups. The incomplete
step17 group is excluded, not assigned an invented task error.

For mean raw quality Q and valid-completion probability p, the existing scoring
definition only guarantees `max(0,14Q-13) <= p <= Q`. Thus Q=0.9 supplies no
positive lower bound on valid-completion probability. This explains the need
for strict metrics without treating affine scaling as a suppressed gradient.

`prepare_base_validation.py` resolves and verifies the official base `val_only`
command against the actual running configuration. It changes only initialization,
validation-only control and run/output paths. It does not launch GPU work,
implement a validator/trainer, or claim a completed baseline. The intended GPU
run must still verify source/data identities and exclusive GPU availability.

`evidence.tar.gz` contains70 members (1,317,170 bytes), SHA-256
`390e30c0f23894ae25658a49b28d7001be8e345fb467cb5b7cd6118c21e73bd2`.
It freezes raw source/configuration checks, both versions of the fixed-task audit,
all original validation truth files and admission metadata, the actual advantage
inputs/outputs and the through21 training snapshot. No trained model weights are
in the archive. Per-member hashes are in `evidence.sha256.json`.

At archive time, main PID400557 was still live with checkpoint24 published.
The full step25 validation and matched original-base GPU results were pending.
See `docs/heldout-learning-and-reward-scale-20260908.md` for the derivation,
metric interpretation, remaining execution work and scope limitations.
