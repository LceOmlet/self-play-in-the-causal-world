Official DAPO progress-control audit, 2026-09-08

Actual GPU acceptance is complete for native update15 -> update16 with the
explicit `progress-v1` source profile. All 496 finite LoRA tensors changed;
496 Adam states and the scheduler are at16, with 992 finite moments. Generated
batches and the real data cursor both advance from18 to19. The actual candidate
load method restores the next eight data rows exactly on CPU (only its GPU
weight RPC is mocked in that separate check). This is not bitwise GPU-memory
restoration or actual GPU execution across epochs. Training continues.

`gpu-update16-and-curves-evidence.tar.gz` freezes 79 members, including the real
GPU load/update logs, update15/16 acceptance, sampler states, source identity,
all raw snapshots for updates1..17, the reader correction and curve inputs.
Its SHA-256 is `868ff881acadd334e8585724d84424c0f3983793b9e6f65e75da2726cdbf2361`.
The associated manifest records each member hash; large model/optimizer files
remain preserved on the server and have their hashes in the acceptance record.

`plot_trusted_progress.py` adapts the existing plotting style. Extract the
archive's `curve-inputs` into an audit directory, then supply all three JSON
files using `--inputs`, plus `--output figures`. The six figures show raw quality,
shaped training reward, update/generation time and recorded task errors; they
do not smooth data or claim paired learning gains. The strict failure panel
counts the one incomplete retained output as failure. In-flight unretained
outputs are excluded. A malformed tool command is kept verbatim by the reader
so the entire executed command sequence can be uniquely matched: the 39 prior
joins and all official metrics remain unchanged, with one additional join.

The source-admission history below predates this real GPU acceptance.

Latest source admission: candidate-v2 preserves the original CRLF bytes. Three
`git apply` reconstruction checks and the 15 CPU control cases pass. The new
explicit `progress-v1` source profile and the unchanged default `baseline` both
verify all 427 pinned files, real imports and original train/validation hashes;
cross-profile substitutions are rejected. Exact production-admissible patch
files are in `patches/verl-dapo-progress-v1.patch` and
`patches/verl-recipe-mask-progress-v1.patch`. The v1 candidate below is retained
as historical evidence; actual GPU recovery with the admitted v2 source is
now documented above.

`candidate-v2/integration-evidence-v2.tar.gz` records all 180 members of this
phase, including the initial incorrect validation filename and its correction
from the actual existing controller. `resume_progress.py` prepares, preserves
and checks an early trusted native checkpoint, then controls only the process
switch and the original verification deadline. It never implements training.

`prepare_candidate.py` builds isolated source copies and exact patches of the
pinned upstream classes. It checks the complete numerical update block and old
logprob/mask method byte-for-byte, and compares every copied source file.

`probe_progress.py` invokes those actual upstream fit/save/load methods with the
real StatefulDataLoader and sampler. GPU generation, old-logprob, actor update,
and model checkpoint RPCs are explicitly mocked. Reward extraction, filtering,
GRPO advantage, action masks and token TIS still execute upstream. This is a
control-flow regression audit, not model training or a second trainer.

The six original control cases reproduce premature completion, a checkpoint
label one beyond the actual actor call, lost work just before an epoch boundary,
and an extra update after resuming an already completed target. Final
`candidate-04` passes all 15 cases, including a real exhausted iterator state,
the actual update-6 data cursor, the unchanged ten-batch filter error, and
split-versus-uninterrupted generation/retained-row equality.

`evidence.tar.gz` retains all raw attempts, including the initial fixture's
missing `acc` field, exact source snapshots, patch creation, process identities,
JSON results, and real dataloader states. `evidence.sha256.json` records all
member hashes. The archive's `probe_progress_v4.py` is the final probe; previous
versions are retained to make the failed and superseded attempts reviewable.

The candidate makes an explicit update target authoritative over the epoch
budget, persists actual generation/epoch counters, restores the sampler at all
boundaries, and rejects ambiguous or inconsistent saved progress. It does not
change the DAPO numerical update or suppress the existing filter-limit error.

Source base:

- verl: `23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`, with the already reviewed tool termination patch.
- verl-recipe: `ee3aef1690d6bb5e6448052c4842e8efb7a3f76c`, with the already reviewed response-mask guard.

The original candidate roots on the server are `verl-progress-candidate-v1`
and `verl-recipe-progress-candidate-v1` under
`/home/chen/vendor/dapo-official-20260906`. The current run explicitly selects
the corresponding `candidate-v2` roots and the admitted `progress-v1` manifest.

See `docs/dapo-progress-correctness-20260908.md` for the counter invariant,
reproduced failures, compatibility assumptions, and remaining proof obligations.
