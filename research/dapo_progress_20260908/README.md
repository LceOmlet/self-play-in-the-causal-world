Official DAPO progress-control audit, 2026-09-08

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

Candidate roots on the server are `verl-progress-candidate-v1` and
`verl-recipe-progress-candidate-v1` under `/home/chen/vendor/dapo-official-20260906`.
The candidate remains outside the production provenance manifest until its
exact patches and real GPU recovery have been admitted and verified. Current
compiled verification training continues on its original source roots.

See `docs/dapo-progress-correctness-20260908.md` for the counter invariant,
reproduced failures, compatibility assumptions, and remaining proof obligations.
