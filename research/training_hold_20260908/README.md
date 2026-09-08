Actual updates through 39 and research-priority correction, 2026-09-08

The explicit hold stopped the continuous verification after published update 39.
The user's subsequent permission to train while investigating supersedes that
hold. The official job resumed at native 39 and published update 40. No shadow
trainer, reward change, task generator change or orientation experiment ran.

`preserve_and_stop.py` binds process identity, copies/hashes the published native
checkpoint, then uses the existing audited process-tree terminator.
`resume_after_hold.py` checks the stopped identities, native content, exact
configuration differences, source provenance and official CPU data restoration,
then calls the existing official launcher. Only GPU RPC is mocked during the
separate CPU audit. The actual GPU job loads weights, Adam, RNG and scheduler.
Its verification deadline and 500-update target are inherited; the final 10000
requirement remains pending in the full goal.

`training-signal-through39.json` comes from the existing capture reader and the
pinned official scalar advantage function. It covers 39 actual updates, 45
generated groups and 156 retained trajectories; 153 submitted answers. This is a
selected training cohort, not a matched validation improvement.

`inspect_new_successes.py` checks all three newly correct strong-decision
trajectories in updates 30--39. It compares actual histogram counts and the model's
recorded formula with production population quantities. It does not send hidden
truth to the model or treat an accidentally correct answer as an algorithm bug.

The 112-member archive is 2,269,553 bytes, SHA-256:
`3f13aaa1c3f090eb8e148700826ab323b31789ea4dd7d1dc49323fc3c18b0d98`.
It contains the four complete snapshots, full new successful outputs/events,
two original task worlds, failed first capture invocation, native 39 hashes/data,
hold and resumed configurations/logs, source code and live identities. Full LoRA
and Adam tensors remain on the server under `user-hold-01/global_step_39` and the
resumed run's `preserved/global_step_39`.

Executed sources are archived exactly. Repository versions only wrap lines;
all three Python ASTs were checked equal, so the recorded execution hashes must
be resolved against `executed-sources/`, not assumed equal to reformatted files.
Run `python research/training_hold_20260908/verify_evidence.py` for byte/AST/count
verification without importing training libraries or accessing the server.

The direction of further work is recorded in
`docs/research-priorities-and-learning-evidence-20260908.md`. The earlier weak-edge
certificates remain boundary results, and observation-rank comparisons remain
evaluation caveats. Neither is declared the chief cause of current learning
performance. A/A2, C/D, B/E and the final 10000-update requirement remain open.
