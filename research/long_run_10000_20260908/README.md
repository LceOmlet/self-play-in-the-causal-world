# Continue official DAPO to 10,000 actual updates

`control.py` prepares the existing trusted continuous-training chain for an
absolute target of 10,000 completed optimizer updates. It preserves the official
launcher, recipe, numerical configuration and update implementation. Preparation
does not stop training or launch a model.

The accepted source at `e87bea8` has now been deployed. The actual GPU run resumed
the preserved native checkpoint at update 96 and generation count 110, retaining
the continuous task stream and cached tasks. Its first new update, 97, completed
and was saved with generation count 111 and journal next index 112. The observed
gradient norm was `0.1396484375`, learning rate `1e-6`, mean reward `0.26984128`,
and step duration `533.94` seconds. This establishes successful native GPU
continuation; it does not establish improved learning or completion of 10,000
updates. No additional standalone model validation was run.

The only training setting changed is `trainer.total_training_steps=10000`.
Runtime/output/source paths follow the new run. The existing constant learning
rate of `1e-6`, zero warmup, disabled separate validation, and native Adam,
scheduler and RNG state are retained. The accepted `progress-v1` recipe gives an
explicit update target precedence over `total_epochs`; its generation counter
includes dynamically filtered groups and is never substituted for update count.
The supervisor uses `process.wait()` without a wall-clock timeout. It records a
completed target only when the native checkpoint progress reaches 10,000.
The deployed run uses this cumulative target and has no inherited 500-update or
24-hour cutoff. Separate fixed validation remains disabled.

Preparation deliberately supports the already accepted counterfactual-kernel
source migration. It calls
`research/terminal_packing_20260908/migrate_stream_source.py`, which preserves all
cached tasks, response truths, generator cursors and native training files;
only the journal/source identity in `data.pt` is rebound. The original checkpoint
and journal remain evidence. Old and new source trees must differ in exactly the
accepted kernel bytes, and the acceptance receipt must bind those bytes.

Use the existing continuous controller to preserve the latest complete native
checkpoint at a chosen restart boundary. Then prepare this run against that
preservation; an offline CPU preflight may use an isolated, quiescent fixture.
Preparation never invokes the existing controller's pause or launch functions.

The real CPU preflight has passed using a copied native checkpoint at update 94
and generation count 107. The official load hook retained both counters; the
already generated next task was retained in the migrated journal. Resolved
configuration changed only the update target and project/run paths. All 427
pinned official source files passed verification. This preflight did not stop
the original run, execute model inference, or authorize launching concurrently.

```sh
python research/long_run_10000_20260908/control.py prepare \
  --project /path/to/accepted-project \
  --old-run /path/to/previous-run \
  --hold /path/to/preserved-native-checkpoint \
  --run /path/to/new-run \
  --evidence /path/to/kernel-acceptance.json
```

Review `prepared.json`, `checkpoint-acceptance.json`, `resolved.yaml` and
`preflight-source-verification.json` before the separate `launch` action. The
configuration audit rejects algorithm, reward, model, batch-size, scheduler or
validation changes. Launch requires the prior process identities in the hold's
`stopped.json` to have exited. A CPU fixture is not permission to launch while
the original training remains active.

```sh
python research/long_run_10000_20260908/control.py launch \
  --project /path/to/accepted-project --run /path/to/new-run
```

Uncertified counterfactual tasks continue to be rejected by the task generator.
Outstanding coverage, global-calibration or learning-efficiency research does
not invalidate certified tasks by itself. Actual wrong targets, broken source
or journal identity, inconsistent native progress/state, or nonfinite updates
require investigation before continuing the affected training chain.
