# Reproducible five-family benchmark, 2026-10-19

This directory freezes 100 fresh native tasks from each of five families (500
total), and a predetermined subset of 20 per family (100 total) for a paired
three-way evaluation: isolated agents, original Qwen3.5-9B, and checkpoint-2350.

Dataset SHA256: `c8cf45ae84272984bfd707982da7c8f2d8ae320f2f90820375595e6e970f2de5`.

## What is frozen

`data/tasks/*.json.gz` contains each explicit causal world, exact round-trip
binary64 CPT entries, original query, measurement/intervention permissions,
budget, OutcomeTape key, prepared protocol-v1.1 prompt, and native scoring truth.
The frozen owner source and protocol are in `frozen/`. Each compressed file,
canonical capsule, owner file, and evaluation selection has a SHA256 in
`data/manifest.json`. The benchmark does not depend on regenerating a seed or
repeating a time-limited counterfactual admission search.

Generation begins at seed 202610210000 with a 10000 family stride. Native
counterfactual certification rejects candidates under the unchanged native
five-second admission rule; rejected attempts are recorded. No outcomes or
evaluation scores are used to select cases. Best-intervention has 20 concordant
and 80 strong-reversal cases; its evaluated subset has 4 and 16 respectively.
`data/eval100.json` fixes the subset by SHA256 ranking of family and ordinal.

Restore recreates the original native objects, including mask iteration order
and native mediator truth tuple types. It restores cached native CF bounds;
it does **not** claim an independent mathematical recertification of those
bounds. The native solver's endpoint certificate tolerance remains 0.001.

## Verify and restore

From the repository root, a dependency-free downloaded-byte check is:

```sh
python benchmarks/repro500_20261019/scripts/verify_hashes.py
```

Full verification requires Python 3.12, NumPy 2.3.5 and PySCIPOpt 6.2.1.
The exact generation runtime versions are recorded in the manifest. Then:

```sh
python benchmarks/repro500_20261019/tests_restore.py
python benchmarks/repro500_20261019/verify_suite.py --verify-replays
```

`capsule.restore_episode(capsule)` restores one native episode without sampling;
`FrozenEnvironment.reset(capsule=capsule, **capsule['row'])` restores the public
environment. Counterfactual permissions and native scoring remain enforced.

The recorded verification covers all 500 capsules, two independent cold
processes, exact public feedback/budget/native-score agreement, and seven
synthetic restore regression tests. See `data/restoration-verification.json`
for runtime and verification provenance. Its replay digest is
`34b140909f6bef2da8f5e082c57e9a22e60a0300a335bcae6fda00086cdd5eea`.

## Sealed evaluation

`server.py` provides the same act/numerics owner to five isolated family agents
and the 9B runner. A role can open only its assigned IDs. Qwen cannot open any
case until every agent case is terminal. At the user's explicit request both
base and trained weights are evaluated. Results remain private until all three
100-case runs finish. Requests are journaled before execution and UUID retries
are deduplicated within the running service. Do not restart a live exam or
retry an experiment under a new UUID after a transport interruption.

The workstation `public_client.py` first uses an optional persistent SSH local
forward, `127.0.0.1:28429` to the original owner's `127.0.0.1:18729`. If the
forward is absent or disconnects, it falls back to the original SSH transport
with the exact same payload and UUID; SSH timeouts also retain that UUID.
This transport optimization was introduced during the agent phase (first
post-change status: 44 completed cases). Task commands, owner execution,
budgets and model settings are unchanged. `scripts/tests_transport.py` verifies
the happy path and disconnected-tunnel/SSH-timeout recovery synthetically.
An execution tool's returned `session_id` means the local client is still
running: poll that process for its result before attempting transport recovery.

The service normalizes public prompt/tool-history accounting to the Qwen
tokenizer with a 32768-token ceiling. Qwen additionally counts its complete
native message context, including generated assistant text. Agents use five
independent conversations, one per family of 20 cases, and receive no owner
source, capsules, answers, other-agent solutions, or parent solution advice.
The solver runtime does not expose a context-reset tool: each agent can retain
its own conversation across its 20 cases, whereas Qwen starts fresh per case.
Numerical state is fresh per case for both. Thus this compares agent systems,
not strictly matched reasoning-token/context budgets. Only the current case's
public prompt and acquired feedback are permitted as question evidence.
Local Python is transport only; all analysis computations use numerics.

Each numerical session uses NsJail, 1 CPU, 1 GiB, no external network/host
files, persistent per-answer NumPy/SciPy, 30 seconds per call and 8192 output
bytes. `frozen/protocol/sandbox_bridge.py` and `sandbox_tool_mode.py` preserve
the deployed in-sandbox adapters (the latter is installed as `tool_mode.py`).
`results/numeric-runtime.json` fingerprints the installed read-only rootfs and
its package versions. Linux NsJail/cgroup-v2 provisioning at
`/opt/cpt-numerics` is a deployment prerequisite, not included as an OS image.
Run the exam in a `cpt-numerics-*.service` systemd user unit with `Delegate=yes`.
The numerical library installation is separate from the owner Python venv;
null package entries in the owner manifest do not describe the sandbox.

`scripts/preflight.py` checks the real numerical bridge, idempotent budget
charging, per-role completion and phase gating using synthetic cases only.
Its report is `results/synthetic-preflight.json`.

## 9B profile and reproducible invocation

`model_profile.json` records checkpoint-2350, base model and adapter paths,
thinking=false (as in its training configuration), temperature=0,
seed=20261019, 32K context, at most 8192 generated tokens per turn, four
concurrent cases, and no model sampling retries. A failed or truncated answer
is retained, never resampled. Each case gets its own transcript and numerical
session. Model files remain in the existing model/checkpoint store.

Both weight sets use this same profile (thinking=false, temperature=0, seed,
context and tool limits). The base model is already served alongside the LoRA
model by the same inference server. Model/config/tokenizer file hashes for
both are recorded in `results/model-identity.json`.

```sh
python benchmarks/repro500_20261019/run_9b.py --self-test
python benchmarks/repro500_20261019/serve_9b.py --model checkpoint2350
python benchmarks/repro500_20261019/run_9b.py --model checkpoint2350 \
  --owner-url http://127.0.0.1:18729/PRIVATE_ROLE_TOKEN \
  --index benchmarks/repro500_20261019/data/eval100.json \
  --output-dir NEW_TRANSCRIPT_DIRECTORY
```

Use the tested inference venv on PATH for serving. `serve_9b.py` applies only
a process-local platform override for the host's NVML/kernel mismatch.
No installed package, driver, model weight, or training state is modified.
Do not commit runtime role credentials or numerical private sockets.

After the original agent phase finishes, `serve_base_owner.py` starts a second
owner on port 18730 in a new runtime directory. It verifies the original
completed-agent phase and fingerprints its opaque completion records, but
does not import scores, episodes or numerical state. Run it in its own
`cpt-numerics-*.service` with `Delegate=yes`:

```sh
python benchmarks/repro500_20261019/serve_base_owner.py \
  --agent-runtime ORIGINAL_EXAM_RUNTIME --runtime NEW_BASE_RUNTIME
python benchmarks/repro500_20261019/run_9b.py --model base \
  --owner-url http://127.0.0.1:18730/PRIVATE_BASE_ROLE_TOKEN \
  --index benchmarks/repro500_20261019/data/eval100.json \
  --output-dir NEW_BASE_TRANSCRIPT_DIRECTORY
```

Only after all 300 cases terminate, export the three-way comparison:

```sh
python benchmarks/repro500_20261019/scripts/compile_threeway.py \
  --agent-runtime ORIGINAL_EXAM_RUNTIME \
  --trained-transcripts TRAINED_TRANSCRIPT_DIRECTORY \
  --base-runtime BASE_EXAM_RUNTIME --base-transcripts BASE_TRANSCRIPT_DIRECTORY \
  --output NEW_RESULTS_DIRECTORY
```

`scripts/tests_export.py` checks the unsealing gates, identical case pairing,
summary-file handling, metrics and archives using 100 synthetic cases only.
The main table reports native reward over all assigned cases, with unfinished
cases scoring zero. A second table reports native TV error / CF endpoint MAE
over submitted cases (with the submitted denominator shown), and optimal
action / valid adjustment set / joint mediator-and-order exact counts over
all 20 assigned cases. These diagnostics were specified before unsealing any
formal score; continuous reward is not presented as an accuracy percentage.
`scripts/tests_base_gate.py` checks the base-owner admission proof without
opening any real capsule. Preparation and synthetic tests are not evaluation
results. The final three-way table and raw transcript provenance will be
added after all three runs complete.
