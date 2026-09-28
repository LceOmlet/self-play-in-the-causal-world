# Fresh numerical-tool training and hourly monitoring

This entry starts from the pretrained Qwen3.5-9B base, new FP32 LoRA, empty Adam
and RL step zero; it never resumes an old RL checkpoint. It is deployed for
training, but no controlled claim about numerical-tool learning gains or plateau
resolution is made. The target is 10000 native optimizer steps with an infinite
stream of the original five task families.

The runtime is explicitly versioned **TRL 1.14.0+cpt.rollout1**, using the
official `GRPOTrainer` with `loss_type=dapo`. The registered scheduling patch is
documented in `../rollout_efficiency_repair_20260928/`; it is not an unmodified
official release or the complete DAPO dynamic-filtering recipe. The production
entry has no Trainer subclass, method replacement or custom rollout function.

Task semantics come from commit `85cb577c73b17ccc6cde3bc6944db3ce2e80376f`, not
the newer environment at this public branch's HEAD. Install that task package
and the published `official_train.py` entry from the scheduling-repair directory
as `cpt_world.official_train` in the explicitly pinned runtime. `require_contract`
checks the dependency versions. Keep each deployed source/runtime immutable.

Four answers form one task group. Original sampling, permissions, terminal
rewards and the 32768 total-context / 31497 configured completion limits remain.
Tool definitions/prompts consume context space; generated length metrics are
not a measure of total context occupancy. Only `act` and the isolated numerical
tool are registered. See `../numeric_tool_mode_20260928/README.md` for isolation.

## Launch layout

Provision and verify the numerical sandbox first. In a new trusted run root,
put `train_from_zero.py`, `check_trainer_tools.py`, `tool_mode.py`,
`secure_backend.py`, and `nsjail-secure.cfg.in` into `source/`. Set
`CPT_MODEL_DIR` to the local model path for both validation and training.
Use the same explicit runtime, and a dedicated `cpt-numerics-*.service` user
service with `Delegate=yes`, sufficient total RAM/tasks, and `Restart=no`.

Run `check_trainer_tools.py` first under such a service. It uses the actual
official Trainer, a tiny random CPU model, a scripted public rollout fixture and
real NsJail sessions. It checks native discovery/dispatch, four-answer isolation,
state retention, output masks and original reward ownership, writes
`preflight-passed.json`, and does not train. It does not establish autonomous
Qwen performance. Then execute `python -I source/train_from_zero.py` with the
working directory set to the run root and no `PYTHONPATH` overlay.

The callback records native loss/metrics, source hashes, process identity and
fresh-start proof. The first warmup optimizer step can use zero learning rate;
the first **observed nonzero parameter change** is recorded separately. Complete
checkpoints are verified before publishing their saved step. The first step and
first observed parameter change trigger an extra checkpoint; normal cadence is
every 50 steps. Callbacks do not replace optimization mathematics.

## Read-only telemetry and reports

Copy `capture_remote.py`, `budget_metrics.py`, and `resource_sampler.py` to the
remote run root. Start a separate user service for the sampler; this is
telemetry only, not an automation that trains or sends reports:

```bash
python -I resource_sampler.py --root /absolute/run \
  --unit cpt-numerics-training.service --interval 60
```

The service can use `Nice=10`, `CPUQuota=5%`, `MemoryMax=128M`, `TasksMax=32`,
`NoNewPrivileges=yes`, and `Restart=no`. It samples the training cgroup
(including its answer sandboxes), GPU 0 and host memory without loading a model.
CPU cores used are a cumulative-counter delta divided by elapsed monotonic
time. Counter resets, PID/cgroup recreation and missing samples are not zeros.
GPU utilization, VRAM, power and RAM are snapshots, not full-hour means. Missing
historical resource data are not backfilled.

On the reporting machine, install matplotlib and run:

```bash
python research/numeric_training_from_zero_20260928/monitor_once.py \
  --host user@host --port 22 --remote-root /absolute/run \
  --remote-python /absolute/runtime/bin/python \
  --unit cpt-numerics-training.service
```

The remote runtime also needs its existing pandas/Parquet dependency. Every
capture gets a new timestamped directory. SSH uses batch authentication,
connection timeout and keepalives. The archive SHA256 and file manifest are
verified, traversal/links/special files rejected, and previous evidence preserved.
Failures write `latest-check.json` with the last trusted time. An old image must
never be reported as a fresh observation.

The report emits `summary.json` and five figures, copied to `latest/`:

| Figure | Definition |
|---|---|
| `rewards.png` | Five families' original terminal reward; one point is a four-answer group |
| `task-errors.png` | Native ATE/CF pooled MSE then square root; count-weighted normalized regret; valid terminal coverage |
| `resources.png` | Minute GPU/VRAM/power, cgroup CPU/RAM and filesystem free space |
| `rollout-length.png` | Native generated-token mean/min/max, clipping, tool calls/failures and step duration |
| `budget-usage.png` | Logged actual observation charges, remaining budget, accepted/rejected queries, numerical calls and accounting coverage |

The error aggregation reuses the original published plotter. Backdoor
validity/edit, separate mediator node/edge F1 and strict correctness are absent
from the original owner and remain missing. Do not infer them from rewards.
Generated token lengths exclude tool feedback. Native step duration includes
cold compilation when it occurred.

Budget accounting reads completed-step native Parquet exports without running
their code. It checks initial budget, act/response pairing, histogram totals and
the scalar charge chain. Numerical output is never treated as causal feedback.
Explicitly rejected commands, including rejected post-terminal calls, cost zero.
Only complete, unambiguous logged answers enter budget means, with coverage
shown. Partial traces are observed lower bounds; missing/ambiguous traces are
not zeros. Decoded text exports are **not an independently authenticated owner
ledger**. No new training hook or tool dispatcher was introduced for monitoring.

`summary.json` distinguishes completed and fully saved steps, last-60-minute
updates, increment/elapsed time since the previous trusted capture, source/process
health, telemetry age, and data coverage. A first actual capture was rendered and
checked; monitoring unit tests cover budget spoofing, missing/truncated frames,
invalid charges, post-terminal rejection and CPU resets.

Schedule this one-shot command hourly in the user's existing reporting service
and display all five fresh figures even when no new updates occurred. In Codex,
the current-thread hourly heartbeat is configured separately through the app;
this repository does not embed account/thread IDs or create cron replacements.
The heartbeat has been enabled for the deployed branch. Do not automatically
restart/stop training, run extra model/GPU validation, change the runtime or
publish raw evidence from that monitor. Raw worlds, prompts, trajectories and
logs are deliberately excluded from this public source publication.

Run the offline unit tests with `python -m unittest -v test_monitor.py` in this
directory. All plots describe online training only; they are not controlled
comparisons across algorithms or tool modes.
