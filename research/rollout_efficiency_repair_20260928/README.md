# TRL 32K candidate: rollout scheduling repair and scoped tolerances

This candidate uses **TRL 1.14.0+cpt.rollout1**, an explicitly versioned derivative
of TRL 1.14.0. It retains the official GRPO Trainer and its configured `dapo` loss;
it does not implement the full DAPO dynamic-filtering recipe. It is an isolated
validation candidate and has not been deployed for long training.

The [complete patch](../../patches/trl-1.14.0-rollout-residency.patch) adds two
default-off/default-inherited configuration fields: `vllm_rollout_residency` and
`vllm_enable_prefix_caching`. The unchanged official tool-generation body runs
inside one resource context. vLLM stays resident between tool calls, then sleeps
at the group's normal or exceptional exit before scoring/optimization. Nested
contexts release once. Official weight synchronization still resets the cache,
so the cache is never intentionally reused across a weight update. Loss,
advantage, masks, importance sampling, backward and optimizer math are not
reimplemented. The [source contract](source-contract.json) records 28 unchanged
Trainer method bodies and the upstream/patch SHA256 values.

## Numerical acceptance correction

The earlier whole-RL failure conclusion drawn from generic dtype-default
tensor thresholds and gradient relative-L2 differences is withdrawn. Those
measurements are diagnostics, not a project-specific official tolerance.

Actual checkpoint-2400 BF16 ordinary/chunked output heads were compared at a
mutually feasible total length of 2048 tokens, temperature 1, using two fixtures.
The frozen head's backward comparison uses `sum(logprobs)` on both sides, matching
the upstream test's reduction. All six component assertions passed:

| Quantity | TRL 1.14.0 upstream test | atol / rtol | Tool-fixture maximum absolute difference |
|---|---|---|---:|
| Token log-probability | `TestPatchChunkedLMHead.test_forward` | 0.005 / 0.005 | 0.00006706 |
| Token entropy | Same | 0.005 / 0.005 | 0.0004541 |
| Head gradient into BF16 hidden states | `TestChunkedLogProbFunction.test_backward_bfloat16` | 0.01 / 0.01 | 0.0009766 |

These are the original component-specific upstream thresholds, not tolerances
for every LoRA/Adam tensor or percentages of acceptable reward degradation.
The saved upstream test SHA256 is in [summary.json](summary.json).

A separate real off-policy fixture held old log-probabilities, advantages and
IS weights fixed: none of 1765 effective tokens changed high/low clipping
decisions, and the maximum microbatch loss difference was 5.821e-10. This
describes the measured objective effects; it does not define a new threshold.
The scheduling-only patch also preserved all observed objective quantities on
the same frozen input. The upstream vLLM 0.26 hybrid-prefix-cache first-divergence
reciprocal-top-5 protocol passed 60 continuations; its scope is narrower than a
full-distribution or whole-RL equivalence guarantee.

## Generation overhead repair

Matched work: checkpoint-2400, the same saved prefixes, four sequences, five
calls, exactly 128 new tokens per sequence/call (2560 per group), matching
sampling and precision. Timings include weight synchronization and sleep, with
one warmup and two steady repetitions per mode.

| Equally instrumented ablation | Seconds/group | Sync / reset-cache / sleep calls |
|---|---:|---|
| Original schedule | 45.4937 | 5 / 5 / 5 |
| Group residency | 36.5280 | 1 / 1 / 1 |
| Residency plus prefix caching | 24.3414 | 1 / 1 / 1 |

Without call instrumentation, the repaired candidate took **23.2266 s/group**
and the legacy performance reference took **23.2273 s/group**. This resolves the
extra generation cost in this small matched benchmark. The legacy path is a
speed reference, not the correctness oracle. Separate compute-only measurements
at the original 32768/31497 limits took 192.262 s per four-microbatch update versus
203.865 s on that performance reference. These isolated timings must not be
added together and presented as a measured end-to-end training speed.

Native integration with the original 32768/31497 caps completed four valid ATE
terminations, old-logprob/IS processing, four microbatches and one nonzero-gradient
FP32 Adam update. Actual completion lengths were 4195, 17503, 11362 and 4392;
this integration does not claim all four completions reached the 32K boundary.
Separate exact-32K scoring and optimizer resource checks were completed.

Still unverified: controlled equal-work complete-pipeline throughput, autonomous
coverage of all five task families, and long-run learning or plateau causality.
No production updates were added by these isolated checks.

## Reproduce the source and CPU lifecycle checks

Use the TRL tag archive and SHA256 in `source-contract.json`. In a disposable
directory, verify the archive hash, extract two copies, and apply the patch to
one copy (the commands below assume directories named `upstream` and `patched`):

```bash
git -C patched apply --check /path/to/repo/patches/trl-1.14.0-rollout-residency.patch
git -C patched apply /path/to/repo/patches/trl-1.14.0-rollout-residency.patch
python /path/to/repo/research/rollout_efficiency_repair_20260928/verify_patch.py \
  --upstream upstream --patched patched
```

The CPU check executes the patched lifecycle/decorator AST directly without
loading a model; it does not replace GPU integration or tolerance tests.
Build/install the patched tree as a normal wheel in an isolated environment.
There are no `PYTHONPATH` package overlays or per-instance Trainer replacements.

`official_train.py` preserves the tested project entry and exact dependency
version lock. Its task source is repository commit
`85cb577c73b17ccc6cde3bc6944db3ce2e80376f`, **not this branch's newer environment**.
To reproduce that task package, export `src/`, `pyproject.toml` and `README.md`
from that commit into a separate source directory, add this entry as
`src/cpt_world/official_train.py`, and install it normally. Do not apply legacy
runtime patches. Supply your model location explicitly; the saved entry's
default reflects the validation host. Launch with `python -I` and unset
`PYTHONPATH`. The entry uses the public PyTorch setting
`allow_bf16_reduced_precision_reduction=False` and preserves BF16 base weights
and FP32 LoRA/Adam.

The saved entry retains default-disabled scheduling extensions. The validated
integration explicitly selected both switches before constructing the Trainer:

```python
args = configuration(output_dir)
args.vllm_rollout_residency = True
args.vllm_enable_prefix_caching = True
trainer = build_trainer(model=model, args=args, train_dataset=data,
                        peft_config=lora_configuration())
```

This publication contains source, reproducible CPU checks and aggregate results.
Original task instances, trajectories, tensors, logs and evidence archives stay
in private storage. Their archive hash is retained in the summary for provenance.
