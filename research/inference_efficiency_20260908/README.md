Qwen3.5-9B inference configuration verification, 2026-09-08

The actual model is `/home/chen/models/Qwen/Qwen3.5-9B`, architecture
`Qwen3_5ForConditionalGeneration`: 32 text layers, 24 linear-attention and eight
full-attention layers. Both historical and current runs use that base model;
the old run's trained adapter and the current fresh adapter are different weights.
Generic Qwen2 upstream tests cannot establish this model's runtime compatibility.

`variants.json` contains the existing eager/adapter baseline and two official
configuration candidates: compiled inference with the adapter, then compiled
inference with upstream `model.lora.merge=true`. Both keep DAPO, sampling, task
data, rewards, budgets, batch size, training LoRA rank and optimizer unchanged.
Capture sizes [1,2,4] cover decode batches with at most four sequences and no
speculative decoding. These sizes do not cap response length or training steps.

`validate_candidates.py` resolves the actual submission's options through the
official Hydra entry point with `--cfg job --resolve`, runs official configuration
validation with CUDA hidden, and checks the complete resolved configuration diff.
It also verifies the pinned official sources and actual 9B model config. It does
not construct or train a model. Source/configuration acceptance is preliminary;
GPU memory, compiled execution, timing and probability agreement remain required.
The live production configuration is not changed by this work.

Official source inspection establishes that merge mode disables runtime LoRA
and streams merged weights through FSDP's own implementation. That implementation
keeps the merge context open while materializing tensors, clones plain tensors,
and restores the base from a backup rather than subtracting an approximate delta.
This avoids known aliasing and repeated unmerge-rounding failure modes by design;
the actual 9B synchronized tensors and nonzero-adapter behavior still need checking.
The upstream `test_engine_workers_lora_sync.py` copies dispatch logic into a test
helper; its success would not prove the production function executed, so it is
not used as our runtime acceptance evidence.

Next GPU verification must use the actual Qwen3.5-9B, a recorded nonzero adapter,
the same token inputs and equivalent output work. Compare compilation separately
from adapter merging, measure warmed generation and weight-sync costs, check
logprob semantics and official Token-TIS, and then observe an official update.
Merging `W + sBA` is algebraically valid but BF16 rounding need not match separate
`Wx + sB(Ax)` bit for bit. Correct probability tracking and finite useful weights
remain required; neither nominal compilation support nor a tiny-model test can
stand in for those checks. No competing GPU benchmark is launched while the
current training occupies the device; preserve its verified checkpoints first.

Executed result: all three candidates passed official configuration validation.
The complete resolved diff contains exactly zero, two and three changes,
respectively, all confined to the declared inference settings. The model config
SHA-256 is `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`.
GPU execution, probability checks and measured speed remain unverified.

Successful output location:
`/home/chen/runs/inference-efficiency-20260908/configuration-v2`.
The first inspection attempt put candidate overrides after Hydra's inspection
flags and failed in argument parsing before composing that candidate. Commit
`fbf1df3` fixes this inspection command; it does not alter the trainer. Both the
failed first attempt and successful second attempt are retained in
`configuration-evidence.tar.gz`, with hashes in `configuration-evidence.sha256.json`.

Actual generation evidence: the trainer's own Ray stdout confirms two updates,
while the driver log in the same snapshot only forwards the first metric record.
The reader now identifies the live DAPOTaskRunner child of the recorded main PID,
saves its stdout and source identity, and requires duplicate metrics to agree.
All 108 first-step metrics match the prior archive; all eight retained rollouts
match the original environment command sequences uniquely.

`analyze_generation_balance.py` applies an interval-union bound to the official
cumulative await timers. Step 2 spends at least 1163.31 seconds with only one
pending trajectory, regardless of trajectory start offsets. This is a request
concurrency bound, not GPU utilization or a measured optimization speedup.
The next GPU comparison must include one active trajectory and long contexts.
See `docs/generation-balance-20260908.md` for the derivation and task errors.
`generation-evidence.tar.gz` preserves the raw snapshot and timing sources;
`generation-evidence.sha256.json` records its hashes. The current `/metrics`
endpoint contains no vLLM token counters; the actual config disables log stats.

`plot_generation_evidence.py` reuses the earlier `original_plot_training_curves.py`
style and renders the two-update snapshot to PNG, SVG and PDF, with trajectory
CSV and source hashes in `figures/`. Rendering was done locally using the existing
`audit/plot_dependencies` Matplotlib runtime and Microsoft YaHei font; no plotting
packages were installed into the running training environment. The first local
invocation lacked that existing dependency path, then succeeded with it supplied
via `PYTHONPATH`. The figure is not a learning curve across matched tasks.
