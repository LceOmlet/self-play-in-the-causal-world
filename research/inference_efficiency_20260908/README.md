Qwen3.5-9B inference configuration verification, 2026-09-08

GPU comparison completed: all eight workload/cache conditions improve, by
2.58–6.59x, with identical cached-token counts. Startup is 77.48 seconds eager
versus 273.58 seconds compiled. Both runs completed every fixed-length request;
the actual log records PIECEWISE and FULL graph capture. Probability equality is
not claimed: only 7/40 request pairs have identical full outputs, and common-
history sampled-logprob differences reach 0.47992. See
`docs/gpu-compilation-comparison-20260908.md` and `gpu-compilation-evidence.tar.gz`.

Actual official recovery has now loaded native update 5 (LoRA, optimizer, RNG,
scheduler) in `resume-compiled-02`. Its first three observed tasks match saved
sampler rows 496, 324 and 306. After filtering the first two groups, actual update
6 completed on a counterfactual group: finite nonzero gradient norm 0.1225586,
Token-TIS mean 1.0000582 and ESS fraction 0.9989741. Its 1,399.67-second step
includes three generated groups, so it is not an equal-work speed comparison
against the original one-group first step. Full actual update and trajectory
evidence is in `resumed-first-groups-evidence.tar.gz` (all nine members verified).
This is the active
verification run; the original run stays stopped with its checkpoint preserved.
The PNG/SVG compilation comparison was rendered and visually inspected locally.

Native update 6 is additionally preserved on the server and audited in
`resumed-update6-native-acceptance.json`: all 496 finite LoRA tensors changed
since update 5; 496 Adam states and the scheduler are at step 6, with 992 finite
moments. The data cursor advanced from six to nine generated batches, and the
next eight restored rows equal the uninterrupted sampler. This is published
checkpoint verification, not a bitwise observation of GPU memory restoration.

Latest actual state (02:53): update 5 published its native checkpoint, with six
generated batches consumed. The native file audit and eight-index sampler check
passed; the full checkpoint was copied and hashed before the identified original
training process tree was stopped. The sole GPU is now running the eager timing
baseline. `checkpoint5-and-pause-evidence.tar.gz` records this transition; large
native weights remain in the preserved server directory. Actual training resume
and the compiled speed comparison are still pending. Earlier preparation-only
statements below describe the preceding stages.

`compare_compilation.py` reads both completed GPU runs, requires equal inputs,
weights, sampling, source and library versions, and checks the complete engine
option diff. It compares the two measured repetitions for each workload and
cache phase. Selected-token logprobs are compared only on identical generated
histories before the first divergence, never on later coincidentally equal tokens.
It cannot stand in for the actual verl synchronization, TIS and resumed update.

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

`benchmark_compilation.py` now prepares and measures equal-work inference through
official `vllm.LLM.generate`. CPU preparation succeeded on the real tokenizer:
four reencoded rollout lengths are 6912, 15326, 26788 and 5818 tokens. Sliding
excerpts of the longest record provide identical 2048/24576-token inputs for
one/four requests, with distinct first cache blocks. These are timing fixtures,
not recovered original action-token counts or additional task solutions.

Both variants use the same bitwise-verified nonzero adapter, BF16, 32768 context,
four-sequence engine limit, prefix caching and 8192-token scheduler budget.
The only variant differences are eager mode and capture sizes [1,2,4]. Each
request generates exactly 256 tokens with official SamplingParams; ignoring EOS
is confined to this timing workload. Per-shape warmup precedes two cold-prefix /
reused-prefix pairs. Logs retain generated token IDs, selected-token logprobs,
actual cached-token counts, startup time and generation wall time. Wall throughput
includes prefill and host overhead; it is not raw decode throughput.

CPU checks bound both option dictionaries to the actual installed EngineArgs
signature and validated official CompilationConfig, LoRAConfig and SamplingParams;
CUDA remained uninitialized. Invoking the actual benchmark entry point while
training occupies the GPU correctly refused before creating an output directory
or a model. There is still **no GPU timing result**. The standalone benchmark
also does not exercise verl weight synchronization, training-side probability
recomputation, TIS or checkpoint loading; those remain actual-trainer checks.
Prepared inputs and CPU records are in `benchmark-preparation-evidence.tar.gz`.
The 02:10:47 local-time snapshot still has four logged updates and no checkpoint;
the trainer reports zero retained prompts after the first generation batch for
update 5 and continues sampling. Updates 1–4 each used one generation batch.
This observed filtering cost explains additional waiting in update 5, not the
already slow generation in the preceding updates. The training remains live.

`resume_runtime.py` prepares isolated eager/compiled recovery commands through
the official Hydra entry point and `validate_config`. Both preparations passed
in `resume-eager-02` and `resume-compiled-02`. The first preparation's diff check
also saw the derived trace experiment name and Ray environment entries omitted
by the previous inspection; that failed attempt is preserved. The comparison
now uses those environment entries from the actual submitted command.

The recovery launcher preserves the original 500-update/20-epoch verification
and its original wall-clock deadline. It uses native `resume_mode=resume_path`,
enables official vLLM statistics, and saves each update to make the first resumed
checkpoint available promptly. No smaller update cap is added. This is not the
final 10,000-update run, whose epoch/resume requirements remain outstanding.

Before releasing the GPU, `preserve_and_pause` requires the official completed
checkpoint tracker at step 5, finite model and optimizer tensors, optimizer and
scheduler counters at 5, and a saved first-epoch data position between 5 and 50.
It compares eight restored sampler indices with the uninterrupted stream, copies
and hashes the complete native checkpoint, revalidates the exact original PID,
then terminates only its owned process tree. The termination helper was tested
on a separately created parent/child pair: they exited, an unrelated sentinel
remained live, and the actual trainer remained live. Native checkpoint auditing,
the real pause, GPU benchmarks and actual resume have **not yet executed**.

The additional raw snapshot uniquely joins all 16 retained trajectories. Update
4 contains one exact mediator/order answer (quality 1) and three incorrect answers
(0.225, 0.25, 0.225). The filtered decision group consists of four identical
second-best actions, `state_2`; both the causal and population-observational
optimum are `state_3`. This is a concordant task, not a strong-reversal failure.
For update 3's ATE task, all four model TV errors exceed the zero-vector error
0.03262952; the population observational shortcut error is 0.10714027. Three
model answers beat that observational shortcut, but this alone does not beat the
constant baseline. These real examples distinguish task signal from demonstrated
learning. Raw files and prepared recovery commands are archived in
`runtime-switch-preparation.tar.gz`.
