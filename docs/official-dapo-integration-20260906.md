# Official verl DAPO integration — 2026-09-06

This integration follows the user's explicit requirement to use the complete
official DAPO implementation. The project does not implement a trainer, policy
loss, advantage estimator, dynamic sampler, optimizer, or substitute rollout.
The long training run remains stopped. Historical RL checkpoints are
disqualified. On 2026-09-07, a real one-update audit exposed a further upstream
recipe defect: `compute_kl_related_metrics` replaced the tool loop's action mask
with an attention mask. The audit failed despite exiting zero; its checkpoint
is disqualified and the incomplete component-level acceptance was withdrawn.
The current recipe applies the pinned official RayPPOTrainer's existing guard
to preserve a supplied action mask. A fresh run, `run-02-mask-v1-audit-only`,
is validating this repair from the original base. Its execution acceptance must
pass before claiming that this actual update is correct. Numerical equality
between rollout and actor is not a correctness requirement; the earlier blanket
rejection of numerical mismatch was too strong.

## Ownership and source provenance

The entry point is `scripts/run_official_dapo.sh`, which executes
`python -m dapo.main_dapo`. Hydra applies the project's configuration group
over the official `dapo_trainer` primary configuration.

| Component | Official source and explicit compatibility changes |
| --- | --- |
| Training loop and dynamic group filtering | `verl-recipe/dapo/dapo_ray_trainer.py`; reviewed action-mask preservation guard, unchanged algorithm formulas |
| GRPO advantages and asymmetric clipped token-mean loss | `verl/verl/trainer/ppo/core_algos.py` |
| Actor and microbatch gradient accumulation | `verl/verl/workers/engine_workers.py`, `verl/verl/workers/engine/fsdp/transformer_impl.py` |
| Generation and response masks | Official `tool_agent_loop.py`, with the separately reviewed terminal-transition patch described below |
| Soft overlong reward shaping | `verl/verl/experimental/reward_loop/reward_manager/dapo.py` |
| Qwen3.5 fused output and chunked PyTorch cross-entropy | `verl/verl/models/transformers/qwen3_5.py`, `verl/verl/utils/experimental/torch_functional.py` |

Pinned repositories:

- [verl commit 23af6a7](https://github.com/verl-project/verl/tree/23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3)
- [verl-recipe commit ee3aef1](https://github.com/verl-project/verl-recipe/tree/ee3aef1690d6bb5e6448052c4842e8efb7a3f76c/dapo)

`configs/verl/upstream_sources.json` records the commits and SHA-256 of all
427 Python/configuration files under the loaded official packages. The
launcher checks their contents and the actual imported module paths before
execution. The original 427 hashes are retained; `reviewed_runtime_patches`
separately records the terminal patch and the action-mask patch, their SHA-256
values, and the before/after hashes of the two tool runtime files and one recipe
file. The recipe exception is additionally bound to the exact reviewed patch
hash in the verifier; an arbitrary replacement trainer is rejected. The immutable
upstream exports remain available. The launcher uses
`/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1` and
`/home/chen/vendor/dapo-official-20260906/verl-recipe-mask-v1`.
The independent environment has no TRL installation. Hashes prove
provenance and detect changes; they do not prove mathematical correctness.

Project-owned code is limited to:

- `scripts/prepare_verl_cpt_data.py`: materialize the balanced task generator
  with the new population-identification contract as standard parquet,
  preserving certified counterfactual truth.
- `src/cpt_world/verl_environment.py`: connect official tool calls to the
  existing `CPTWorldEnvironment`, transfer its terminal metadata to the
  official DAPO reward manager, and return the environment's raw quality.
- Configuration, launcher, provenance checks and diagnostic output.

The metadata adapter subclasses `DAPORewardManager` only to copy the official
tool-loop metadata into its `extra_info` argument. It immediately calls the
unchanged parent `run_single`; it does not calculate the overlong penalty or
reimplement reward shaping. Tool calls return zero step reward, so the terminal
quality is counted once. Tool state belongs to the official trajectory's
`AgentData`, survives per-call create/release, and cannot cross task tapes.
The historical module name `cpt_world.trl_environment` denotes the existing
environment owner; importing it does not import TRL.

## Exact training objective and its limits

The environment still reports raw terminal quality in `[0, 1]`. That fact alone
does **not** mean the optimizer is an unbiased estimator of equal-family mean
raw quality. The standard DAPO configuration performs all of these operations:

1. Add the official soft overlong penalty to raw quality.
2. Keep prompt groups whose raw-quality standard deviation is nonzero, and
   generate more groups until the official batch requirement is met or its
   generation limit raises an error.
3. Center and divide shaped rewards by their within-group standard deviation.
4. Apply asymmetric ratio clipping (0.2 below, 0.28 above), with token-mean
   aggregation across the full optimizer batch. There is no KL or entropy
   bonus in this profile.
5. Apply the official `decoupled_token_is` configuration: token weights
   `min(2, exp(old_log_prob - rollout_log_prob))`, no batch normalization,
   no rejection sampling, and no bypass mode. The official trainer
   computes and passes these weights to its existing clipped policy loss.
   This is DAPO with upstream Token-TIS, an explicit extension to the original
   paper's recipe. Token-wise correction and truncation retain bias; this is
   not a claim of an exactly unbiased full-trajectory policy gradient.

These are algorithm choices required by the selected standard, not properties
that disappear when the code comes from an official repository. In particular:

- With binary success probability `p` and `G` independent attempts, the group
  survives filtering with probability `1 - p**G - (1-p)**G`. Consequently,
  uniform input family mass need not yield uniform accepted update mass.
  For continuous quality the exact retention probability depends on its full
  distribution; the binary formula must not be applied indiscriminately.
- For `R' = a*R+b`, `a>0`, normalized advantages are approximately unchanged
  (exactly unchanged without the numerical epsilon). Absolute quality-gap
  magnitude is therefore not preserved by the standardized advantage.
- The official multi-turn reward manager measures response length from the
  response attention mask. This includes tool feedback. The policy loss mask
  excludes tool feedback. Thus a large environment observation can contribute
  to a length penalty without being a policy action token. No local code
  changes this upstream definition.
- Dynamic filtering saves updates on zero-advantage groups, but the discarded
  groups still cost generation and environment execution. Fewer optimizer steps
  do not by themselves prove lower wall time or fewer environment calls.
- The pinned upstream loop originally ignored environment termination. The
  reviewed patch adds a strict Boolean `ToolResponse.terminate`, propagates it
  through `_call_tool`, and returns `AgentState.TERMINATED` before appending tool
  feedback when a tool completes the episode. The adapter takes this flag from
  the real environment's `episode.completed`. The terminating action's token
  IDs, loss mask and log probabilities remain intact; no confirmation or later
  assistant turn is added. No substitute generation loop or DAPO loss exists.
- The real one-update audit reported a pre-update rollout/actor KL estimate of
  0.325954, a nonnegative k3 estimate of 0.277033, and mean/max absolute sampled
  token probability differences of 0.051545/1.0 under the response mask. These
  diagnostics compare rollout probabilities with recomputed old actor
  probabilities before the optimizer step. The cause is not established.
  Different numerical backends can evaluate identical BF16 weights differently;
  the discrepancy alone does not prove a broken implementation or explain poor
  learning. `actor/ppo_kl=0` compares actor evaluations with each other and does
  not measure rollout agreement. The earlier profile enabled metrics only;
  the repaired profile enables official Token-TIS. Token alignment, actual
  sampling probabilities, correction-weight tails and effective sample size
  remain the relevant acceptance evidence, rather than bitwise equality.
- The later `run-01-audit-only` captured the mask overwrite directly: 11,437
  genuine action tokens became a 14,413-token loss denominator, including 2,976
  tool tokens. The largest absolute log-probability discrepancy was 33.6095 on
  tool tokens, whose rollout log-probabilities are zero placeholders; on actual
  actions the maximum was 0.557921. This establishes the cause of the later
  run's contaminated metric and loss, not every earlier numerical discrepancy.
  The new audit checks mask identity along the entire real trajectory-to-loss
  path, in addition to comparing formulas with their inputs.

The [official rollout-correction documentation](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)
explicitly covers identical weights with different precision/backends and provides
the Token-TIS preset and DAPO integration. Our independent fixed-token short-input
replay found exact equality between native Hugging Face and the official verl
patched model, with rollout/actor mean probability differences of 0.006466 and
0.001928. This comparison excludes full FSDP/LoRA and long real tool trajectories.
A separate 4096-token replay isolated blanket BF16 casting of RoPE buffers:
the last 256 token probabilities changed by 0.002763 on average and restoring
the original buffers restored exact results. This proves a numerical source,
not the cause of the earlier 0.051545 discrepancy or failed learning. No RoPE
source patch was deployed on the strength of this limited evidence.

Repair evidence is under
`/home/chen/runs/official-dapo-integration-20260906/repair-20260906`:
`repair-upstream-tests.log` (86 passed), `repair-project-tests.log` (13 passed),
`probability-comparison.json`, `rope-cast-proof.json`, and
`launch-preview/official-source-verification.json`. The patch also passed a
reverse-apply check against the exact reviewed exported file bytes.

The existing ADR 0020 (`scale_rewards="none"`) is incompatible with the user's
subsequent requirement for standard DAPO. Its unstandardized-advantage decision
does not govern this entry point. The v10 environment reward owner is retained;
DAPO's shaped training reward must be reported separately from raw quality.
The uniform-mixture document describes the source data distribution here, not
an assertion that dynamic filtering preserves the accepted family distribution.

The original [DAPO paper](https://arxiv.org/abs/2503.14476) establishes empirical
results on mathematical reasoning. It does not establish optimality for this
multi-turn causal environment. Correct implementation, suitability of the
surrogate objective, and task identifiability are separate requirements.

## Server configuration and reproducible invocation

Server project:
`/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831`.

Independent Python:
`/home/chen/.venvs/dolens-dapo-official/bin/python`.

Current reviewed runtime source roots:
`/home/chen/vendor/dapo-official-20260906/{verl-tool-termination-v1,verl-recipe-mask-v1}`.

The selected stack is Python 3.12, torch 2.11.0+cu130, vLLM 0.24.0,
transformers 5.9.0, PEFT 0.20.0, Ray 2.58.0 and the pinned official source
commits above. Dependencies were installed with `uv` in the independent venv;
the old training venv and its patches are not used by this entry point.

The profile uses one GPU, official FSDP2, BF16 LoRA, CPU optimizer offload,
SDPA without remove-padding, and the official chunked PyTorch fused output. The actor
base stays on GPU. vLLM loads the same original safetensors directly, avoiding
the initial full-base synchronization. These standard settings avoid the
64-GiB host-memory failure observed when both stacks held additional base
weight copies. vLLM retains its official LoRA sleep and adapter-update logic.
It defaults to one optimizer update so invoking it cannot silently resume a
10,000-step run. The default context limit is 32,768 with a 30,720-token
response budget and a 4,096-token overlong buffer.

The previous Triton configuration failed deterministically at the first real
actor log-probability calculation. `LinearCrossEntropy.forward` flattens its
outputs and Qwen3.5 returns them directly, while FSDP's non-remove-padding
`prepare_model_outputs` indexes `output.log_probs.shape[1]`. A one-dimensional
tensor cannot satisfy that contract. The full traceback is preserved in
`audit-base-environment-v1-03/train.log`; all four real trajectories completed
with different terminal qualities before this failure. No update was accepted.
The official `impl_backend: torch` preserves `(batch, sequence)` dimensions and
processes 512 tokens per output-head chunk. Selecting it changes only the
official numerical backend, not DAPO's objective or implementation ownership.

```bash
cd /home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831
export CPT_WORLD_TRAIN_DATA=/absolute/path/train.parquet
export CPT_WORLD_VAL_DATA=/absolute/path/val.parquet
export CPT_WORLD_RUN_DIR=/absolute/path/new-isolated-run
bash scripts/run_official_dapo.sh
```

The adapter is newly initialized on the original base. Historical initialization
or resume environment variables cause the launcher to exit. Eighteen old
checkpoints were moved out of their original paths into directories named
`checkpoints.disqualified-20260906`; diagnostic logs remain in place. The full
mapping is in `historical-checkpoint-quarantine.json` under the audit root.
The earlier audit that loaded checkpoint-850 was stopped and is not accepted
as current execution evidence.

All new outputs go to the new run directory. Data preparation produces a
finite dataset, not an infinite online task stream. A long run requires
deliberate dataset sizing and epoch configuration. Old parquet data fail the
environment-version preflight before model loading; the reward owner also
rejects a stale version even if no tool call occurred.

## Evidence so far

The bullets below describe the earlier 2026-09-06 integration, not current
end-to-end acceptance. The [2026-09-07 research archive](../research/rl_correctness_20260907/README.md)
preserves the later mask failure, exact patch, three direct recipe-mask tests,
seven provenance tests, fresh-run evidence and remaining proof obligations.
The current audited train fixture SHA-256 is
`4d093268da03fbfe665494ef7c2ecad1efa0d2d83137b9fd0c36e9b3537ae82f`;
its 20 tasks were recomputed with the integrated environment before launch.
The inspected 250-row diagnostic validation fixture SHA-256 is
`362b9b15461e4317ab7902258995c303821aaa0b0d2096e08aa721a10ad8b944`.
Neither is an untouched capability test, and the one-step audit does not run a
post-update capability evaluation.

Audit root: `/home/chen/runs/official-dapo-integration-20260906`.

- Ninety-two environment, rendering, sampler, adapter and runtime regression
  tests passed (plus five parameterized subtests). This includes six
  population-identification tests and seven official DAPO tool-adapter tests.
- Ten selected unchanged upstream GRPO/DAPO algorithm and reward tests passed.
- The unchanged upstream GPU vocabulary-padding kernel regression passed.
- Thirty-nine additional unchanged upstream aggregation, FSDP accumulation,
  temperature-scaling and LoRA-checkpoint tests passed.
- The older upstream `TestLinearCrossEntropy.verify_correctness` helper failed
  before the tested kernel: its reference path passes two-dimensional labels
  to two-dimensional logits, causing `torch.gather` to reject their ranks.
  `upstream-kernel-tests.log` preserves the traceback. This failed helper is
  not changed to make the test pass. A separate audit invoked the unchanged
  upstream PyTorch reference and the unchanged official Triton kernel directly
  with the upstream input generator. Log-probability, entropy, hidden-gradient
  and output-head-gradient checks passed the original upstream BF16 tolerances;
  see `numerical-owner-acceptance.log` and `verify_numerical_owners.py`.
- The official asymmetric token-mean loss and its derivatives were compared
  with an analytic reference for varying lengths and positive/negative
  advantages, with 1, 2 and 4 rows per microbatch. The loss was 0.0586 in all
  partitions; the maximum gradient error was 1.39e-17. This verifier is outside
  the training entry point and performs no optimizer updates.
- Official configuration resolution and module provenance checks passed.
- Six unchanged upstream chunked-output tests passed. GPU FP32 and BF16
  comparisons against direct PyTorch autograd also passed, including a
  514-token case crossing the 512-token chunk boundary. Maximum BF16 errors
  were 0 for log probabilities, 0.001954 for hidden gradients and 0.007813
  for head gradients, within the recorded tolerances. The separate verifier
  is `verify_chunked_output.py`; numerical evidence is
  `chunked-output-numerics.json`. This is the backend used by the current run.
- The isolated real-model execution audit has finished. The short-context audit
  produced unfinished zero-quality groups; the actual official loop skipped
  those groups and resampled. It was stopped before installing the verified
  environment changes. The current audit uses the full profile and a separate
  execution fixture requesting at most two experiments before the terminal
  answer. This is an instruction, not an enforced interaction limit: one
  sampled trajectory exceeded forty experiments. The official token limit
  remains the hard bound. The real model generates all commands and answers
  and the existing environment computes every reward.
  The fixture is not formal training data or capability-evaluation evidence.
  After the deterministic Triton shape failure, audit
  `audit-base-environment-v1-04` repeated the same full-context fixture using
  the verified official chunked PyTorch backend, original base and fresh LoRA.
  It exited 0 after one optimizer update, checkpoint saving and rollout weight
  synchronization. The gradient norm was 1.484375 and optimizer step was 1.
  All 248 LoRA B matrices became nonzero from their default zero initialization;
  all 496 saved adapter tensors were finite. This proves an actual update
  occurred, not that the complete target or rollout probabilities are correct.
  The audit checkpoint was moved to `checkpoints.audit-only-20260906` and marked
  disqualified for training initialization and capability claims. No GPU
  training process remained after the audit. See `step-1-metrics.json` and
  `checkpoint-verification.json` in that run directory.

The environment contract is `full-nonanchor-identification-v1`.
The earlier 2026-09-06 20-row train fixture had four rows per family, SHA-256
`9fdd2b1337e4a5473fc3c414ad51bdc7c9343528d6d1ded93700a7be33e2039b`.
The regenerated five-row validation set has one row per family, SHA-256
`aa2133c0b3bae485356e86a045a3ebf3c67e01b4539ab565b2501ae5bad5f47c`.
These are integration fixtures, not a statistical evaluation of learning.
