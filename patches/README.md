# Pinned upstream patches

## Current official verl DAPO path

Only `scripts/run_official_dapo.sh` is the current training entry. It uses the
official DAPO recipe with two explicit compatibility patches:

- `verl-tool-termination-v1.patch`: propagate real environment termination and
  retain the terminal action without adding tool feedback or another model turn.
- `verl-recipe-action-mask-v1.patch`: preserve the tool loop's existing action
  mask in `compute_kl_related_metrics`; use the official RayPPOTrainer guard to
  synthesize a mask only when none exists. The unpatched recipe incorrectly put
  2,976 tool tokens into a real audit update's 14,413-token denominator.

The original 427 upstream hashes, reviewed runtime file hashes and patch hashes
are recorded in `configs/verl/upstream_sources.json`. The verifier additionally
pins the recipe patch hash in code. Git treats patch files as binary to preserve
their original line endings and SHA-256 values. The recipe file is therefore
explicitly patched; its algorithms are still owned by the official sources.
See [integration status](../docs/official-dapo-integration-20260906.md) and
[research evidence](../research/rl_correctness_20260907/README.md).

## Historical TRL records — not the current training path

All TRL instructions below document earlier work only. Do not apply them to the
official verl environment, and do not use their checkpoints or locally derived
dynamic-sampling code as evidence for the current DAPO implementation.

### DAPO loss: verbatim upstream fix, no custom sampling

`trl-1.10.0-dapo-token-normalization.patch` now backports the exact
`GRPOTrainer.compute_liger_loss` method from the official TRL 1.12.0 wheel
into the installed 1.10.0 owner after the two earlier performance patches.
It replaces the previous locally derived normalization patch. The input
trainer SHA is `1e34e9f6b0d92577837c3ee7d7b25b0aa841d8d7a6408ba67e976aba3e84a7af`;
the result is `38189ac6ffffc027966c8c7cf86d568c0bcffa5e091ad727463e009a38c106f5`.
The complete version/hash manifest is `rl-owner-manifest.json`.

The custom dynamic-sampling patch has been withdrawn. Do not apply archived
copies: they are not part of the current training path. Native TRL's DAPO
loss alone does not implement the full DAPO dynamic-sampling recipe.
See `docs/dapo-normalization-fix-20260906.md` for scope and validation limits.

## TRL 1.10.0 fused old-policy log probabilities

`trl-1.10.0-qwen35-fused-old-logps.patch` changes the owning TRL
`GRPOTrainer` implementation instead of adding a second trainer to this project.
It uses Liger's fused linear scaled-cross-entropy primitive to obtain only the
selected completion-token log probabilities, avoiding a full
`sequence_length x vocabulary_size` FP32 logits tensor.

The fast path is deliberately limited to the configuration validated here:

- TRL 1.10.0 and Liger Kernel 0.8.2;
- Qwen3.5 text models with a bias-free linear output head;
- `use_liger_kernel=True`, temperature 1.0;
- text-only input, with entropy and MoE auxiliary loss disabled.

All other configurations retain TRL's original path. The expected SHA-256 of
the unpatched `trl/trainer/grpo_trainer.py` is
`73034b5806e400a93d5977d489dd990a2edadace2c08676d815ff145a9386d4f`.

Apply from the environment's `site-packages` directory:

```bash
sha256sum trl/trainer/grpo_trainer.py
cp -p trl/trainer/grpo_trainer.py trl/trainer/grpo_trainer.py.pre-fused-logps
patch --forward -p1 < /path/to/self-play-in-the-causal-world/patches/trl-1.10.0-qwen35-fused-old-logps.patch
python -m py_compile trl/trainer/grpo_trainer.py
```

The patched file SHA-256 is
`7b18a74f679276e26b7ce184ce1cf10d9a15dfe30074816953825a74c20b8ca6`.

Validation on Qwen3.5-9B BF16 + LoRA found a maximum absolute difference of
`1.24238431e-6` against a full FP32 selected-token log-softmax computed from
the same hidden states. Against TRL's separate generic model-forward path, the
maximum difference was `0.00290772691`, below BF16 unit roundoff; nearly all of
that difference came from the two BF16 transformer forward paths.

The original 16K smoke configuration (`max_model_length=16384`,
`max_completion_length=14336`) completed one full GRPO optimizer step in
732.2 seconds. GPU telemetry recorded 41,310 MiB peak memory used and 7,194 MiB
minimum free memory on the 48 GiB RTX 4090.

## TRL 1.10.0 colocated rollout inference acceleration

`trl-1.10.0-rollout-inference-acceleration.patch` extends the owning TRL vLLM
integration with default-inert seams for three staged experiments. It does not
replace `GRPOTrainer` or duplicate its tool loop.

| Stage | Enabled behavior |
| --- | --- |
| P0 | Keep weights and KV cache resident across all tool turns in one rollout; sleep once before training. |
| P0+APC | P0 plus vLLM automatic prefix caching for the growing multi-turn prefix. |
| P0+APC+MTP-1 | The prior stage plus the checkpoint's native one-token Qwen3.5 MTP draft head. |

The patch adds `vllm_rollout_residency`, `vllm_enable_prefix_caching`,
`vllm_speculative_config`, and `vllm_sleep_level` to `GRPOConfig`. Their
defaults preserve the installed TRL/vLLM behavior. MTP runs use sleep level 1:
sleep level 2 discards the draft weights, while TRL weight synchronization only
owns the Transformers training-model weights and therefore cannot reconstruct
the checkpoint-only MTP tensors.

Apply this patch after the fused-old-log-probability patch. The expected input
SHA-256 values are:

- `trl/generation/vllm_generation.py`:
  `d0843aa5b97a32dcbfe61dbcebd285c081e715fbf7d939ea1ae564308cac8bbb`;
- `trl/trainer/grpo_config.py`:
  `e69ad0011e7ffa3467c89e87cdafb9a4fb1f9b69b41c301e308ba6e20232c319`;
- `trl/trainer/grpo_trainer.py`:
  `7b18a74f679276e26b7ce184ce1cf10d9a15dfe30074816953825a74c20b8ca6`.

Apply from the environment's `site-packages` directory:

```bash
patch --dry-run --forward -p1 < /path/to/self-play-in-the-causal-world/patches/trl-1.10.0-rollout-inference-acceleration.patch
patch --forward -p1 < /path/to/self-play-in-the-causal-world/patches/trl-1.10.0-rollout-inference-acceleration.patch
python -m py_compile trl/generation/vllm_generation.py trl/trainer/grpo_config.py trl/trainer/grpo_trainer.py
```

The corresponding patched SHA-256 values are
`7a91eb42c7c9614c23a20af0e7f8d6a37d1de5442cee7086cef3551a2b1ad2d4`,
`6c78f3a8dae872e8e1d5c9746c24abf11f1108eaa7cd864c37f6b79fa9b78bb8`,
and `1e34e9f6b0d92577837c3ee7d7b25b0aa841d8d7a6408ba67e976aba3e84a7af`
in the same order. Run `scripts/validate_trl_rollout_acceleration.py` with the
patched package on `PYTHONPATH` before any GPU experiment.

The experiment gate is deliberately stricter than successful startup. Every
stage starts from the same checkpoint and five-task seed stream. P0 requires
token IDs, reward, and tool trace parity. APC and MTP-1 preserve the greedy-token,
selected-token-log-probability, importance-sampling, reward, and tool contracts;
exact stochastic transcript equality is not required. Selected-token logprob
noise is calibrated against a second unoptimized baseline process because a
fixed `2^-7` absolute bound is tighter than vLLM's own repeat-run BF16 variation.
A stage is deployable only if it passes its semantic gate and improves
end-to-end effective generation throughput by at least 5%. If two passing
stages differ by less than 5%, prefer the smaller, lower-memory change.

The 2026-08-26 checkpoint-50 A/B selected P0+APC+MTP-1:

| Stage | Effective tok/s | Peak GPU MiB | Peak temp | Result |
| --- | ---: | ---: | ---: | --- |
| Installed baseline | 48.64 | 42,940 | 85 C | control |
| P0+APC, warm | 55.79 | 42,804 | 84 C | passes |
| P0+APC+MTP-1, warm | 62.99 | 46,942 | 83 C | selected |

MTP-1 improved effective throughput by 12.9% over warm APC and 29.5% over the
installed baseline. All five selected-stage optimizer steps completed without
truncation or OOM. Four fixed prompts produced exactly the same 204 greedy token
IDs in baseline, APC, and MTP-1. The MTP-1 selected-token logprob differences
(mean 0.00136, p99 0.0282, max 0.0495) were inside the repeat-baseline numerical
envelope (mean 0.00241, p99 0.0577, max 0.0590). P0 alone was not deployed: its
changed vLLM sleep/request lifecycle also changed unseeded sampling RNG
consumption, so it failed the exact stochastic-transcript gate even though its
owner-layer lifecycle contract passed.
