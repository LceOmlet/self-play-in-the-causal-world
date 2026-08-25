# Pinned upstream patches

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
