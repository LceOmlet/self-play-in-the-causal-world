Official checkpoint continuity audit, 2026-09-08

`audit_verified_checkpoint.py` compares the real, previously verified run-02
checkpoint with its actual post-Adam observation: 496 parameter tensors and 992
moment tensors match bit for bit. The 760 frozen optimizer entries are empty
dictionaries. This is not an audit of disqualified checkpoints or a claim that
the current live run has already saved a checkpoint.

The script calls upstream FSDP shard-loading/merging methods and upstream
`save_lora_adapter`; it does not reproduce them. The resulting 496 adapter
tensors also match exactly. The exported inference fixture remains at
`/home/chen/runs/checkpoint-continuity-20260908/saved-state-v2/lora_adapter`.
Weights are retained on the server; small metadata and their hashes are saved
here. Native training restoration still requires the original checkpoint,
optimizer, training configuration and data state.

`probe_data_resume.py` executes the actual imported upstream `_load_checkpoint`
and official sampler factory on row indices of the current 500-row dataset.
Only the GPU checkpoint-load RPC is mocked. It deterministically reproduces
wrong subsequent row order at the original inferred epoch boundary, including
a 500-update / 600-generated-batch state that dynamic filtering can produce.
These are CPU data-control fixtures, not model checkpoints or shadow trainers.

`prepare_data_resume_candidate.py` copies the live upstream source to the
isolated `verl-data-resume-candidate-v1`, changing only data-state restoration.
All three row-stream cases pass there. The script verifies the live tree is
unchanged and records a one-file patch. The candidate is not admitted to the
production provenance manifest, and no training process imports it.

Remaining work: preserve DAPO's real generated-batch / epoch state, verify an
actual 9B native checkpoint load and resumed update, then test compilation and
merged inference with matched inputs. Saved training RNG state does not prove
that separate vLLM sampling RNG continues bit for bit. The final 10,000-update
epoch requirement must count generated and filtered batches correctly.

`evidence.tar.gz` and `evidence.sha256.json` preserve successes, the first audit
helper's metadata-comparison failure, the original bug probes and candidate
results. The first helper confused tuple/list metadata representation; correcting
it did not change any checkpoint. CUDA remained uninitialized in all CPU probes.
The detailed derivation is in `docs/checkpoint-continuity-20260908.md`.
