# Original versus checkpoint-2350: training rollout generation

Same fixed 100 tasks and final on-demand-data-v1.1 environment. Actual training TRL generation/tokenization/parsing/tool-loop methods are reused unchanged. No extra 8192-token per-turn cap.

Training settings: 32768 context, 31497 total rollout completion budget (including tool feedback), temperature 1, top_p 1, top_k 0, repetition penalty 1, thinking disabled. One rollout per evaluation task. This replaces the earlier custom 8K/temperature-0 evaluation as the requested model comparison.

**Generation logic matches training; the later frozen task interface intentionally differs from training's old automatic-observation-feedback adapter. Reward is not accuracy.**

| Family | n | Original 9B | Checkpoint-2350 | Difference | Submitted original / trained |
|---|---:|---:|---:|---:|---:|
| ate | 20 | 0.1860 | 0.2073 | +0.0213 | 9 / 10 |
| backadj_minimal_sets | 20 | 0.5821 | 0.7036 | +0.1214 | 13 / 17 |
| best_intervention | 20 | 0.4843 | 0.6384 | +0.1541 | 14 / 17 |
| individual_counterfactual_probability | 20 | 0.2806 | 0.1571 | -0.1235 | 16 / 9 |
| mediator_set | 20 | 0.0715 | 0.1217 | +0.0502 | 10 / 14 |
| Overall | 100 | 0.3209 | 0.3656 | +0.0447 | 62 / 67 |

Trained / tie / original wins: 30 / 35 / 35. Paired bootstrap 95% interval: [-0.029701495257838845, 0.11947406274198277].

The old 8K results are retained separately and must not be pooled with this run. Native agents did not have matched hidden compute; their previous scores are not part of this new two-model test.

Raw token IDs, sampled-token logprobs, generated text, parsed completions, tool masks, tool calls/results, original acquired observations, capsule identities and source fingerprints are retained in the archive. No optimizer or training weights were changed.
