# Agent versus checkpoint2350, same 100 tasks

Native terminal-quality-v10 mean reward (higher is better, 0–1). Unfinished cases remain zero.

| Family | n | Isolated agent | 9B | Difference | Agent / tie / 9B wins |
|---|---:|---:|---:|---:|---:|
| ate | 20 | 0.6578 | 0.0533 | +0.6045 | 19 / 0 / 1 |
| backadj_minimal_sets | 20 | 0.9536 | 0.3464 | +0.6071 | 17 / 3 / 0 |
| best_intervention | 20 | 0.8771 | 0.3284 | +0.5487 | 13 / 6 / 1 |
| individual_counterfactual_probability | 20 | 0.6729 | 0.0784 | +0.5945 | 19 / 0 / 1 |
| mediator_set | 20 | 0.3185 | 0.0129 | +0.3056 | 11 / 9 / 0 |
| Overall | 100 | 0.6960 | 0.1639 | +0.5321 | 79 / 18 / 3 |

Five isolated agent conversations of 20 cases; Qwen fresh conversation per case. Fresh numerical sessions for both; reasoning-token budgets are not matched.

All paired prompts, tool schemas and capsule hashes match. All 100 agent cases terminated before Qwen actions began.

Per-case native diagnostics, submissions and resource use are in paired-results.json/csv. Raw owner events and model responses are in transcripts.tar.gz.

Bootstrap intervals in comparison.json are descriptive paired case-resampling intervals (10,000 draws, fixed seed); each family has only 20 cases.
