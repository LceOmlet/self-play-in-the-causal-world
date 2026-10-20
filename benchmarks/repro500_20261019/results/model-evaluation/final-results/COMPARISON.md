# Three-way results on the same 100 frozen cases

**The existing agent run is exploratory, not a fair controlled comparison with 9B.**
Agents retained 20-case histories and did not have the same enforced native generation/context budgets. Matching task prompts, tools, capsules and observation budgets is insufficient. These results do not fulfill the requested fair agent comparison. The two 9B runs share a runner and profile.

Native terminal-quality-v10 reward [0,1], higher is better; unfinished=0

| Family | n | Isolated agents | Original 9B | Checkpoint-2350 | Trained − base |
|---|---:|---:|---:|---:|---:|
| ate | 20 | 0.6578 | 0.0309 | 0.0533 | +0.0224 |
| backadj_minimal_sets | 20 | 0.9536 | 0.3000 | 0.3464 | +0.0464 |
| best_intervention | 20 | 0.8771 | 0.2269 | 0.3284 | +0.1015 |
| individual_counterfactual_probability | 20 | 0.6729 | 0.0000 | 0.0784 | +0.0784 |
| mediator_set | 20 | 0.3185 | 0.0436 | 0.0129 | -0.0307 |
| Overall | 100 | 0.6960 | 0.1203 | 0.1639 | +0.0436 |

Native diagnostics (errors are averaged over submitted answers only; exact counts use every assigned case):

| Family | Metric | Isolated agents | Original 9B | Checkpoint-2350 |
|---|---|---:|---:|---:|
| ate | TV error ↓ | 0.0305 (20/20 submitted) | 0.1192 (2/20 submitted) | 0.0641 (2/20 submitted) |
| backadj_minimal_sets | Valid adjustment sets ↑ | 13/20 | 1/20 | 1/20 |
| best_intervention | Optimal actions ↑ | 14/20 | 2/20 | 4/20 |
| individual_counterfactual_probability | Endpoint MAE ↓ | 0.0903 (20/20 submitted) | — (0/20 submitted) | 0.0817 (2/20 submitted) |
| mediator_set | Mediators and order both exact ↑ | 3/20 | 0/20 | 0/20 |

Five isolated agent conversations of 20 tasks; each 9B run starts fresh per task. Numerical state is fresh per task for all three. Internal reasoning-token budgets are not matched.

The five-family subset was fixed before solving. Both 9B runs began after the original 100 agent cases terminated. No terminal case was resampled.

Each pair directory contains exact diagnostics, per-case CSV/JSON and raw owner/model transcripts. Confidence intervals are descriptive paired bootstrap intervals with 10,000 draws and seed 20261019; each family contains only 20 cases.
