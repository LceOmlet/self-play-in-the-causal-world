# Per-case native subagents versus original and trained 9B

All three groups completed the same fixed 100 cases. Native agents submitted 100 final answers; original 9B submitted 20 and checkpoint-2350 submitted 24.

**Reward is not accuracy. Native hidden reasoning/generation compute and full system context are not matched to 9B.**
Each native case uses a fresh fork_turns=none conversation with a single-task capability. Task prompts, tool schemas, capsules, observation budgets, initial public token counts and owner source match both 9B runs. Native solver instructions additionally constrain transport and prohibit external computation. The old five-conversation agent run is excluded.

| Family | n | Native per-case agent | Original 9B | Checkpoint-2350 |
|---|---:|---:|---:|---:|
| ate | 20 | 0.6291 | 0.0309 | 0.0533 |
| backadj_minimal_sets | 20 | 0.9500 | 0.3000 | 0.3464 |
| best_intervention | 20 | 0.8757 | 0.2269 | 0.3284 |
| individual_counterfactual_probability | 20 | 0.6774 | 0.0000 | 0.0784 |
| mediator_set | 20 | 0.3938 | 0.0436 | 0.0129 |
| Overall | 100 | 0.7052 | 0.1203 | 0.1639 |

| Overall comparison | Mean difference | Paired bootstrap 95% interval | Left / tie / right wins |
|---|---:|---|---:|
| agent_minus_base | +0.5849 | [+0.5098, +0.6580] | 85 / 13 / 2 |
| agent_minus_checkpoint2350 | +0.5414 | [+0.4616, +0.6201] | 80 / 15 / 5 |
| checkpoint2350_minus_base | +0.0436 | [-0.0286, +0.1171] | 19 / 68 / 13 |

| Overall resource use | Native agent | Original 9B | Checkpoint-2350 |
|---|---:|---:|---:|
| submitted | 100 | 20 | 24 |
| mean_tool_calls | 20.29 | 32.77 | 35.23 |
| mean_observations | 44900.03 | 4446.65 | 2990.68 |
| mean_budget_fraction | 0.9656 | 0.1428 | 0.1302 |

Tool calls are owner act/numerics calls, excluding prompt and transport retries. Observation budgets vary by family; mean budget fraction is reported separately. Hidden model reasoning/generation tokens are not comparable.

All native cases ended with terminal_answer. Legal 9B failures remain zero reward; infrastructure failures were recovered without resampling and were not treated as performance zeros. The earlier model audit and archives are retained under ../model-evaluation/.

Native diagnostics, exact answers/errors, per-case rewards and paired comparisons are in JSON/CSV. Raw owner requests, responses, request caches, terminal records and source identities are archived. Bootstrap intervals use 10,000 paired case resamples with seed 20261019; they do not capture solver/run-to-run variability. Each family has only 20 cases.

The two 9B groups finished before the replacement native run finished. Scores were sealed from native solvers; the parent first unsealed this native run only after all 100 owner terminal files existed. Overall Codex instructions and hidden native generation/context limits cannot be audited as identical to the 9B model profile.
