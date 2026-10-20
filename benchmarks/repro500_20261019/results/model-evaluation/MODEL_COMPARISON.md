# Original versus trained 9B: same 100 frozen tasks

Both 100-case runs have valid terminal records. Final answers were submitted in 20 original-model cases and 24 trained-model cases. Legal unfinished cases remain zero reward; transport/executor failures are excluded and were recovered without resampling returned answers.

| Family | n | Original 9B reward | Checkpoint-2350 reward | Difference | Final answers: original / trained |
|---|---:|---:|---:|---:|---:|
| ATE | 20 | 0.0309 | 0.0533 | +0.0224 | 2/20 / 2/20 |
| Backdoor adjustment sets | 20 | 0.3000 | 0.3464 | +0.0464 | 7/20 / 8/20 |
| Best intervention | 20 | 0.2269 | 0.3284 | +0.1015 | 7/20 / 10/20 |
| Individual counterfactual | 20 | 0.0000 | 0.0784 | +0.0784 | 0/20 / 2/20 |
| Mediator set | 20 | 0.0436 | 0.0129 | -0.0307 | 4/20 / 2/20 |
| Overall | 100 | 0.1203 | 0.1639 | +0.0436 | 20/100 / 24/100 |

Paired cases: trained wins 19, ties 68, original wins 13. Mean reward difference: +0.0436; descriptive paired-bootstrap 95% interval [-0.0286, +0.1171]. The interval includes zero; this run does not establish a reliable training improvement.

Reward is the unchanged native terminal-quality-v10 value in [0,1], not accuracy. Both models use the same runner, task owner, prompts, tools, observation budgets, sandbox, thinking=false, temperature=0, seed=20261019, 32768-token context and 8192-token per-turn generation cap.

Download/provenance verification: `download-verification.json`; exact first replies, all returned completion prefixes and replayed tool feedback were checked from archived bytes for all 200 cases. The 64 checkpoint transport timeouts were continued with 3600-second waits.

**The old five-agent results inside `final-results/` are exploratory only.** They retained 20-case histories and did not match native generation/context limits. The replacement native subagent group uses a fresh conversation per case, is still running, and is not included or combined with this archived group. Its hidden native reasoning/generation budget remains unobservable and must be disclosed in the eventual comparison.
