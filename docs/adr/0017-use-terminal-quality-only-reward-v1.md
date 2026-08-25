---
status: accepted
---

# Use terminal-quality-only reward v1

The current RL milestone uses the continuous scalar definitions in
[`terminal-quality-reward-v1.md`](../terminal-quality-reward-v1.md) and keeps
experimental cost, query count, trajectory length, and binary success as
separate diagnostics. This preserves all scorer-owned quality information and
avoids silently changing the active-causal task into an accuracy-cost trade-off
before that objective is separately defined.

## Consequences

Raw parsing, truth, and diagnostic ownership remains in `task_scoring.py`; the
reward layer only scalarizes its structured output. Backdoor and mediator task
success remain exact structural equality even though their training rewards
provide partial credit, and model-caused unfinished trajectories receive zero.
