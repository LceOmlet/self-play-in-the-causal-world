# Terminal-Quality Reward v9

Superseded by [`terminal-quality-reward-v10`](terminal-quality-reward-v10.md).

`terminal-quality-v9` keeps the v8 rewards for categorical total effect,
individual counterfactual ROI, experimental decision, and mediator structure.
It replaces only the backdoor-adjustment scorer and reward.

The model returns one complete `adjustment_set`. The scorer considers every
graphically valid adjustment set for the requested treatment and outcome. A
valid set excludes treatment descendants and d-separates treatment and outcome
in the backdoor graph. It finds the minimum number of variable additions and
removals needed to turn the submitted set into any valid set.

Backdoor terminal quality is one divided by one plus this exact edit distance.
Consequently, a valid set receives `1`, a one-variable error receives `1/2`, a
two-variable error receives `1/3`, and so on. A harmless nonminimal set receives
`1` whenever the complete submitted set itself remains graphically valid.

The scorer uses only the DAG. It does not enumerate CPT states, standardize
observational distributions, weight variables by mechanism strength, compare
families of minimal sets, or compute set F1. The nearest-set search is exact on
the complete generated support of at most 16 nodes.

All five owner-produced terminal qualities enter GRPO unchanged.
