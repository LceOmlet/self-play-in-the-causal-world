# Terminal-Quality Reward v10

`terminal-quality-v10` keeps the v9 rewards for categorical total effect,
individual counterfactual ROI, experimental decision, and mediator structure.
It replaces only the scalarization of the exact backdoor-adjustment edit
distance.

The model returns one complete `adjustment_set`. The scorer considers every
graphically valid adjustment set for the requested treatment and outcome. A
valid set excludes treatment descendants and d-separates treatment and outcome
in the backdoor graph. It finds the minimum number of variable additions and
removals needed to turn the submitted set into any valid set.

Let `N_max` be the maximum graph size configured for the complete task
distribution. Treatment and outcome cannot occur in an adjustment set, so
`N_max - 2` is a distribution-wide upper bound on the edit distance. Backdoor
terminal quality is

```text
1 - nearest-valid-set edit distance / (N_max - 2).
```

Every additional mistaken variable choice therefore subtracts exactly the
same amount in every task drawn from that configured distribution. The scale
does not depend on the current world's size, number of valid sets, covering
radius, or model outputs. With the default 8--16-node grammar, `N_max` is 16
and every edit subtracts `1/14`. A valid set receives `1`; the largest possible
distance over the configured support receives `0`.

The runtime rejects a world larger than its configured `N_max` and rejects an
edit-distance diagnostic outside the corresponding support. It never clips or
silently rescales the reward. A harmless nonminimal set receives `1` whenever
the complete submitted set itself remains graphically valid.

The scorer uses only the DAG. It does not enumerate CPT states, standardize
observational distributions, weight variables by mechanism strength, compare
families of minimal sets, or compute set F1. The nearest-set search is exact on
the complete generated support of at most 16 nodes.

All five owner-produced terminal qualities enter GRPO unchanged. Benchmark
reports retain nearest-valid-set edit distance and valid-set rate as the
primary backdoor metrics; mean terminal quality is a training-compatible
summary of the same fixed linear scale.
