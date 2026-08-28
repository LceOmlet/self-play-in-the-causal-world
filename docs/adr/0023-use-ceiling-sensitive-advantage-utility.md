---
status: accepted
implementation: implemented
amended_by: 0030-normalize-decision-regret-by-candidate-span
---

# Use a bounded ceiling-sensitive utility for selected GRPO task families

The environment-owned `terminal-quality-v1` remains the semantic reward and
the reported evaluation quantity. Before GRPO group-mean subtraction, the
trainer maps terminal quality `Q` to

\[
U_\epsilon(Q)=
\frac{\log\left((1+\epsilon)/(1-Q+\epsilon)\right)}
     {\log\left((1+\epsilon)/\epsilon\right)},
\qquad \epsilon=0.02.
\]

This bounded monotone transform fixes both endpoints, preserves every within-
group ordering, and expands differences near full quality without the
singularity of an unbounded negative-log residual. It is a training utility,
not a likelihood and not a new terminal reward.

The transform applies to `ate`, `individual_counterfactual_probability`, and
`backadj_minimal_sets`. Reward v4 gives `best_intervention` a span-normalized
semantic reward, so that family now uses identity utility together with
`mediator_set`. Raw terminal quality remains an explicit zero-weight TRL reward
source for diagnostics, while the selected utility is the sole source used to
form advantages. Both values are logged by task family.

Group standard-deviation normalization remains disabled. Equal-quality groups
still produce exactly zero advantage. The task mixture, terminal parsers,
truth owners, success criteria, and environment reward contract are unchanged.
