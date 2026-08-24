---
status: accepted
---

# Preserve node count under task eligibility sampling

Each episode first samples its node count uniformly from 3 through 15. That
count remains fixed while DAGs of the same size are resampled until the chosen
task's structural eligibility condition is satisfied.

Resampling the node count together with the DAG would favor larger worlds,
because they more often contain paths and motifs required by the task. Holding
the sampled size fixed preserves the declared uniform node-count distribution
while allowing each task to condition on its required graph structure.
