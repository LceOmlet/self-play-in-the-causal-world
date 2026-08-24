---
status: accepted
---

# Use a bounded parent-count structure prior

Generated worlds first uniformly randomize the node order. At position (j),
the node draws (k) uniformly from (0,\ldots,\min(3,j)), then draws a
(k)-element parent subset uniformly from the preceding nodes.

We intentionally do not make labeled DAGs equiprobable. Equal coverage of the
available local interaction orders and a hard three-parent CPT bound matter
more for this task generator than graph-level equiprobability. The bound keeps
multivalued CPTs tractable while allowing long paths, forks, confounders, and
colliders as the node count grows.
