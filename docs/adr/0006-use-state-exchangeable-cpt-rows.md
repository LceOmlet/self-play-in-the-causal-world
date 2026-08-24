---
status: superseded
superseded_by: 0007-base-and-joint-effect-cpt-parameterization
---

# Use state-exchangeable CPT-row sampling

Every root distribution and every conditional row is sampled independently
from \(\operatorname{Dirichlet}(1,\ldots,1)\) at the node's cardinality. This
draws a generally nonuniform categorical distribution uniformly over the
probability simplex while making every permutation of the state labels equally
likely. For binary variables it reduces to a uniform draw of one state's
probability on \((0,1)\).

This replaces the state-index matching mechanism, multiplicative odds, and
pre-sampled signed edge effects. A signed scalar edge parameter has no
label-invariant meaning for nominal multistate variables. Task effects are
computed from the completed CPT. The table shape is unchanged, so under the
five-state and three-parent bounds the construction, storage, experimental
sampling, and exact-inference complexity retain the same asymptotic bounds.
