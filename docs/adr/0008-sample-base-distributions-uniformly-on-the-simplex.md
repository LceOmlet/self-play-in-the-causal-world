---
status: accepted
---

# Sample base distributions uniformly on the probability simplex

For every \(d\)-state node, sample its base distribution as

\[
b_Y\sim\operatorname{Dirichlet}(1,\ldots,1).
\]

This distribution is uniform on the \(d\)-state probability simplex and invariant to every permutation of the state labels. For a binary node, it reduces exactly to sampling one state's probability uniformly from \([0,1]\).

No entropy target, concentration mixture, or radial construction is added. The joint parent-effect table is sampled separately under its centering and legality constraints.
