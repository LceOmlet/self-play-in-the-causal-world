---
status: accepted
implementation: implemented
amended_by: 0022-use-et-v2-exponential-tilting
---

# Separate parent main-effect blocks under one conserved energy budget

For a child node (Y) with direct parents (X_1,\ldots,X_k), retain the
accepted base-plus-effect parameterization

\[
P(Y=y\mid x)=b_Y(y)+s_Ya_{\max}D_Y(x,y).
\]

Use the uniform measure over the finite parent-state product.  The pure main
effect of parent (X_j) is the subspace of row-centred tables that depend on
the joint parent assignment only through (x_j).  Its orthogonal projection is

\[
g_j(x_j,y)=\frac{1}{\prod_{\ell\ne j}d_\ell}
\sum_{x_{-j}}D_Y(x_j,x_{-j},y).
\]

Draw one isotropic unit direction (D_j) independently in each parent main-
effect subspace.  When (k\ge2), draw one further isotropic unit direction
(D_{\ge2}) in the orthogonal complement of all main-effect subspaces.  This
last direction deliberately remains the unsplit direct sum of all interaction
orders two and above; no higher-order partition or support-selection rule is
introduced by this decision.

Draw squared-energy shares uniformly on the component simplex,

\[
(w_1,\ldots,w_k,w_{\ge2})\sim\operatorname{Dirichlet}(1,\ldots,1),
\]

omitting (w_{\ge2}) when (k=1), and combine

\[
D_Y=\sum_{j=1}^k\sqrt{w_j}D_j+\sqrt{w_{\ge2}}D_{\ge2}.
\]

The component subspaces are orthogonal and every direction has unit
Frobenius norm, hence

\[
\lVert D_Y\rVert_F^2=\sum_jw_j+w_{\ge2}=1.
\]

Adding a component therefore reallocates a fixed effect budget rather than
increasing total effect energy.  Parent and state labels remain exchangeable.
The accepted simplex-uniform base law and
(s_Y\sim\operatorname{Uniform}(0,1)) are unchanged. This ADR owns the score-
space block decomposition. ADR 0022 owns the downstream map from the combined
score direction to legal CPT probabilities.

This decision removes the previous allocation of expected energy in
proportion to ANOVA-subspace dimension.  It does not decide how order-two-and-
higher interactions should later be partitioned or selected.
