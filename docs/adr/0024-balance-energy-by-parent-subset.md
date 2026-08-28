---
status: accepted
implementation: implemented
supersedes: 0021-separate-parent-main-effect-blocks
amended_by: 0027-sample-clique-supported-parent-interactions
---

# Balance expected score energy by parent subset

For a child node with parent positions \(\{1,\ldots,p\}\), index score blocks
by every nonempty parent subset

\[
\mathcal S_p=\{S:S\subseteq\{1,\ldots,p\},\ S\ne\varnothing\}.
\]

For each \(S\in\mathcal S_p\), independently draw an isotropic unit direction
\(D_S\) in the pure categorical functional-ANOVA interaction subspace for
exactly that subset. Draw squared-energy shares jointly as

\[
(q_S)_{S\in\mathcal S_p}\sim\operatorname{Dirichlet}(1,\ldots,1)
\]

and combine them as

\[
D_Y=\sum_{S\in\mathcal S_p}\sqrt{q_S}D_S.
\]

The pure subset subspaces are orthogonal, so \(\lVert D_Y\rVert_F=1\). Every
subset has expected share \(1/(2^p-1)\), while the realized shares remain
random rather than being fixed to their mean. All subsets are active almost
surely; this decision introduces neither sparsity nor task-answer filtering.

For three parents, the seven blocks are
\(A,B,C,AB,AC,BC,ABC\). Their individual expected shares are \(1/7\), so the
aggregate expected energies of orders one, two, and three are \(3/7,3/7,1/7\).
This avoids assigning the lone three-way interaction the same expected energy
as all three main effects together.

ET-V2 remains the probability-row mapping. The active node-level strength law
is owned by ADR 0025. This decision changes only the score-energy
decomposition.

On task seeds 0--1999, relative to equal expected energy per interaction order,
the absolute ATE median changed from 0.03118 to 0.03128, the decision-gap median
from 0.01822 to 0.02069, and the individual-counterfactual Fréchet-width median
from 0.48671 to 0.44477. A 30-seed exact counterfactual probe closed 28 tasks
under a five-second per-endpoint allocation. Sampling 300 complete worlds was
about 15% slower; graph structure and node domains remained identical for
every compared seed.

ADR 0027 retains this symmetric energy law conditional on the sampled active
support. It no longer activates every nonempty subset in every multi-parent
CPT.
