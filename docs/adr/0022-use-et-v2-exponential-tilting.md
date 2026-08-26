---
status: accepted
implementation: implemented
amended_by: 0025-use-unit-expected-et-v2-score-strength
---

# Use ET-V2 exponential tilting for sampled CPT transitions

For a non-root child node, retain the simplex-uniform base distribution
\(b_Y\), the combined score direction \(D_Y(x,y)\) from the active score-block
owner (currently ADR 0024), and the strength law owned by ADR 0025. Let

\[
\sigma_D^2=\frac{1}{|\mathcal X|d_Y}
\sum_{x,y}D_Y(x,y)^2.
\]

ET-V2 defines each CPT row by

\[
P(Y=y\mid x)=
\frac{b_Y(y)\exp\!\left(s_YD_Y(x,y)/\sigma_D\right)}
{\sum_z b_Y(z)\exp\!\left(s_YD_Y(x,z)/\sigma_D\right)}.
\]

Every output is strictly positive and row-normalized. The transition needs no
additive boundary scale: a rare state changes by its own positive multiplier
and cannot cap every other row, state, or parent effect. Unit elementwise RMS
makes the strength coordinate independent of the number of CPT rows and child
states. The existing parent/state exchangeability and block-energy law are
unchanged; main effects and interactions are now score-space effects.

The fixed-seed prototype profile that admitted this decision changed the
single-parent ATE median from 0.0267 to 0.0764 and the five-state median from
0.00724 to 0.0349. Across five task families, ATE and decision medians rose by
about 5.4 times, the 30-seed exact counterfactual closure stayed 27/30 with the
same unresolved seeds, and graph-only backdoor and mediator answers remained
identical.
