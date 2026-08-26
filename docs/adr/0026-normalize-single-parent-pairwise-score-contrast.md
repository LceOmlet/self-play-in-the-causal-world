---
status: accepted
implementation: implemented
amends: 0022-use-et-v2-exponential-tilting
---

# Normalize single-parent scores by mean pairwise contrast

For a single-parent CPT with parent cardinality (k), retain the
simplex-uniform base, the isotropic pure-main-effect score direction, the
ET-V2 exponential map, and the amplitude law from ADR 0025. After the usual
elementwise RMS normalization, multiply the complete score table by

\[
c_k=\sqrt{\frac{2(k-1)}{k}}.
\]

Equivalently, if (D(x,y)) is the centred score table, define

\[
\sigma_{\rm pair}^2=
\frac{1}{\binom{k}{2}d_Y}
\sum_{x<x'}\sum_y [D(x,y)-D(x',y)]^2
\]

and render the ET-V2 score as (2D/\sigma_{\rm pair}). The resulting mean
elementwise squared contrast over all parent-state pairs is exactly four.
For (k=2), the two centred rows are opposites, (c_2=1), and the generated
CPT is bit-for-bit unchanged.

This scale follows from the centred-table identity: an elementwise
RMS-normalized (k)-row table has mean pairwise squared score contrast
(2k/(k-1)), whereas the binary value is four. It therefore removes only the
parent-cardinality attenuation. It treats every parent-state pair
symmetrically, changes neither the sampled direction nor child-state transfer
direction, and introduces no task-answer or realized-effect filtering.

The correction is intentionally limited to single-parent CPTs. Multi-parent
tables retain ADR 0024 and ADR 0025 unchanged until their conditional
main-effect and interaction geometry is reviewed separately.

In a paired, fixed-seed 2,000-world prototype, the correction left binary
single-route CPTs unchanged. For parent cardinalities three, four, and five,
the median realized row-pair TV changed from 0.380/0.323/0.297 to
0.428/0.379/0.357, while the Kolmogorov--Smirnov distance to a uniform TV law
changed from 0.125/0.185/0.209 to 0.078/0.122/0.144. For five-state parents,
the share of generated rows with maximum probability above 0.99 changed from
about 2.9% to 4.8%.
