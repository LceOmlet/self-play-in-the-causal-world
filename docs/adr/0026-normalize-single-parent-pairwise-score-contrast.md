---
status: accepted
implementation: implemented
amends: 0022-use-et-v2-exponential-tilting
---

# Normalize scores by contextual parent-pair contrast

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

For a multi-parent score table, extend the same geometry without changing its
sampled ANOVA direction. For each parent, compare every pair of its states
while holding all other parents fixed. Average the squared elementwise score
contrast symmetrically over parents, fixed contexts, parent-state pairs, and
child states:

\[
\sigma_{\rm tr}^2(D)
=
\operatorname{Avg}_{j,a_j<a'_j,a_{-j},y}
\left[D(a_j,a_{-j},y)-D(a'_j,a_{-j},y)\right]^2.
\]

Render the ET-V2 score as \(2D/\sigma_{\rm tr}\). This changes only the radial
coordinate. Parent-subset directions and their realized energy shares from
ADR 0024 remain identical. High-order interactions count as transmission when
changing a participating parent changes the child distribution in a fixed
context; they are not suppressed because their marginal effect may cancel.
The single-parent branch continues to use the analytic scale above and is
bit-for-bit unchanged.

In a paired, fixed-seed 2,000-world prototype, the correction left binary
single-route CPTs unchanged. For parent cardinalities three, four, and five,
the median realized row-pair TV changed from 0.380/0.323/0.297 to
0.428/0.379/0.357, while the Kolmogorov--Smirnov distance to a uniform TV law
changed from 0.125/0.185/0.209 to 0.078/0.122/0.144. For five-state parents,
the share of generated rows with maximum probability above 0.99 changed from
about 2.9% to 4.8%.

In a paired 1,000-seed probe of the 8--16-node task sampler, the contextual
extension changed the median conditional row-pair TV for two- and three-parent
CPTs from 0.259/0.251 to 0.333/0.332. The corresponding Kolmogorov--Smirnov
distances to a uniform TV law changed from 0.337/0.415 to 0.234/0.310. Median
absolute ATE changed from 0.0230 to 0.0364 and its below-0.01 share from 38.1%
to 29.5%. Median decision gap changed from 0.0289 to 0.0372 and its below-0.01
share from 32.8% to 28.4%. The share of CPT rows with maximum probability above
0.99 changed from 2.83% to 6.72%; 93.28% of rows remain outside that near-
deterministic tail. Graphs, roles, energy directions, and all structural task
truths were paired identically.
