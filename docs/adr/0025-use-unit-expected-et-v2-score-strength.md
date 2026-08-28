---
status: accepted
implementation: implemented
amends: 0022-use-et-v2-exponential-tilting
---

# Use unit-expected squared score energy for ET-V2 strength

ET-V2 first divides the combined score table \(D_Y\) by its elementwise RMS
\(\sigma_D\). Therefore the normalized score table

\[
G_Y=D_Y/\sigma_D
\]

has unit elementwise squared energy. Draw the nonnegative amplitude as

\[
s_Y\sim\operatorname{Uniform}(0,\sqrt{3}).
\]

The log-score perturbation is \(s_YG_Y\), and

\[
\mathbb E[s_Y^2]=\frac{(\sqrt3)^2}{3}=1.
\]

Thus every non-root node receives one unit of expected squared log-score
energy independently of its CPT row count and child cardinality. Combined
with ADRs 0024 and 0027, that energy is split symmetrically across the active
clique-supported parent subsets. Realized strengths, interaction supports,
and active-subset shares remain random.

This replaces the earlier \(\operatorname{Uniform}(0,1)\) amplitude, whose
expected squared score energy was \(1/3\). The new bound is derived from the
existing RMS energy unit rather than fitted to ATE, decision gaps, path
lengths, or model performance. It leaves the simplex-uniform base,
parent-subset directions, ET-V2 probability map, topology, task roles, K, M,
and state-anchor distributions unchanged.

The bounded support keeps every generated real-valued ET-V2 row strictly
positive and avoids the numerical and semantic boundary behavior of an
unbounded strength tail.

On task seeds 0--1999, relative to amplitude
\(\operatorname{Uniform}(0,1)\), the absolute ATE median changed from 0.03128
to 0.05512, the indirect-only ATE median from 0.00432 to 0.01210, the fraction
of indirect effects below 0.01 from 63.42% to 45.23%, and the decision-gap
median from 0.02069 to 0.03634. The individual-counterfactual Fréchet-width
median changed from 0.44477 to 0.40283. Across sampled CPT rows, 2.72% had
maximum probability above 0.99 and 0.47% above 0.999. The 30-seed exact
counterfactual probe closed 28 tasks under a five-second per-endpoint
allocation. Five thousand complete sampled worlds contained no zero
probability; sampling 500 worlds added about 2.2% wall time.
