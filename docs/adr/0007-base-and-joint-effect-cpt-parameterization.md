---
status: superseded
superseded_by: 0022-use-et-v2-exponential-tilting
---

# Parameterize each CPT by a base distribution and a joint parent-effect table

For a node \(Y\) with \(R\) joint parent configurations and \(d\) states, write

\[
P(Y=y\mid \operatorname{Pa}(Y)=c)=b_Y(y)+\Delta_Y(c,y).
\]

The effect table satisfies

\[
\sum_y \Delta_Y(c,y)=0\quad\text{for every }c,
\qquad
\frac{1}{R}\sum_c \Delta_Y(c,y)=0\quad\text{for every }y.
\]

Root nodes use \(b_Y\) directly. Every legal CPT has the unique decomposition

\[
b_Y=\frac{1}{R}\sum_c P(Y\mid c),
\qquad
\Delta_Y(c)=P(Y\mid c)-b_Y.
\]

The decision therefore does not restrict the set of representable CPTs. Treating all parents jointly through \(c\) permits arbitrary main and interaction effects without separate parameter families.

Generate an effect table by drawing a centered direction \(D_Y\) whose law is invariant to parent, parent-state, and child-state relabelling; finding the largest positive scale \(a_{\max}\) for which every \(b_Y+aD_Y(c)\) is nonnegative; drawing \(s_Y\sim\operatorname{Uniform}(0,1)\); and setting

\[
\Delta_Y=s_Ya_{\max}D_Y.
\]

Thus \(s_Y\) uniformly covers the available effect strength along the sampled direction. In the binary one-parent case with \(b_Y=(1/2,1/2)\), symmetric direction and magnitude recover the earlier signed effect sampled uniformly from \([-1/2,1/2]\).

The exact distribution of \(D_Y\) is specified separately. This decision makes no claim that the realized marginal effect of every individual parent edge is uniform.
