---
status: superseded
superseded_by: 0021-separate-parent-main-effect-blocks
---

# Sample isotropic directions in the complete joint-effect space

For a node with \(R\) joint parent configurations and \(d\) child states, first draw an \(R\times d\) table \(G\) with independent standard-normal entries. Project \(G\) onto the subspace

\[
\mathcal E=\left\{D:\sum_yD(c,y)=0\ \forall c,\quad
\sum_cD(c,y)=0\ \forall y\right\},
\]

then divide the projection by its Frobenius norm. The resulting \(D_Y\) is uniformly distributed over the unit sphere in \(\mathcal E\). The Gaussian values are only a device for drawing an isotropic direction; they are not the distribution of the final effect magnitudes.

The completed effect table remains

\[
\Delta_Y=s_Ya_{\max}D_Y,
\qquad s_Y\sim\operatorname{Uniform}(0,1),
\]

where \(a_{\max}\) is the largest positive legal scale around \(b_Y\).

No separate main-, pairwise-, or higher-order interaction strengths are sampled. When a node has more parents or parent states, higher-order interaction subspaces contain more dimensions and therefore receive more aggregate variation under isotropic sampling. This is an accepted property: the existing parent-count and node-cardinality priors supply the mixture of simpler and more complex mechanisms.
