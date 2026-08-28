---
status: accepted
implementation: implemented
amends: 0024-balance-energy-by-parent-subset
---

# Sample clique-supported parent interactions

For a node with parent positions \(1,\ldots,p\), sample an undirected graph
uniformly from the \(2^{\binom p2}\) labeled graphs on those parents. Activate
the functional-ANOVA block for a nonempty parent subset \(S\) exactly when
\(S\) is a clique. Singleton main effects are therefore always active. A
higher-order interaction is active only when every pair inside it interacts.

For three parents, the eight equally likely edge masks produce: main effects
only with probability \(1/8\), exactly one pair interaction with probability
\(3/8\), exactly two pair interactions with probability \(3/8\), and the
complete seven-block mechanism with probability \(1/8\). For two parents,
the pair interaction is active with probability \(1/2\). The one-parent law
is unchanged and consumes no support-sampling randomness.

Conditional on the active support, retain ADR 0024's independent isotropic
block directions and symmetric simplex energy split. The combined direction
still has unit Frobenius norm. The simplex-uniform base, ET-V2 map, node
strength, graph sampler, roles, state anchors, K, M, and task owners are
unchanged. The construction adds no tunable sparsity, order, rank, or task
filter parameter and remains invariant to parent and state relabeling.

On 500 sampled worlds, the median ratio of uniformly marginalized TV to
fixed-context TV changed from 0.792 to 0.985 for two-parent mechanisms and
from 0.469 to 0.822 for three-parent mechanisms. Median fixed-context TV
changed from 0.333 to 0.307 and from 0.315 to 0.284 respectively, so the
larger marginal effects came from reduced cancellation rather than larger
conditional contrasts. CPT rows with maximum probability above 0.99 changed
from 6.62% to 8.74%, and those above 0.999 from 1.92% to 2.79%.

On 400 fixed structural task seeds, absolute ATE median changed from 0.0370
to 0.0418 and the share below 0.01 from 29.75% to 25.75%. Best-intervention
span median changed from 0.1096 to 0.1441 and its below-0.01 share from 16.50%
to 10.75%. Back-door and mediator truths depend only on the unchanged graph.

Under the formal individual-counterfactual endpoint tolerance and a five-
second SCIP allocation per endpoint, seeds 0--29 changed from 24 exact plus
3 epsilon-sharp certificates to 24 exact plus 1 epsilon-sharp certificate.
This task retains its existing unresolved fallback and is not used to alter
the world distribution or the accepted counterfactual semantics.

This 27/30-to-25/30 count is a distribution probe, not a paired solver-
complexity ablation. Sampling the parent-interaction graph consumes randomness
before the remaining CPT draws, so a common integer seed preserves the sampled
DAG but does not preserve the CPT realization. Across seeds 0--29, the mean
per-world CPT-row TV between the old and new realizations was 0.475, and 14 of
30 individual-counterfactual tasks sampled a different factual outcome state.
The two newly unresolved tasks, seeds 9 and 11, were both in that set. Holding
the old CPT fixed and changing only to the new factual state made both tasks
time out; holding the new CPT fixed and restoring the old factual state also
left both unresolved. In every failure, response pricing closed and the global
maximization proof retained the gap. The closure change therefore combines a
different CPT realization and a different query event; it is not evidence that
fewer active score blocks increase or decrease verifier complexity.

## Upstream answer-reference panel

The pinned CLadder and bnlearn worlds are used only as an external answer-
distribution reference. They do not enter, condition, filter, or replace the
main world sampler. For each upstream world, the diagnostic evaluates every
structurally legal ordered role assignment once. Numerical tasks use one
seed-fixed draw from the existing symmetric state-anchor rule per role. The
table therefore describes this finite reference panel, not a population law.

| World | median \(|\mathrm{ATE}|\) | median decision gap | empty back-door answer | mean mediator count | certified CF / roles; median width |
|---|---:|---:|---:|---:|---:|
| Main sampler | 0.0418 (400) | 0.1441 (400) | 25.25% (400) | 1.80 (400) | 25/30; 0.2291 |
| CLadder mediation | 0.4557 | 0.4557 | 66.67% | 1.00 | 3/3; 0.6903 |
| CLadder diamond-cut | 0.3309 | 0.3309 | 60.00% | 2.00 | 5/5; 0.2760 |
| CLadder collision | 0.3535 | 0.3535 | 100.00% | -- | 2/2; 0.2843 |
| CLadder confounding | 0.5760 | 0.5760 | 66.67% | 1.00 | 3/3; 0.1449 |
| Asia | 0.3289 | 0.3289 | 83.33% | 1.50 | 18/18; 0.0185 |
| Cancer | 0.0198 | 0.0198 | 100.00% | 1.00 | 8/8; 0.0176 |
| Earthquake | 0.6670 | 0.6670 | 100.00% | 1.00 | 8/8; 0.0132 |
| Survey | 0.0060 | 0.0096 | 84.62% | 1.71 | 13/13; 0.2500 |

The reference panel brackets the main sampler on the three numerical task
families: Cancer and Survey supply weak-effect cases, while Earthquake, Asia,
and the CLadder motifs supply strong-effect cases. The main sampler's back-
door answers are less often empty and its mediator counts lie within the
upstream range. All 60 upstream counterfactual role queries closed exactly;
their smaller graphs make this a semantic reference rather than a solver-
scalability comparison.
