# Counterfactual root-relaxation run

The solver remains at 23/30 certified tasks. No solver candidate from this run
was retained.

The decisive aggregate result is that six of seven unresolved tasks never
leave root-node processing, and every unresolved endpoint is dual-limited.
Five have more than 5,000 master variables, while local response context widths
remain 8 or 16. This locates the remaining bottleneck in the global nonlinear
root formulation rather than branch selection or local response pricing.

Seven semantics-preserving candidates reached the frozen closure gate and were
rejected: broader official RLT separation, single-shared-variable
stratified Frechet bounds, official quadratic intersection cuts, static
materialization of small wide response blocks, and official objective
reoptimization, official nonlinear OBBT, and propagated message lower bounds.
Two additional circuit-reduction candidates were rejected by the focused
performance gate before cohort measurement. No candidate increased closure.

The symbolic elimination audit also rules out high-order multiplication as the
common cause: every unresolved circuit has simultaneous product arity at most
two, and the fraction of terms with arity three or greater is zero. The
remaining hard object is a large shared bilinear circuit, not a recursively
associated three-way or higher-order product.

The next admissible direction must reduce or convexify the global elimination
circuit itself without expanding its affine subexpressions. Additional solver
toggles, scalar message bounds, response pricing changes, redundant-row
removal, and small-block routing are no longer supported by the aggregate
evidence as routes to higher closure.

A subsequent bilinear-aware ordering reduced the unresolved cohort's median
bilinear auxiliary count to 8.23% of the baseline and cut aggregate runtime to
171.39 seconds, but closure remained 23/30. Combining that formulation with
message lower bounds or external spatial branching also left closure
unchanged. Exact and epsilon certificates based on attained global outer
endpoints found no additional certifiable task. Finally, SCIP's projected and
hidden-product RLT path produced no closure gain. These results separate model
size from relaxation strength: the remaining obstacle is the joint convex
relaxation across shared bilinear mechanism messages, not merely the number of
products, root branching policy, or an unused termination certificate.

Two later exact structural projections also failed the closure gate.  A
binary single-edge response mechanism can be reduced to its two transportation
vertices, but this did not close its unresolved member.  A dedicated pure
indirect one-mediator formulation eliminated the generic arithmetic circuit;
after replacing explicit response expansion with exact on-demand pricing and
aggregating shared assignments before the bilinear product, its resource use
returned to normal and it correctly owned the intended wide class.  The class
still failed to certify within five seconds.  This rules out local response
enumeration and two-mechanism circuit redundancy as the missing source of a
closure gain.

The final diagnostics compare every failed joint-scale bracket directly with
the task's accepted endpoint gap under the 0.001 conditional contract.  Five
to seven failures, depending on fixed-budget runtime variation, remain more
than 100 times outside tolerance.  Only one or two failures approach the
2--10-times range.  One is a direct-terminal layered graph that is
structurally eligible for the existing exact two-mediator owner but exceeds
its upstream 4096-vertex-product guard; another observed near case is an
attainable endpoint-only chain with a five-state first mediator.

Routing oversized explicit layered objectives to the existing exact pricer
did not reach the former class and regressed one previously closed task.  A
strictly equivalent compact formulation then removed the upstream Cartesian
product, with reference endpoints agreeing to at most 7.82e-10.  It correctly
took ownership of both structural classes but still produced no new
five-second certificate.  The main solver therefore retains none of these
candidates.

A complete shared-parent-boundary stratification tightened the best
single-variable Frechet interval on 16 of 30 optimization instances, reducing
the median outer width from 0.10190 to 0.08287.  None of those outer intervals
was itself within the 0.001 endpoint tolerance, and passing the tighter bounds
to the unchanged exact owner left closure at 23/30 while increasing total
runtime from 325.41 to 413.29 seconds and P95 from 23.83 to 145.66 seconds.
This rejects further outer-bound stratification as the current closure lever:
the unresolved root gap lies inside the compatible-mechanism optimization,
not in the available interventional Frechet envelope.

Sharp local transport bounds were also tested at every numeric leaf
contraction.  The tiny residual-flow solver matched the existing exact
transport LP to 3.89e-16 over 80 random two- through five-state problems, but
the frozen cohort again remained 23/30.  Exact entrywise message boxes are
therefore insufficient: the residual dual gap depends on simultaneous
consistency across messages that share the same response mechanisms.

The official nonlinear handler's ten-auxiliary-expression cap was then
removed entirely for every bilinear term.  Closure again remained 23/30 while
total runtime rose to 417.75 seconds and P95 to 147.62 seconds.  The shared
mechanism information is therefore absent from the relaxation itself, rather
than merely hidden behind SCIP's auxiliary-expression registration limit.

Finally, the original circuit and the regression-safe minimum-fill circuit
were compared instance-by-instance while reporting only their aggregate
contingency.  All 23 closed tasks were shared and all seven failures were
shared; neither formulation had an exclusive certificate.  The sequential
audit also reached an observed 1.72 GB working set.  A parallel formulation
portfolio therefore has no closure-set union gain and would only duplicate
the expensive global model.

A custom two-mechanism RLT lift then targeted the remaining mathematical
gap directly.  It replaced each quadratic kernel term by an explicit product,
retained the exact nonlinear equality, and added both transportation tables'
row- and column-marginal product identities.  The complete 10,000-cell class
passed query-truth endpoint regression but left closure at 23/30.  Extending
the same complete lift to 100,000 cells reached an observed 1.72 GB working
set during construction.  Thus first-level pairwise mechanism consistency is
still too weak, while materializing the next wider pairwise surface is already
outside the useful five-second regime.

The root-separator fast path was also generalized to shared non-root variables
whose state indexes every affected response mechanism.  The construction kept
the separator as evidence, so correlations with its shared ancestors were not
dropped, and its endpoints matched the unsplit exact owner.  On the complete
eligible frozen subcohort, however, closure fell from 5/6 to 4/6 and measured
runtime rose from 13.74 to 27.27 seconds.  Repeated state-specific model builds
cost more than the decomposition removed, so this exact route was rejected
before a full-cohort run.

The earlier bilinear-aware elimination order was then rebuilt under the final
0.001 endpoint tolerance.  It again reduced runtime (177.31 seconds for the
frozen cohort) but remained at 23/30.  An anonymous instance-level contingency
showed 23 tasks closed by both formulations, no exclusive certificate on
either side, and seven common failures.  The smaller circuit therefore cannot
be used as a formulation portfolio to expand closure.

Batch response pricing was tested after the final profiler showed four failed
owners without a completed pricing certificate.  Each exact MAP response was
augmented by every rigorously improving Hamming-one neighbor.  Closure remained
23/30; more importantly, pricing-closed failures fell from three to two while
the nonlinear master accumulated more columns.  The incomplete pricing state
is therefore downstream of the root bottleneck rather than evidence that
one-column pricing rounds are the bottleneck.

An exact junction-tree response formulation then replaced cyclic local
response tables by clique marginals and separator consistency.  The 100,000-
cell and existing 3,125-cell explicit boundaries both remained at 23/30, with
P95 latency increasing to 152.19 and 176.68 seconds respectively.  Compactly
closing one local marginal polytope therefore does not close the global
cross-mechanism relaxation and is too expensive as an explicit master.

The final scalar-bound audit added the exact Frechet lower bound to every
pair-response kernel.  Alone it cut total runtime to 194.48 seconds; combined
with bilinear-aware elimination it reached 138.61 seconds and P95 17.71
seconds.  The latter also reduced failures more than 100 times outside the
accepted gap from four to two and left one endpoint within a factor of two.
Closure nevertheless stayed 23/30.  Adding official projected/hidden RLT raised
P95 to 108.60 seconds, while propagating the lower bounds through elimination
messages raised P95 to 51.91 seconds; neither closed the near endpoint.  The
remaining missing information is therefore joint shared-mechanism consistency,
not another entrywise kernel or message bound.

A full twin-outcome partition then added two disjoint complement circuits and
the exact identity that target plus complements equals one.  Restricting this
to static-response joint owners preserved the branch-and-price lifecycle and
passed 71 focused tests, but closure again remained 23/30 while P95 rose to
151.39 seconds.  Outcome conservation is therefore too costly when represented
by duplicate circuits; any useful higher-order consistency must be generated
implicitly.

The next structural audit ruled out a tempting but irrelevant exact fast path.
None of the seven unresolved instances is fully binary, none has only forest
response-context components, and neither terminal-endpoint reduction leaves a
binary forest instance.  Exact 2U-style propagation or binary transport
enumeration therefore has zero coverage of the remaining closure gap.

Two upstream separators that had not been covered by the earlier RLT and minor
experiments were then tested without local tuning.  Enabling SCIP's flower
separator on continuous product expressions kept closure at 23/30 and raised
P95 latency to 153.26 seconds.  Enabling the edge-concave separator to aggregate
quadratic rows also kept closure at 23/30 and raised P95 latency to 151.65
seconds.  Generic product-hypergraph and signed-cycle inequalities therefore do
not propagate the response-marginal equalities through the elimination layers;
the next candidate must target that missing equality structure directly.

The individualized conditioning event exposes a cheaper conservation identity:
the target joint mass plus its counterfactual complement within the same
factual event equals the known factual mass.  A one-complement implementation
passed 70 focused tests (one skip) and reduced the duplicate-circuit P95 from
the earlier 151.39 seconds to 43.15 seconds, but closure remained 23/30 and
total runtime remained above baseline at 419.17 seconds.  The identity is exact
but still cannot be materialized as another full circuit; any further use must
separate its consequence implicitly in the existing graph.

The remaining failures were then classified by the response mechanisms that
actually survive endpoint reduction.  Three of seven use only two mechanisms,
and each combines one cyclic response polytope with one forest response
polytope.  This does not yield an enumerable exact side: the smallest complete
forest vertex product already exceeds one million responses.  Exact
one-mechanism enumeration is therefore not a five-second fast path for any of
these two-mechanism failures.

Finally, an exact principal-moment lift exposed a square relation for every
probability variable that already participates in a bilinear product.  This
gave SCIP's upstream minor separator the square moments needed to couple
repeated products through positive-semidefinite principal cuts, while
preserving the original feasible set.  All focused owner comparisons passed,
but the frozen cohort remained 23/30 and P95 rose from the 23.83-second
baseline to 49.32 seconds.  Second-order moment consistency is consequently
too weak; the missing constraint class must transmit complete
response-marginal equalities across multiple elimination layers.

The same square lift was then paired with SCIP's disabled-by-default
`interminor` separator, which enforces zero 2-by-2 determinants of the product
matrix through intersection cuts instead of keeping only principal
semidefinite inequalities.  This stronger rank-one treatment also remained at
23/30, while P95 rose to 152.47 seconds.  Both generic second-order routes have
therefore been exhausted: the next candidate must explicitly reuse the linear
response-marginal equalities at the elimination-message level.

A dedicated aggregate profiler then measured that proposed
response-marginal/message lift before implementation.  Across the seven failed
owners, completing the exact row- and column-marginal products would require
6,369,908 products, of which 5,720,016 are absent; the per-owner missing count
has median 172,800 and P95 2,692,800.  A zero-expansion projection is also
impossible: 419,492 kernel/message groups contain one existing product, the
remaining 115,200 contain two products that never share a kernel row or
column, and therefore no projectable row/column subset exists.  This closes
both implementations of message RLT before solver development rather than
repeating the earlier Cartesian-product failure.

An exact transport-coordinate formulation next removed the row/column linear
redundancy before nonlinear elimination.  Each forest table retained a free
block and one pivot variable, while every other entry became an affine function
of the fixed CPT marginals.  The representation passed all focused owner
comparisons, but closure stayed 23/30 and P95 rose to 153.61 seconds because
the dependent entries expanded inside downstream products.  The missing dual
strength therefore belongs to the joint transport/message convex hull, not to
the coordinate description of one transport polytope.

The product audit was then lifted from individual transport tables to complete
response owners.  Although each nonlinear equation contains only one product
from any response-owner/message group, the same message is reused globally:
7,584 distinct response signatures appear across the seven failed owners.
Only 384 signatures contain a cycle in their response-context graph, and every
strict exact support improvement occurs in this cyclic subset (97 upper and 42
lower signatures, reused 19,400 and 4,200 times respectively); no forest
signature improves on its entrywise Frechet bounds.  This gives a complete,
relabeling-invariant screen and reduces bidirectional exact support work from
22.70 to 2.07 seconds.  A redundant aggregate product lift passed all focused
comparisons, including an activated fixture within 5.56e-9 of exhaustive
endpoints, but left both the frozen cohort (23/30) and formal cohort (20/30)
unchanged.  The useful cycle facets are dispersed across separate nonlinear
equations, so one aggregate equality does not expose their perspective in the
root relaxation.

The direct perspective form was then tested on the complete eligible class.
Among 288,000 cyclic product occurrences, exact support was strict for 23,600
groups covering 129,000 products; the entire strict class happened to reside
in one structurally eligible failed owner, with no identity-based selection.
Adding all 23,600 perspective rows passed the focused endpoint suite but again
left closure at 23/30, while total time rose to 353.75 seconds and P95 to 33.98
seconds.  The candidate was removed: local cycle-support facets remain too weak
even when placed directly in the shared product layer.  A subsequent anonymous
cross-profile placed its sole covered owner more than 100 times outside the
accepted endpoint gap, ruling out an unsupported combination experiment with
the previously faster elimination order.
