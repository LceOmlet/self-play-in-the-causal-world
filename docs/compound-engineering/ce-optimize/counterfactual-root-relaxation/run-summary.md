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
