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
