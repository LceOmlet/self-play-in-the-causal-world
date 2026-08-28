# Counterfactual root-relaxation run

The solver remains at 23/30 certified tasks. No solver candidate from this run
was retained.

The decisive aggregate result is that six of seven unresolved tasks never
leave root-node processing, and every unresolved endpoint is dual-limited.
Five have more than 5,000 master variables, while local response context widths
remain 8 or 16. This locates the remaining bottleneck in the global nonlinear
root formulation rather than branch selection or local response pricing.

Five semantics-preserving candidates were tested and rejected by the frozen
closure gate: broader official RLT separation, single-shared-variable
stratified Frechet bounds, official quadratic intersection cuts, static
materialization of small wide response blocks, and official objective
reoptimization. The last candidate reduced aggregate runtime and P95, but did
not increase closure and changed one exact certificate into epsilon-sharp, so
it was not merged under the stated objective.

The next admissible direction must reduce or convexify the global elimination
circuit itself. Additional solver toggles, probability outer bounds, response
pricing changes, and small-block routing are no longer supported by the
aggregate evidence as routes to higher closure.

