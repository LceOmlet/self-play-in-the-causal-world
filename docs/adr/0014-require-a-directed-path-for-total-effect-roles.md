---
status: accepted
implementation: planned
---

# Require a directed path for total-effect roles

An ordered variable-role assignment \((X,Y)\) is structurally eligible for a total-effect task exactly when

\[
X\ne Y
\qquad\text{and}\qquad
X\text{ is an ancestor of }Y.
\]

Under the finite DAG and mechanism-replacement semantics of a CPT-World, an intervention on \(X\) can change the distribution of \(Y\) only through a directed path from \(X\) to \(Y\). A common ancestor may make \(X\) and \(Y\) observationally associated, but intervening on \(X\) cuts the incoming arrows to \(X\) and does not turn that association into a causal effect.

The path condition is only structural. The sampler does not inspect the realized effect magnitude, sign, or task answer. A directed path is necessary for a possible effect but does not guarantee that the realized numerical contrast is nonzero.
