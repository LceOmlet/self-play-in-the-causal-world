---
status: accepted
implementation: planned
---

# Use ancestor pairs for experimental-decision roles

For an experimental-decision task, an ordered variable-role assignment \((D,Y)\) is structurally eligible exactly when

\[
D\ne Y
\qquad\text{and}\qquad
D\text{ is an ancestor of }Y.
\]

The outcome may have children. Descendants of \(Y\) do not change the terminal object, which compares \(P(Y=y^\star\mid do(D=d))\) over all states \(d\) of the decision variable. Requiring \(Y\) to be a leaf would therefore condition the task distribution on an irrelevant graph property.

The existence of the directed path already implies that \(Y\) has a parent, so no separate parent-count condition is used.

After \((D,Y)\) is selected, sample `minimize` or `maximize` independently and uniformly. The objective is a task attribute, not part of the variable-role assignment.

The current implementation's leaf-outcome restriction and objective-bearing anchor records are superseded and have not yet been migrated.
