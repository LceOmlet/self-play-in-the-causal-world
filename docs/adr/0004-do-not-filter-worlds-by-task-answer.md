---
status: accepted
---

# Do not filter worlds by the numerical task answer

Mechanism samples are retained whenever the CPT world is legal and the chosen
task is structurally eligible. The generator does not reject exact zero total
effects or tied optimal actions.

Both cases have well-defined terminal answers. Filtering them would condition
the mechanism distribution on the task truth and, for counterfactual bounds,
could discard a nontrivial interval merely because its associated average
treatment effect is zero. Null-effect and tie frequencies are reported as
distribution statistics rather than used as acceptance gates.
