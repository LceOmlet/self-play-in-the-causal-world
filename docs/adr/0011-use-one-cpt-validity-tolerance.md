---
status: accepted
implementation: implemented
---

# Use one numerical tolerance only for CPT validity

Set the generated-world CPT validity tolerance to

\[
\tau=10^{-12}.
\]

CPT finalization follows one rule:

1. reject nonfinite entries;
2. reject an entry below \(-\tau\);
3. replace an entry in \([-\tau,0)\) by zero;
4. reject a row whose sum differs from one by more than \(\tau\);
5. renormalize every accepted row to sum to one.

The finalized CPT is stored in the WorldSpec and is the sole probability law used by both interactive samples and task-truth computation.

This tolerance repairs only floating-point roundoff at the probability-law boundary. It is not used to round terminal predictions, identify two answers, threshold causal effects, discretize continuous rewards, or filter sampled worlds.

Generated CPT rows are finalized with this rule before entering the runtime or truth owners.
