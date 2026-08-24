---
status: accepted
implementation: planned
---

# Treat every generated CPT-World as causally sufficient

Every common cause is represented by a node in the WorldSpec DAG. Graph-external hidden confounding is not admitted, and the exogenous mechanisms of distinct nodes are independent.

Consequently, the DAG and CPT rows determine every single-world hard-intervention distribution through mechanism replacement and truncated factorization. This is the semantic basis already used by the ATE and experimental-decision truth owners.

Causal sufficiency does not select a parametric or deterministic SCM. For each node, the joint coupling of its responses across different parent configurations remains unrestricted subject to its CPT row marginals. Counterfactual bounds optimize over those compatible within-node response couplings instead of choosing one.

The hidden-world protocol hides the graph and CPT from the model. It does not mean that the verifier's WorldSpec omits causal variables.
