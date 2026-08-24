---
status: accepted
implementation: implemented
---

# Use float64 for generated worlds and retain exact rationals only as references

The main CPT-World sampler produces continuous base distributions, joint-effect directions, and effect strengths. Generated CPT rows, probability inference, intervention laws, and interactive sampling therefore use `float64` with explicit validation tolerances.

The main sampler does not rationalize generated probabilities and does not expose a bounded-denominator precision parameter. Rationalization would alter the declared continuous world distribution, collapse distinct small effects, and introduce an additional parameter without causal meaning.

Episode reproducibility is obtained by storing the fully realized WorldSpec. A random seed alone is not the persistent identity of a generated world.

Exact `Fraction` worlds remain useful as fixed fixtures and reference tests. They may verify analytic examples and compare algorithms on a shared exact input, but they do not define the numeric representation of newly sampled worlds.

Algorithmic acceleration and numeric representation remain separate concerns. Variable elimination and ancestral sampling reduce the number of probability operations; `float64` avoids the growth of exact-rational numerators and denominators within those operations.

The generated-world path now stores and propagates float64 CPT rows. Fixed fixtures and reference tests may still supply `Fraction` rows through the same WorldSpec interface.
