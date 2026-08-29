---
status: accepted
implementation: in-progress
requires: 0016-assume-causal-sufficiency
---

# Bound counterfactuals over complete CPT-World mechanism completions

For a counterfactual target \(q\), define the legal set from every finite nonparametric mechanism completion jointly compatible with the complete public causal semantics of the sealed CPT-World:

\[
\mathcal C(W)=\{M:M\text{ respects the DAG, every CPT row, consistency, mechanism replacement, and all other public assumptions}\}.
\]

The terminal truth is

\[
L(W)=\min_{M\in\mathcal C(W)}q(M),
\qquad
U(W)=\max_{M\in\mathcal C(W)}q(M).
\]

The current task target is the transition event

\[
q(M)=P_M\!\left(Y_{x_{\rm cmp}}=y^\star,
Y_{x_{\rm ref}}\ne y^\star\right).
\]

The verifier does not sample a hidden SCM, choose a functional form, impose a prior over completions, or expose response functions to the model. Finite response types are universal internal optimization coordinates rather than a generated mechanism family.

Let \(\mathcal C_{\mathrm{endpoint}}(W)\) contain every coupling of only the two endpoint interventional event marginals. Because every full-world-compatible completion induces such a coupling,

\[
\mathcal C(W)\subseteq\mathcal C_{\mathrm{endpoint}}(W),
\]

and therefore the full-world sharp interval is weakly tighter:

\[
L(W)\ge L_{\mathrm{endpoint}}(W),
\qquad
U(W)\le U_{\mathrm{endpoint}}(W).
\]

The endpoint Fréchet interval remains a rigorous outer diagnostic. It is a task
truth only where a separate argument proves it sharp for the complete world.

An implementation may change coordinates, decompose the twin world, and
generate response columns on demand. It may not change \(\mathcal C(W)\),
select one completion, or add a functional family. The production truth owner
returns either the exact pair \([L(W),U(W)]\), or a safe outer pair whose lower
and upper endpoints are each within `0.001` of the corresponding sharp endpoint
on the final conditional-probability scale. The latter is recorded as
`epsilon_sharp` together with the certified endpoint error. A larger gap or
incomplete response pricing fails closed. The Fréchet interval remains a
separate diagnostic outer bound.
