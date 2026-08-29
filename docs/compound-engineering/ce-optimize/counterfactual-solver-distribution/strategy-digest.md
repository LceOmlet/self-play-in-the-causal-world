# Strategy digest

## Current best

The retained projected-message and root-pricing certificate reaches 24/30 on
all three frozen-distribution repeats, up from the 23/30 baseline. Formal
regression remains 20/30 on all three repeats.

## Tested

- Exact direct-only shared-parent Frechet mixture: mathematically exact and all
  gates passed, but aggregate closure stayed at baseline after confirmation.
  The approach accelerates cases that already close and was reverted.
- Pre-model constant-zero direct lower endpoint: the existing strict
  certificate was moved ahead of model construction, but primary unresolved
  stayed 7/30 and the formal sample regressed. The certificate does not cover
  the closure-limiting endpoints and the change was reverted.
- Exact priced routing for oversized explicit layered objectives: semantic
  parity passed, but primary unresolved stayed 7/30 in both measurements;
  formal evidence was within the paired noise threshold. The change was
  reverted.
- Globally affine/one-sided terminal endpoint certificates: exact and fully
  gated, but both cohorts stayed at baseline closure. The certificate coverage
  lies among already-closed instances and the change was reverted.

## Evidence frontier

- Exact one-world projection tightened 62.3 percent of profiled message upper
  bounds and gave 11.3 percent a positive lower bound, but alone stayed at
  23/30. A completed exact root pricing round then supplied a rigorous
  full-master omitted-column correction and closed one additional endpoint as
  epsilon-sharp.
- Six frozen instances remain unresolved. Further work requires new aggregate
  evidence for a complete mathematical class; the retained implementation
  provides no justification for seed-specific follow-up.
- Pricing callback micro-optimization remains low priority because aggregate
  callback time is negligible relative to SCIP optimization and model setup.
- Every next trigger must remain graph-isomorphism and renaming invariant; no
  finite failed-case information may enter implementation or selection.
