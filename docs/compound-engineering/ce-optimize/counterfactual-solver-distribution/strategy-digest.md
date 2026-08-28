# Strategy digest

## Current best

Baseline remains best: distribution closure 23/30 and formal regression
closure 20/30 under the frozen measurement contract.

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

- Four planned experiments are complete. None reduced primary unresolved below
  7/30 or raised the stable formal baseline above 20/30. Direct analytic paths,
  the existing zero certificate, priced layered routing, and terminal affine
  certificates all primarily affect already-closed instances.
- The next evidence phase must classify unresolved endpoints by aggregate
  mathematical obstruction before proposing code: affected-mechanism count,
  context width, upstream transport vertex product, and which exact owner or
  pricing certificate fails. No further solver change is justified by the
  current aggregate profile alone.
- Pricing callback micro-optimization remains low priority because aggregate
  callback time is negligible relative to SCIP optimization and model setup.
- Every next trigger must remain graph-isomorphism and renaming invariant; no
  finite failed-case information may enter implementation or selection.
