# Aggregate unresolved-endpoint evidence

The frozen distribution cohort contains 23 closed and 7 unresolved instances.
No sample identity, variable label, topology name, CPT fingerprint, or
per-instance outcome was emitted.

## Structural concentration

- 5/7 unresolved instances contain a convergence node with at least two
  treatment-affected parents; only 3/23 closed instances do.
- 6/7 unresolved instances contain three or four affected mechanisms; 22/23
  closed instances contain one or two.
- The failing response blocks have only 8 or 16 local contexts. The dominant
  scale is therefore not a single enormous response table.

## Certification stage

- Endpoint owner: three terminal lower, one terminal upper, one joint lower,
  and two joint upper failures.
- All seven failures terminate by the unchanged endpoint time allowance.
- Two failures finish pricing but do not close the global master.
- Four failures execute more than eight pricing rounds; three failures do not
  invoke a dynamic pricing MAP at all.

## Formulation scale and gap

- Five failures contain more than 500 variable-elimination auxiliaries and more
  than 5,000 master variables.
- Four failures retain a normalized primal-dual gap greater than 0.5; one is at
  most 0.5, one at most 0.1, and one at most 0.01.
- Pricing backends among failures are two min-sum-only, two SCIP fallback, and
  three with no pricing MAP.

## Supported hypothesis

The evidence supports testing a stronger exact message bound before any
pricing micro-optimization. The current variable-elimination upper bound sums
entrywise kernel bounds independently and ignores that each kernel matrix must
satisfy its fixed row and column marginals. For a fixed downstream upper-cost
matrix `u`, the exact local envelope is the transportation optimum

`max_K sum_ab K_ab u_ab`

subject to the two CPT marginals. This leaves every compatible response
mechanism feasible and only tightens the master relaxation. A shadow pass must
first establish how frequently and how strongly this envelope improves the
existing bound.
