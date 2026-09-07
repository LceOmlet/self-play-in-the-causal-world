Kernel compilation integration, 2026-09-07

The production working tree now includes three semantics-preserving changes:

- Ordinary inference prunes nonancestors in the mutilated intervention graph, substitutes fixed intervention values, and contracts multiplication with summation without materializing the full product.
- The two-mediator path checks a transport-polytope dimension lower bound before enumerating vertices that necessarily exceed its existing 4,096-combination guard. It retains the existing fallback and nonpositive-CPT handling.
- `_SparseResponseModel._compile_shared_boundary` marginalizes shared ancestors that are not parents of any affected mechanism, including the terminal outcome. It retains the same response variables and replaces the numeric shared distribution with normalized chain conditionals of its boundary joint law. It only selects this representation when the boundary joint has no more cells than the original shared CPT inputs. This is an input-table guard, not a general intermediate-memory bound.

The boundary identity is `sum_s P(s) F(s_B,q) = sum_b P(b) F(b,q)`. No response mechanism is optimized independently across strata that originally shared it. Correlation between boundary nodes and zero-mass contexts have dedicated regression coverage.

The final production validation passed 90 tests and 22,183 subtests. Ruff passed for both changed source modules and the two new test files. Existing independent rational-reference coverage includes 11,968 ordinary inference queries. The new shared-chain counterfactual reference is exactly `[30/103,65/103]`.

The 74 frozen counterfactual candidates were replayed with their original worlds, CPTs, queries, 5-second endpoint limits, approximately 11-second parent limits, and 8 GiB address-space limits. The baseline and the first two changes completed 50 cases; the final boundary compilation completed 51. All previous 50 remained accepted, and each endpoint change was within the sum of reported endpoint errors plus `1e-8`. This development set is not a population success-rate estimate.

Newly accepted case 46 reduced the upper model from 1,329 to 129 variables and from 1,249 to 49 auxiliaries at construction. It completed in approximately 0.261 seconds excluding process startup, returning `[0,0.2681157088716053]`. An independent formulation enumerated all 212 mediator transport vertices and solved 1,272 full-response LPs, agreeing within floating-point precision. The solver's `exact` category is not a rational-arithmetic proof for this floating-point world.

Cases 42, 50, and 61 have smaller models but remain uncertified. A further local shared-leaf contraction candidate was not integrated: it produced no additional completed cases or reduction of the critical scopes in the six construction failures.

Remaining kernel work has two concrete references:

- Case 43's first symbolic elimination materializes 360,000 output cells and evaluates 9 summands per cell. Its two removable shared ancestors are outside this scope. This is a cost of the current dense representation, not a lower bound for all algorithms.
- In the three-node chain `X -> M -> Y`, let both treatment-conditioned M distributions be uniform over three states, and let binary Y be fair in every M context. The sharp conditional transition interval is exactly `[0,2/3]`; the rational response-vertex reference confirms it. A deterministic Y response partitions M into k zero states and m-k one states, so the joint crossing probability is at most `min(k,m-k)/m <= floor(m/2)/m`. A transposition coupling and complementary Y responses attain this bound. The production model has only 31 variables and 27 constraints but still fails global certification in 5 seconds after pricing closes. Independently optimizing pairwise Frechet tables would incorrectly permit conditional upper bound 1. Future formulations must retain joint response realizability.

A strong-reversal family was also derived and evaluated through the existing ET-V2 kernel. With fair binary Z, `P(X=1|Z=0)=epsilon`, `P(X=1|Z=1)=1-epsilon`, and `logit P(Y=1|X,Z)=beta(X-1/2)-gamma(Z-1/2)`, define `u=sigmoid((gamma+beta)/2)` and `v=sigmoid((gamma-beta)/2)`. For `gamma>beta>0`, reversal with regret above tau requires and suffices within this family that `epsilon < (2v-1)/(2(u+v-1))` and `u-v > tau`. The example `(epsilon,beta,gamma)=(0.05,1,4)` lies inside the ET-V2 strength limits and has causal regret 0.10656734. No generator change has been made from this construction; preserving a desired conditional sampling law also requires handling its density and slice masses.

Training remains stopped. There are still 23 uncertified frozen candidates and no verified task-wide statistical budget calibration. Official DAPO provenance passed for 419 verl and 8 verl-recipe files and their runtime imports, with the existing two declared tool-termination patches; DAPO algorithm sources retain their upstream fingerprints. These checks do not establish arbitrary-environment training correctness.

Authoritative evidence is under `/home/chen/runs/kernel-integration-20260907`; original frozen inputs and replay logs are under `/home/chen/runs/kernel-root-cause-20260907`. See `acceptance-summary.json`, `integrated-kernel.patch`, `case46-independent-certificate.json`, `large-message-scopes.json`, `global-response-gap-proof.json`, `strong-reversal-proof.json`, `official-source-verification-final.json`, and the final production test log. The complete Chinese report is `kernel-acceptance-report.md` in the integration evidence directory.

Integrated SHA256 fingerprints:

- `query_truth.py`: `e47fb6b8c85b64ce4af830a8fba890cb05739eab5725363be56ff5cba41989cf`
- `counterfactual_solver.py`: `c7a4100f07ab0cc4a2f40a01bfafeef6c2ac200a8b762c6756345b417a34b2f2`

No git commit is implied by this integration record; the changes are in the project working tree.
