# CPT-World paper evidence map

This file is a drafting aid. It is not part of the submitted paper.

## Central claim

CPT-World turns finite causal models into interactive, hidden-world episodes in
which a language model chooses experiments, receives selected joint counts, and
returns a structured causal answer checked by an exact oracle.

## Claim-to-evidence map

| Paper claim | Evidence | Allowed wording |
| --- | --- | --- |
| The environment hides graph, CPT entries, internal names, truth, and scorer while exposing opaque variables, legal actions, readable variables, and the terminal schema. | `README.md`; `src/cpt_world/rendering.py`; prompt snapshots in `scripts/demo_worldspec_runtime.py`; renderer tests. | “The prompt exposes the experimental interface and hides the causal mechanism.” |
| A single interaction owner supports passive observation and batched hard interventions with selected-variable feedback. | `src/cpt_world/world_runtime.py`; `src/cpt_world/query_truth.py`; `tests/test_world_runtime.py`. | “Each action returns sparse joint counts over the requested readout.” |
| Five causal task families share one WorldSpec sampler, renderer, runtime, and exact truth layer. | `README.md`; `docs/query-task-registry.md`; `src/cpt_world/world_space.py`; `src/cpt_world/query_truth.py`; `src/cpt_world/task_scoring.py`. | “The same hidden world supports effect, counterfactual, decision, adjustment, and mediation queries.” |
| ATE and counterfactual query endpoints are read-only in the generic sampler; the deployment variable is read-only during experimental-decision episodes. | `src/cpt_world/world_space.py`; `src/cpt_world/rendering.py`; tests covering default manipulability and strict parsing. | “Query and action are separated at the interaction surface.” |
| Counterfactual labels are sharp compatible intervals or compatible scalar values. | `counterfactual_transition_bounds` in `src/cpt_world/query_truth.py`; counterfactual parser/scorer; exact bounds tests. | “The verifier scores the set of cross-world couplings compatible with the CPT world.” |
| K controls the legal intervention subset and M controls maximum readout width; both are sampled after family-level answerability. | `src/cpt_world/world_space.py`; `docs/world-space-sampler-contract.md`; tests for randomized interaction surfaces. | “K and M vary evidence access while the underlying query truth stays fixed.” |
| Exact selected marginals preserve the full-joint Fraction semantics. | Variable-elimination implementation in `query_truth.py`; 24-world exact differential tests. | “Variable elimination matches full-joint enumeration exactly.” |
| Ancestral batch sampling preserves hard-do semantics and action-keyed reproducibility. | `world_runtime.py`; `world.py`; 12-DAG exact mass tests; split/interleave/rename tests. | “The same action-index pair receives the same random draw under batching, projection, and surface renaming.” |
| The accelerated engine scales beyond the old joint-state path. | `scripts/benchmark_worldspec_acceleration.py`; frozen README table. | “At 15 binary nodes, selected exact inference and 64-sample feedback are 338.4x and 102.5x faster on the repository benchmark.” |
| End-to-end prompt, experiment, feedback, terminal parsing, and exact scoring execute for all five task types. | `scripts/demo_worldspec_runtime.py`; runtime tests; `scripts/verify_task_family_seed_math.py`. | “All five modes complete the same prompt-to-verdict loop.” |
| The repository regression suite passes. | Local run at paper preparation time. | “All 69 tests pass; Ruff lint and formatting checks pass.” |

## External evidence

| Related claim | Primary source | Verified support |
| --- | --- | --- |
| CLadder is a static natural-language causal inference benchmark. | Jin et al., NeurIPS 2023, official proceedings. | 10K questions from causal graphs and associational/interventional/counterfactual queries; symbolic questions and oracle answers are translated to natural language. |
| CLadder provides the model specification and sufficient data in the prompt. | Jin et al., NeurIPS 2023, Sec. 3 and Appendix C. | The formal pipeline constructs identifiable graph/query/data triples and verbalizes the causal model and data. |
| Active Bayesian causal inference selects experiments for a target query. | Toth et al., NeurIPS 2022, official proceedings. | Sequential experiment design jointly updates causal-model and query uncertainty. |
| Multi-target experimental design optimizes batches of intervention target-value pairs. | Tigas et al., ICML 2023, PMLR. | Differentiable batch design over multiple intervention target-state pairs. |
| Indirect experiments can target a causal query when treatment cannot be manipulated. | Dern et al., NeurIPS 2024; Ailer et al., ICML 2023. | Both papers explicitly study indirect or instrumental experiments for causal-effect learning. |
| Scientific-discovery agents benefit from executable experiment environments. | Jansen et al., NeurIPS 2024 Datasets and Benchmarks. | DiscoveryWorld requires hypothesis formation, experiment design, result analysis, and action. |
| Causal effects and counterfactuals can be partially identified by sharp bounds. | Balke and Pearl, JASA 1997; Pearl, 2009. | Compatible causal models induce tight nonparametric bounds; SCM semantics define cross-world quantities. |

## Prohibited claims

- No claim of state-of-the-art LLM performance.
- No claim that current K/M sampling yields a final uniform difficulty mixture.
- No claim that the default world-size support is 4--15; the current default is
  2--4 and the exact engine is benchmarked through 15 binary nodes.
- No claim that the repository implements a self-play optimization algorithm.
- No claim that compatible intervals identify a unique individual-level
  counterfactual when their endpoints differ.

