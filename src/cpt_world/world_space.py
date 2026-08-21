"""Sampler for the CPT-World seed subspace.

The sampler declares an explicit probability model:

- node count: uniform over ``node_counts``
- domain size: uniform over ``{2, ..., max_domain_size}``
- topological order: uniform over all orders
- edge subset: uniform over all forward-edge subsets of that order
- root prior: uniform over ``(0, 1)``, exactly rationalized
- edge effect: uniform over ``(-1/2, 1/2)``, exactly rationalized

``seed`` is used only as a reproducible RNG seed; it never encodes a grid,
array index, or coordinate expansion.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from typing import Any

from .registry import (
    COUNTERFACTUAL_ANSWER_MODES,
    HIDING_MODES,
    QUERY_TYPES,
    TASK_HEADS,
    counterfactual_answer_mode,
    query_task_compatible,
)

_LABEL_POOL = "DEFGHIJKLMNOPQRSTUVW"

_DEFAULT_NODE_COUNTS = (2, 3, 4)
_MAX_GRAMMAR_NODES = 6


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """A finite discrete DAG world sampled from the declared grammar."""

    family: str
    topology: str
    variables: tuple[str, ...]
    domains: tuple[int, ...]
    state_names: tuple[tuple[str, ...], ...]
    edges: tuple[tuple[int, int], ...]
    parents: Mapping[int, tuple[int, ...]]
    cpt: Mapping[int, tuple[tuple[Fraction, ...], ...]]

    def child_count(self, node: int) -> int:
        return sum(1 for parent, child in self.edges if parent == node)

    def parent_count(self, node: int) -> int:
        return len(self.parents.get(node, ()))

    def path_exists(self, source: int, target: int) -> bool:
        path = _shortest_path_nodes(self, source, target)
        return path is not None and len(path) >= 2

    def shortest_path_nodes(self, source: int, target: int) -> tuple[int, ...] | None:
        path = _shortest_path_nodes(self, source, target)
        if path is None or len(path) < 2:
            return None
        return path

    def has_indirect_path(self, source: int, target: int) -> bool:
        if source == target:
            return False
        adjacency: dict[int, list[int]] = {node: [] for node in range(len(self.variables))}
        for parent, child in self.edges:
            adjacency[parent].append(child)
        stack: list[tuple[int, int]] = [(source, 0)]
        while stack:
            node, depth = stack.pop()
            for child in adjacency[node]:
                if child == target and depth + 1 >= 2:
                    return True
                stack.append((child, depth + 1))
        return False


@dataclass(frozen=True, slots=True)
class WorldGrammar:
    """Declared sampling distribution for the CPT-World seed subspace.

    The fields describe the probability model, not a fixed grid:

    - ``node_counts``: support of the uniform node-count distribution
    - ``max_domain_size``: upper bound for the uniform domain-size distribution
    - ``edge_effect_range``: open interval for the signed edge-effect uniform
    - ``root_prior_range``: open interval for the root-prior uniform
    - ``rational_denominator_bound``: exact-rational precision bound
    - ``max_stability_attempts``: computational bound for fail-closed stability
    """

    node_counts: tuple[int, ...] = _DEFAULT_NODE_COUNTS
    max_domain_size: int = 5
    edge_effect_range: tuple[Fraction, Fraction] = (Fraction(-1, 2), Fraction(1, 2))
    root_prior_range: tuple[Fraction, Fraction] = (Fraction(0), Fraction(1))
    rational_denominator_bound: int = 1000
    max_stability_attempts: int = 10000

    def __post_init__(self) -> None:
        if not self.node_counts:
            raise ValueError("node_counts must not be empty")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 2
            for count in self.node_counts
        ):
            raise ValueError("node_counts must contain integers >= 2")
        if any(count > _MAX_GRAMMAR_NODES for count in self.node_counts):
            raise ValueError(f"node_counts must not exceed {_MAX_GRAMMAR_NODES}")
        if len(set(self.node_counts)) != len(self.node_counts):
            raise ValueError("node_counts must not contain duplicates")
        if (
            isinstance(self.max_domain_size, bool)
            or not isinstance(self.max_domain_size, int)
            or self.max_domain_size < 2
        ):
            raise ValueError("max_domain_size must be an integer >= 2")
        if len(self.edge_effect_range) != 2:
            raise ValueError("edge_effect_range must contain two endpoints")
        if self.edge_effect_range[0] >= self.edge_effect_range[1]:
            raise ValueError("edge_effect_range must be an increasing interval")
        if not -Fraction(1, 2) <= self.edge_effect_range[0]:
            raise ValueError("edge_effect_range lower bound must be >= -1/2")
        if self.edge_effect_range[1] > Fraction(1, 2):
            raise ValueError("edge_effect_range upper bound must be <= 1/2")
        if len(self.root_prior_range) != 2:
            raise ValueError("root_prior_range must contain two endpoints")
        if not 0 <= self.root_prior_range[0] < self.root_prior_range[1] <= 1:
            raise ValueError("root_prior_range must lie inside [0, 1]")
        if (
            isinstance(self.rational_denominator_bound, bool)
            or not isinstance(self.rational_denominator_bound, int)
            or self.rational_denominator_bound <= 0
        ):
            raise ValueError("rational_denominator_bound must be a positive integer")
        if (
            isinstance(self.max_stability_attempts, bool)
            or not isinstance(self.max_stability_attempts, int)
            or self.max_stability_attempts <= 0
        ):
            raise ValueError("max_stability_attempts must be a positive integer")


def _root_cpt_row(domain_size: int, root_prior: Fraction) -> tuple[Fraction, ...]:
    """Canonical root distribution.

    State 1 receives ``root_prior``; the remaining mass is uniform over every
    other state. For binary domains this reduces to the original root prior.
    """

    other_mass = (1 - root_prior) / (domain_size - 1)
    row = [other_mass] * domain_size
    row[1] = root_prior
    return tuple(row)


def _child_cpt_row(
    parent_values: tuple[int, ...],
    domain_size: int,
    parent_effects: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    """Signed multiplicative finite-domain child mechanism.

    Each parent ``p`` with signed effect ``e_p`` prefers child state
    ``parent_state mod domain_size``. Its multiplicative odds are

    ``q_p = (1/2 + e_p) / (1/2 - e_p)``.

    Raw child-state weight is the product of ``q_p`` for every parent that
    prefers that state, normalized to sum to one. For one binary parent this
    reduces to ``P(preferred)=1/2+e`` and ``P(other)=1/2-e``.
    """

    weights = [Fraction(1)] * domain_size
    for parent_value, effect in zip(parent_values, parent_effects, strict=True):
        preferred = parent_value % domain_size
        odds = (Fraction(1, 2) + effect) / (Fraction(1, 2) - effect)
        weights[preferred] *= odds
    total = sum(weights, Fraction(0))
    return tuple(weight / total for weight in weights)


def _shortest_path_nodes(world: WorldSpec, source: int, target: int) -> tuple[int, ...] | None:
    """Return one shortest directed path from source to target, or None."""

    if source == target:
        return (source,)
    if not 0 <= source < len(world.variables) or not 0 <= target < len(world.variables):
        return None
    adjacency: dict[int, list[int]] = {node: [] for node in range(len(world.variables))}
    for parent, child in world.edges:
        adjacency[parent].append(child)
    predecessor: dict[int, int] = {}
    queue = [source]
    seen = {source}
    while queue:
        node = queue.pop(0)
        if node == target:
            path = [node]
            while path[-1] != source:
                path.append(predecessor[path[-1]])
            return tuple(reversed(path))
        for child in adjacency[node]:
            if child not in seen:
                seen.add(child)
                predecessor[child] = node
                queue.append(child)
    return None


def _opaque_labels(n: int, seed_id: str) -> tuple[str, ...]:
    labels: list[str] = []
    for index in range(n):
        while True:
            digest = hashlib.sha256(
                f"cpt-world-space-labels-v1\0{seed_id}\0{index}\0{len(labels)}".encode()
            ).digest()
            token = "".join(_LABEL_POOL[byte % len(_LABEL_POOL)] for byte in digest[:3])
            if token not in labels:
                labels.append(token)
                break
    return tuple(labels)


def bif_state_names(text: str) -> dict[str, tuple[str, ...]]:
    """Return variable declaration state order for a bnlearn BIF document."""

    names: dict[str, tuple[str, ...]] = {}
    for name, block in re.findall(r"variable\s+(\w+)\s*\{(.*?)\n\}", text, re.S):
        if name in names:
            raise ValueError(f"duplicate variable declaration {name} in BIF text")
        state_blocks = re.findall(r"\{([^{}]*)\}", block)
        if not state_blocks:
            raise ValueError(f"variable {name} has no state block in BIF text")
        states = tuple(part.strip() for part in state_blocks[-1].split(",") if part.strip())
        if not states or len(set(states)) != len(states):
            raise ValueError(f"variable {name} has empty or duplicate states in BIF text")
        names[name] = states
    if not names:
        raise ValueError("no variable declarations found in BIF text")
    return names


def world_state_names(world_source: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Resolve internal state order from a seed's world source.

    This is renderer-side bookkeeping only; it never becomes model-visible.
    """

    if not isinstance(world_source, Mapping):
        raise ValueError("world_source must be a mapping")
    if isinstance(world_source.get("state_names"), Mapping):
        return {str(name): tuple(states) for name, states in world_source["state_names"].items()}
    source_type = world_source.get("type")
    if source_type == "bnlearn_bif":
        bif = world_source.get("bif")
        if isinstance(bif, str):
            return bif_state_names(bif)
        file_path = world_source.get("file")
        if not isinstance(file_path, str):
            raise ValueError("bnlearn_bif world source has neither bif text nor file")
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return bif_state_names(path.read_text(encoding="utf-8"))
    if source_type == "cladder_meta_model":
        structure = str(world_source.get("structure", ""))
        names: list[str] = []
        for token in structure.replace(" ", "").split(","):
            for name in token.split("->"):
                if name and name not in names:
                    names.append(name)
        if not names:
            raise ValueError("cladder_meta_model world source has no structure")
        return {name: ("0", "1") for name in names}
    if source_type in {"sampled_motif", "sampled_dag"}:
        variables = world_source.get("variables", ())
        domains = world_source.get("domains", ())
        if not isinstance(variables, (list, tuple)) or not isinstance(domains, (list, tuple)):
            raise ValueError("sampled_motif world source has no variables/domains")
        return {
            str(name): tuple(f"state_{state}" for state in range(int(size)))
            for name, size in zip(variables, domains, strict=True)
        }
    raise ValueError(f"unsupported world_source type {source_type}")


@dataclass(frozen=True, slots=True)
class _SampledStructure:
    node_count: int
    domains: tuple[int, ...]
    variables: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    parents: tuple[tuple[int, ...], ...]
    topology: str


def _sample_structure(grammar: WorldGrammar, rng: random.Random) -> _SampledStructure:
    node_count = rng.choice(grammar.node_counts)
    domains = tuple(rng.randint(2, grammar.max_domain_size) for _ in range(node_count))
    order = list(range(node_count))
    rng.shuffle(order)
    forward_pairs = [
        (order[left], order[right])
        for left in range(node_count)
        for right in range(left + 1, node_count)
    ]
    if forward_pairs:
        mask = rng.getrandbits(len(forward_pairs))
        edges = tuple(pair for bit, pair in enumerate(forward_pairs) if mask & (1 << bit))
    else:
        edges = ()
    parents_list: list[tuple[int, ...]] = [() for _ in range(node_count)]
    for parent, child in edges:
        parents_list[child] = parents_list[child] + (parent,)
    variables = tuple(f"V{index}" for index in range(node_count))
    topology = f"dag-n{node_count}-d{'-'.join(str(size) for size in domains)}-e{len(edges)}"
    return _SampledStructure(
        node_count=node_count,
        domains=domains,
        variables=variables,
        edges=edges,
        parents=tuple(parents_list),
        topology=topology,
    )


def _sample_edge_effect(grammar: WorldGrammar, rng: random.Random) -> Fraction:
    low, high = grammar.edge_effect_range
    while True:
        value = low + (high - low) * rng.random()
        effect = Fraction.from_float(float(value)).limit_denominator(
            grammar.rational_denominator_bound
        )
        if effect != 0 and low < effect < high:
            return effect


def _sample_root_prior(grammar: WorldGrammar, rng: random.Random) -> Fraction:
    low, high = grammar.root_prior_range
    while True:
        value = low + (high - low) * rng.random()
        prior = Fraction.from_float(float(value)).limit_denominator(
            grammar.rational_denominator_bound
        )
        if low < prior < high:
            return prior


def _sample_parameter_maps(
    grammar: WorldGrammar, structure: _SampledStructure, rng: random.Random
) -> tuple[Mapping[int, Fraction], Mapping[tuple[int, int], Fraction]]:
    root_prior_map = {
        node: _sample_root_prior(grammar, rng)
        for node in range(structure.node_count)
        if not structure.parents[node]
    }
    edge_effect_map = {edge: _sample_edge_effect(grammar, rng) for edge in structure.edges}
    return root_prior_map, edge_effect_map


def _build_world(
    grammar: WorldGrammar,
    structure: _SampledStructure,
    root_prior_map: Mapping[int, Fraction],
    edge_effect_map: Mapping[tuple[int, int], Fraction],
) -> WorldSpec:
    del grammar  # structure and parameter maps already carry the CPT semantics
    cpt: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
    for node in range(structure.node_count):
        domain_size = structure.domains[node]
        node_parents = structure.parents[node]
        if not node_parents:
            cpt[node] = (_root_cpt_row(domain_size, root_prior_map[node]),)
        else:
            parent_effects = tuple(edge_effect_map[(parent, node)] for parent in node_parents)
            rows: list[tuple[Fraction, ...]] = []
            parent_ranges = (range(structure.domains[parent]) for parent in node_parents)
            for parent_values in product(*parent_ranges):
                rows.append(_child_cpt_row(parent_values, domain_size, parent_effects))
            cpt[node] = tuple(rows)
    return WorldSpec(
        family="sampled_dag",
        topology=structure.topology,
        variables=structure.variables,
        domains=structure.domains,
        state_names=tuple(tuple(str(i) for i in range(size)) for size in structure.domains),
        edges=structure.edges,
        parents={node: parents for node, parents in enumerate(structure.parents)},
        cpt=cpt,
    )


def sample_world(grammar: WorldGrammar, seed: int) -> WorldSpec:
    """Sample one legal finite-domain DAG from the declared distribution.

    ``seed`` is used only to seed ``random.Random``. Structure and CPT are
    drawn from the distributions declared by ``WorldGrammar``.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(grammar, WorldGrammar):
        raise TypeError("grammar must be a WorldGrammar")
    rng = random.Random(seed)
    structure = _sample_structure(grammar, rng)
    root_prior_map, edge_effect_map = _sample_parameter_maps(grammar, structure, rng)
    return _build_world(grammar, structure, root_prior_map, edge_effect_map)


def _task_target_metrics(
    world: WorldSpec, query_type: str, anchors: Mapping[str, int | str]
) -> Mapping[str, Fraction]:
    """Exact task-relevant targets; no goodness thresholds are applied."""

    from .query_truth import (
        ate_effect,
        best_intervention_truth,
        interventional_probability,
    )

    if query_type in {"backadj_minimal_sets", "mediator_set"}:
        raise NotImplementedError(
            f"revealing-quality metrics for {query_type} are not designed yet"
        )

    outcome = int(anchors["outcome"])
    baseline = interventional_probability(world, {}, outcome, 1)
    if query_type in {"ate", "counterfactual_transition_bounds"}:
        return {
            "target": ate_effect(world, int(anchors["treatment"]), outcome, outcome_state=1),
            "baseline": baseline,
        }
    if query_type == "best_intervention":
        objective = str(anchors.get("objective", "minimize"))
        _, _, best_probability = best_intervention_truth(
            world,
            outcome,
            objective,
            int(anchors["decision_target"]),
            outcome_state=1,
        )
        decision_target = int(anchors["decision_target"])
        probabilities = [
            interventional_probability(world, {decision_target: state}, outcome, 1)
            for state in range(world.domains[decision_target])
        ]
        if objective == "minimize":
            others = [p for p in probabilities if p > best_probability]
            gap = min(others) - best_probability if others else Fraction(0)
        else:
            others = [p for p in probabilities if p < best_probability]
            gap = best_probability - max(others) if others else Fraction(0)
        return {"target": gap, "baseline": baseline}
    raise ValueError(f"unsupported query type {query_type}")


def _numerically_and_causally_stable(metrics: Mapping[str, Fraction]) -> bool:
    """Binary stability gate with no magnitude thresholds.

    Numerical stability: every probability strictly inside (0, 1).
    Causal stability: the task target is nonzero (effect or decision gap).
    """

    probabilities = [
        value
        for key, value in metrics.items()
        if key in {"baseline", "condition_mass_treated", "condition_mass_baseline"}
    ]
    if any(value <= 0 or value >= 1 for value in probabilities):
        return False
    return metrics["target"] != 0


def profile_task_targets(
    grammar: WorldGrammar,
    seed: int,
    query_type: str,
    anchors: Mapping[str, int | str],
    *,
    sample_count: int = 200,
) -> Mapping[str, Any]:
    """Sample and profile the task-target distribution, without filtering.

    ``seed`` is an RNG seed. One structure is drawn, then ``sample_count``
    CPT instances are drawn from the declared parameter distributions. The
    returned profile reports target signs, stability counts, and per-edge
    positive/negative effect counts.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = random.Random(seed)
    structure = _sample_structure(grammar, rng)
    targets: list[Fraction] = []
    stable_count = 0
    edge_sign_counts: dict[tuple[int, int], list[int]] = {edge: [0, 0] for edge in structure.edges}
    for _ in range(sample_count):
        root_prior_map, edge_effect_map = _sample_parameter_maps(grammar, structure, rng)
        for edge, effect in edge_effect_map.items():
            bucket = 0 if effect > 0 else 1
            edge_sign_counts[edge][bucket] += 1
        world = _build_world(grammar, structure, root_prior_map, edge_effect_map)
        metrics = _task_target_metrics(world, query_type, anchors)
        targets.append(metrics["target"])
        if _numerically_and_causally_stable(metrics):
            stable_count += 1
    return {
        "seed": seed,
        "query_type": query_type,
        "sample_count": sample_count,
        "stable_count": stable_count,
        "zero_count": sum(1 for target in targets if target == 0),
        "negative_count": sum(1 for target in targets if target < 0),
        "positive_count": sum(1 for target in targets if target > 0),
        "target_min": float(min(targets, default=Fraction(0))),
        "target_max": float(max(targets, default=Fraction(0))),
        "edge_sign_counts": {
            str(edge): {"positive": counts[0], "negative": counts[1]}
            for edge, counts in edge_sign_counts.items()
        },
        "max_edge_sign_imbalance": max(
            (abs(counts[0] - counts[1]) for counts in edge_sign_counts.values()),
            default=0,
        ),
    }


def sample_task_world(
    grammar: WorldGrammar,
    seed: int,
    query_type: str,
    anchors: Mapping[str, int | str],
) -> WorldSpec:
    """Sample one numerically and causally stable task world.

    A structure is drawn first, then CPT instances are drawn from the declared
    distributions until one passes the binary stability gate. This is
    acceptance-rejection over the declared parameter distribution, so the
    returned world is distributed as the declared distribution conditioned on
    stability. If no stable draw is found within the declared computational
    bound, the function fails closed.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(seed)
    structure = _sample_structure(grammar, rng)
    for _ in range(grammar.max_stability_attempts):
        root_prior_map, edge_effect_map = _sample_parameter_maps(grammar, structure, rng)
        world = _build_world(grammar, structure, root_prior_map, edge_effect_map)
        metrics = _task_target_metrics(world, query_type, anchors)
        if _numerically_and_causally_stable(metrics):
            return world
    raise ValueError(f"no numerically and causally stable CPT draw for {query_type} at seed {seed}")


def _surface_display_nodes(
    world: WorldSpec,
    seed: Mapping[str, Any],
    surface: object,
) -> tuple[int, ...]:
    from .rendering import (
        RenderedAteQuerySurface,
        RenderedDecisionQuerySurface,
        RenderedDiscoveryQuerySurface,
    )

    if not isinstance(
        surface,
        (RenderedAteQuerySurface, RenderedDecisionQuerySurface, RenderedDiscoveryQuerySurface),
    ):
        raise TypeError("surface must be a supported rendered query surface")
    visible_schema = seed.get("visible_schema")
    if not isinstance(visible_schema, Mapping):
        raise ValueError("seed is missing visible_schema")
    labels_value = visible_schema.get("variable_labels")
    if not isinstance(labels_value, Mapping):
        raise ValueError("visible_schema.variable_labels must be a mapping")
    visible_to_internal = {str(label): str(name) for name, label in labels_value.items()}
    try:
        display_nodes = tuple(
            world.variables.index(visible_to_internal[label]) for label in surface.labels
        )
    except (KeyError, ValueError) as error:
        raise ValueError("rendered labels do not align with the WorldSpec") from error
    if tuple(world.domains[node] for node in display_nodes) != surface.domains:
        raise ValueError("rendered domains do not align with the WorldSpec")
    return display_nodes


def _response_signature(
    world: WorldSpec,
    seed: Mapping[str, Any],
    surface: object,
) -> tuple[object, ...]:
    from .query_truth import worldspec_projected_interventional_distribution
    from .rendering import (
        RenderedAteQuerySurface,
        RenderedDecisionQuerySurface,
        RenderedDiscoveryQuerySurface,
    )

    if not isinstance(
        surface,
        (RenderedAteQuerySurface, RenderedDecisionQuerySurface, RenderedDiscoveryQuerySurface),
    ):
        raise TypeError("surface must be a supported rendered query surface")
    display_nodes = _surface_display_nodes(world, seed, surface)

    signature: list[object] = []
    observation_size = min(
        len(surface.readable),
        surface.measure_max if surface.measure_max is not None else len(surface.readable),
    )
    if observation_size:
        for measure_positions in combinations(surface.readable, observation_size):
            measure = tuple(display_nodes[position] for position in measure_positions)
            law = worldspec_projected_interventional_distribution(world, {}, measure)
            signature.append(
                (
                    "observe",
                    measure_positions,
                    tuple(probability for _, probability in law),
                )
            )
    for target_position in surface.legal_targets:
        target = display_nodes[target_position]
        readable_positions = tuple(
            position for position in surface.readable if position != target_position
        )
        if not readable_positions:
            raise ValueError("rendered target has no legal selected measure")
        measure_size = min(
            len(readable_positions),
            surface.measure_max if surface.measure_max is not None else len(readable_positions),
        )
        measure_position_sets = tuple(combinations(readable_positions, measure_size))
        for state in range(world.domains[target]):
            for measure_positions in measure_position_sets:
                measure = tuple(display_nodes[position] for position in measure_positions)
                law = worldspec_projected_interventional_distribution(
                    world, {target: state}, measure
                )
                signature.append(
                    (
                        target_position,
                        state,
                        measure_positions,
                        tuple(probability for _, probability in law),
                    )
                )
    return tuple(signature)


def _acceptable_answers(
    world: WorldSpec,
    seed: Mapping[str, Any],
    surface: object,
) -> frozenset[tuple[object, ...]]:
    from .query_truth import best_intervention_states, compute_query_truth
    from .rendering import (
        RenderedAteQuerySurface,
        RenderedDecisionQuerySurface,
        RenderedDiscoveryQuerySurface,
    )

    if isinstance(surface, RenderedAteQuerySurface):
        truth = compute_query_truth(world, seed)
        if surface.query_type == "ate":
            effect = truth.get("effect")
            if not isinstance(effect, Fraction):
                raise TypeError("ATE truth owner must return an exact Fraction")
            return frozenset({("effect", effect)})
        lower = truth.get("lower")
        upper = truth.get("upper")
        if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
            raise TypeError("counterfactual truth owner must return exact interval endpoints")
        return frozenset({("counterfactual_interval", lower, upper)})
    if isinstance(surface, RenderedDecisionQuerySurface):
        display_nodes = _surface_display_nodes(world, seed, surface)
        states, _ = best_intervention_states(
            world,
            display_nodes[surface.outcome],
            surface.objective,
            display_nodes[surface.decision_target],
            outcome_state=surface.outcome_state,
        )
        return frozenset(("decision", surface.decision_target, state) for state in states)
    if isinstance(surface, RenderedDiscoveryQuerySurface):
        truth = compute_query_truth(world, seed)
        display_nodes = _surface_display_nodes(world, seed, surface)
        display_position = {node: position for position, node in enumerate(display_nodes)}

        def position(name: object) -> int:
            return display_position[world.variables.index(str(name))]

        if surface.query_type == "backadj_minimal_sets":
            adjustment_sets = tuple(
                sorted(
                    tuple(sorted(position(name) for name in adjustment_set))
                    for adjustment_set in truth["adjustment_sets"]
                )
            )
            return frozenset({("backadj_minimal_sets", adjustment_sets)})
        mediators = tuple(sorted(position(name) for name in truth["mediators"]))
        order = tuple(
            sorted((position(source), position(target)) for source, target in truth["order"])
        )
        return frozenset({("mediator_set", mediators, order)})
    raise TypeError("surface must be a supported rendered query surface")


def _family_answerability_seed(
    world: WorldSpec,
    seed: Mapping[str, Any],
    query_type: str,
) -> Mapping[str, Any]:
    """Restore the full query-specific experiment surface before K/M sampling."""

    query = seed.get("query")
    visible_schema = seed.get("visible_schema")
    if not isinstance(query, Mapping) or not isinstance(visible_schema, Mapping):
        raise ValueError("candidate seed is missing query or visible_schema")
    labels = visible_schema.get("variable_labels")
    if not isinstance(labels, Mapping):
        raise ValueError("candidate seed is missing variable_labels")
    visible_to_internal = {str(visible): str(internal) for internal, visible in labels.items()}
    anchors: dict[str, int | str] = {}
    for anchor in QUERY_TYPES[query_type]["anchors"]:
        value = query.get(anchor)
        if value is None:
            raise ValueError(f"candidate query is missing anchor {anchor}")
        internal = visible_to_internal.get(str(value), str(value))
        if internal not in world.variables:
            raise ValueError(f"candidate anchor {anchor} is not a world variable")
        anchors[anchor] = world.variables.index(internal)
    if "objective" in query:
        anchors["objective"] = str(query["objective"])
    return {
        **seed,
        "manipulability": default_manipulability(world, query_type, anchors),
        "readable": {name: True for name in world.variables},
        "observation_bandwidth": len(world.variables),
    }


def task_answerability(
    candidates: Sequence[tuple[WorldSpec, Mapping[str, Any]]],
    *,
    measure_max: int | None = None,
) -> dict[str, str]:
    """Partition task families by exact answerability before K/M sampling.

    Two candidates are observationally equivalent when they expose the same
    query-specific full experiment surface (up to opaque label spelling) and
    every legal passive or hard-do law is identical. Numerical candidates are
    answerable when that evidence determines one exact effect or interval.
    Decision candidates are answerable
    when every indistinguishable world shares at least one zero-regret terminal
    action. Discovery candidates are answerable when the evidence determines
    one exact structural answer.

    The classification is relative to the supplied task family. Per-seed K and
    M, batch sizes, and round counts are intentionally absent: they affect the
    difficulty of gathering evidence, not task-family answerability.
    """

    if measure_max is not None:
        raise ValueError("task answerability is defined before observation bandwidth M")

    from .rendering import (
        rendered_ate_query_surface,
        rendered_counterfactual_query_surface,
        rendered_decision_query_surface,
        rendered_discovery_query_surface,
    )

    records: list[
        tuple[
            str,
            tuple[object, ...],
            tuple[object, ...],
            frozenset[tuple[object, ...]],
        ]
    ] = []
    seen_ids: set[str] = set()
    for world, seed in candidates:
        if not isinstance(world, WorldSpec):
            raise TypeError("candidate world must be a WorldSpec")
        if not isinstance(seed, Mapping):
            raise TypeError("candidate seed must be a mapping")
        seed_id = seed.get("seed_id")
        if not isinstance(seed_id, str) or not seed_id:
            raise ValueError("candidate seed_id must be a nonempty string")
        if seed_id in seen_ids:
            raise ValueError(f"duplicate candidate seed_id: {seed_id}")
        seen_ids.add(seed_id)
        query = seed.get("query")
        query_type = str(query.get("type")) if isinstance(query, Mapping) else ""
        family_seed = _family_answerability_seed(world, seed, query_type)
        if query_type == "ate":
            surface = rendered_ate_query_surface(family_seed)
        elif query_type == "counterfactual_transition_bounds":
            surface = rendered_counterfactual_query_surface(family_seed)
        elif query_type == "best_intervention":
            surface = rendered_decision_query_surface(family_seed)
        elif query_type in {"backadj_minimal_sets", "mediator_set"}:
            surface = rendered_discovery_query_surface(family_seed)
        else:
            raise ValueError(f"task_answerability does not support {query_type}")
        records.append(
            (
                seed_id,
                surface.semantic_key,
                _response_signature(world, family_seed, surface),
                _acceptable_answers(world, family_seed, surface),
            )
        )

    answers_by_observation: dict[tuple[object, ...], list[frozenset[tuple[object, ...]]]] = {}
    for _, public_key, response_signature, answers in records:
        equivalence_key = (public_key, response_signature)
        answers_by_observation.setdefault(equivalence_key, []).append(answers)

    answerability_by_observation: dict[tuple[object, ...], str] = {}
    for equivalence_key, answer_sets in answers_by_observation.items():
        common_answers = set(answer_sets[0])
        for answer_set in answer_sets[1:]:
            common_answers.intersection_update(answer_set)
        answerability_by_observation[equivalence_key] = (
            "answerable" if common_answers else "unanswerable"
        )
    return {
        seed_id: answerability_by_observation[(public_key, response_signature)]
        for seed_id, public_key, response_signature, _ in records
    }


def task_difficulty_profile(
    grammar: WorldGrammar,
    seed: int,
    query_type: str,
    anchors: Mapping[str, int | str],
    *,
    sample_count: int = 200,
) -> Mapping[str, Any]:
    """Report structure and target-distribution difficulty coordinates.

    No thresholds are applied and no planner is used. Answerability is a
    family-level property computed separately by :func:`task_answerability`.
    """

    structural = sample_world(grammar, seed)
    target_profile = profile_task_targets(
        grammar, seed, query_type, anchors, sample_count=sample_count
    )
    structural_coordinates: dict[str, Any] = {
        "node_count": len(structural.variables),
        "edge_count": len(structural.edges),
        "domains": structural.domains,
        "max_domain_size": max(structural.domains),
    }
    if query_type in {
        "ate",
        "counterfactual_transition_bounds",
        "backadj_minimal_sets",
        "mediator_set",
    }:
        path = structural.shortest_path_nodes(int(anchors["treatment"]), int(anchors["outcome"]))
        structural_coordinates["treatment_outcome_path_length"] = len(path or ())
    if query_type == "best_intervention":
        outcome = int(anchors["outcome"])
        decision_target = int(anchors["decision_target"])
        structural_coordinates["decision_state_count"] = structural.domains[decision_target]
        structural_coordinates["experimental_target_count"] = sum(
            1 for node in range(len(structural.variables)) if node not in {decision_target, outcome}
        )
        ancestors: set[int] = set()
        stack = [outcome]
        while stack:
            node = stack.pop()
            for parent in structural.parents[node]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    stack.append(parent)
        structural_coordinates["outcome_ancestor_count"] = len(ancestors)
    return {
        "seed": seed,
        "query_type": query_type,
        "structure": structural_coordinates,
        "target_profile": target_profile,
    }


def load_bnlearn_world(path: str | Path) -> WorldSpec:
    """Load a bnlearn BIF file into the same WorldSpec representation.

    Conditional rows are placed in the canonical parent-assignment order
    declared by the variable blocks, not in the textual order of the BIF file.
    Every declared parent assignment must be present and every row must sum to
    one; otherwise the loader fails closed.
    """

    text = Path(path).read_text(encoding="utf-8")
    declared_state_names = bif_state_names(text)
    variables = list(declared_state_names)
    state_names = [declared_state_names[name] for name in variables]

    index = {name: i for i, name in enumerate(variables)}
    domains = tuple(len(names) for names in state_names)
    parents: dict[int, tuple[int, ...]] = {i: () for i in range(len(variables))}
    edges: list[tuple[int, int]] = []
    cpt: dict[int, list[list[Fraction] | None]] = {i: [None] for i in range(len(variables))}

    probability_blocks = re.findall(
        r"probability\s*\(\s*(\w+)(?:\s*\|\s*([^)]+))?\s*\)\s*\{(.*?)\n\}",
        text,
        re.S,
    )
    if not probability_blocks:
        raise ValueError(f"no probability blocks found in BIF file {path}")
    for child_name, parent_text, body in probability_blocks:
        if child_name not in index:
            raise ValueError(f"probability block references unknown variable {child_name}")
        child = index[child_name]
        parent_names = [name.strip() for name in parent_text.split(",")] if parent_text else []
        if len(set(parent_names)) != len(parent_names):
            raise ValueError(f"duplicate parent in probability block for {child_name}")
        parent_nodes = tuple(index[name] for name in parent_names)
        if child in parent_nodes:
            raise ValueError(f"self-loop detected on {child_name}")
        parents[child] = parent_nodes
        for parent in parent_nodes:
            edge = (parent, child)
            if edge not in edges:
                edges.append(edge)
        row_count = 1
        for parent in parent_nodes:
            row_count *= domains[parent]
        cpt[child] = [None] * row_count

        table_matches = re.findall(r"\btable\s*([0-9.,\s]+)\s*;", body)
        conditional_matches = re.findall(r"\((.*?)\)\s*([0-9.,\s]+)\s*;", body)
        raw_rows: list[tuple[str | None, str]] = []
        for values in table_matches:
            raw_rows.append((None, values))
        for condition, values in conditional_matches:
            raw_rows.append((condition, values))
        if not raw_rows:
            raise ValueError(f"no CPT rows found for {child_name}")

        for condition, values in raw_rows:
            if condition is None:
                if parent_nodes:
                    raise ValueError(f"table row with parents for {child_name}")
                row_index = 0
            else:
                state_tokens = [token.strip() for token in condition.split(",")]
                if len(state_tokens) != len(parent_nodes):
                    raise ValueError(f"wrong parent state count for {child_name}")
                row_index = 0
                for parent, token in zip(parent_nodes, state_tokens, strict=True):
                    if token not in state_names[parent]:
                        raise ValueError(f"unknown parent state {token} for {child_name}")
                    row_index = row_index * domains[parent] + state_names[parent].index(token)
            probabilities = tuple(
                Fraction(value.strip()) for value in values.split(",") if value.strip()
            )
            if len(probabilities) != domains[child]:
                raise ValueError(f"wrong child state count for {child_name}")
            if sum(probabilities, Fraction(0)) != 1:
                raise ValueError(f"CPT row does not sum to one for {child_name}")
            if cpt[child][row_index] is not None:
                raise ValueError(f"duplicate CPT row for {child_name}")
            cpt[child][row_index] = list(probabilities)

    for child, rows in cpt.items():
        if any(row is None for row in rows):
            raise ValueError(f"missing CPT row for {variables[child]}")

    return WorldSpec(
        family="bnlearn_bif",
        topology=Path(path).stem,
        variables=tuple(variables),
        domains=domains,
        state_names=tuple(state_names),
        edges=tuple(edges),
        parents=parents,
        cpt={node: tuple(tuple(row) for row in rows) for node, rows in cpt.items()},
    )


def load_cladder_world(model_id: int, *, source_path: str | Path | None = None) -> WorldSpec:
    """Load a CLadder meta-model into the same WorldSpec representation."""

    path = (
        Path(source_path)
        if source_path
        else Path(__file__).resolve().parents[2] / ("data/worlds/cladder/meta-models-subset.json")
    )
    models = json.loads(path.read_text(encoding="utf-8"))
    model = next((candidate for candidate in models if candidate.get("model_id") == model_id), None)
    if model is None:
        raise ValueError(f"unknown CLadder model_id {model_id}")

    structure = str(model["structure"]).replace(" ", "")
    variables: list[str] = []
    for token in structure.split(","):
        for name in token.split("->"):
            if name not in variables:
                variables.append(name)
    index = {name: i for i, name in enumerate(variables)}
    parents: dict[int, tuple[int, ...]] = {i: () for i in range(len(variables))}
    edges: list[tuple[int, int]] = []
    for token in structure.split(","):
        left, right = token.split("->")
        edges.append((index[left], index[right]))
        parents[index[right]] = parents[index[right]] + (index[left],)

    def to_rows(value: Any) -> tuple[tuple[Fraction, ...], ...]:
        if isinstance(value, list):
            rows: list[tuple[Fraction, ...]] = []
            for item in value:
                rows.extend(to_rows(item))
            return tuple(rows)
        probability = Fraction(str(value))
        if not 0 <= probability <= 1:
            raise ValueError(f"CLadder parameter out of range for model {model_id}")
        return ((1 - probability, probability),)

    params = model["params"]
    cpt: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
    for node, name in enumerate(variables):
        key = f"p({name})"
        if key in params:
            cpt[node] = to_rows(params[key])
        else:
            child_key = next(key for key in params if key.startswith(f"p({name} |"))
            cpt[node] = to_rows(params[child_key])

    return WorldSpec(
        family="cladder_meta_model",
        topology=model["graph_id"],
        variables=tuple(variables),
        domains=tuple(2 for _ in variables),
        state_names=tuple(("0", "1") for _ in variables),
        edges=tuple(edges),
        parents=parents,
        cpt=cpt,
    )


def iter_upstream_worlds() -> tuple[WorldSpec, ...]:
    """Enumerate every upstream world pinned under ``data/worlds``.

    CLadder meta-models and bnlearn BIF files are loaded into the same
    ``WorldSpec`` representation; no second world owner is introduced.
    """

    repo = Path(__file__).resolve().parents[2]
    cladder_path = repo / "data/worlds/cladder/meta-models-subset.json"
    models = json.loads(cladder_path.read_text(encoding="utf-8"))
    worlds: list[WorldSpec] = [
        load_cladder_world(int(model["model_id"]), source_path=cladder_path) for model in models
    ]
    for bif_path in sorted((repo / "data/worlds/bnlearn").glob("*.bif")):
        worlds.append(load_bnlearn_world(bif_path))
    return tuple(worlds)


def _is_acyclic(node_count: int, edges: tuple[tuple[int, int], ...]) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    indegree = [0] * node_count
    for parent, child in edges:
        if not 0 <= parent < node_count or not 0 <= child < node_count:
            return False
        if parent == child:
            return False
        adjacency[parent].append(child)
        indegree[child] += 1
    queue = [node for node, degree in enumerate(indegree) if degree == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return seen == node_count


def legal_world(world: WorldSpec) -> bool:
    if len(world.variables) != len(world.domains):
        return False
    if len(world.state_names) != len(world.variables):
        return False
    if any(size <= 0 for size in world.domains):
        return False
    if any(
        len(names) != size for names, size in zip(world.state_names, world.domains, strict=True)
    ):
        return False
    for child, parent_nodes in world.parents.items():
        if child < 0 or child >= len(world.variables):
            return False
        if len(parent_nodes) != len(set(parent_nodes)):
            return False
        if any(parent < 0 or parent >= len(world.variables) for parent in parent_nodes):
            return False
    if not _is_acyclic(len(world.variables), world.edges):
        return False
    for child in range(len(world.variables)):
        parent_nodes = world.parents.get(child, ())
        parent_combinations = 1
        for parent in parent_nodes:
            parent_combinations *= world.domains[parent]
        rows = world.cpt.get(child, ())
        if len(rows) != parent_combinations:
            return False
        for row in rows:
            if len(row) != world.domains[child]:
                return False
            if any(not 0 <= value <= 1 for value in row):
                return False
            if sum(row, Fraction(0)) != 1:
                return False
    return True


def legal_query_anchors(world: WorldSpec, query_type: str) -> tuple[dict[str, int | str], ...]:
    """Return every structurally legal anchor assignment for a query.

    These are structural legality rules only. Query truth and scorers are not
    implemented here, so an assignment being listed does not yet certify a
    nondegenerate or solvable task.
    """

    if query_type not in QUERY_TYPES:
        return ()
    node_count = len(world.variables)
    if query_type in {"ate", "counterfactual_transition_bounds", "backadj_minimal_sets"}:
        return tuple(
            {"treatment": source, "outcome": target}
            for source in range(node_count)
            for target in range(node_count)
            if world.path_exists(source, target)
        )
    if query_type == "mediator_set":
        return tuple(
            {"treatment": source, "outcome": target}
            for source in range(node_count)
            for target in range(node_count)
            if world.has_indirect_path(source, target)
        )
    if query_type == "best_intervention":
        return tuple(
            {
                "decision_target": decision_target,
                "outcome": outcome,
                "objective": objective,
            }
            for outcome in range(node_count)
            if world.child_count(outcome) == 0 and world.parent_count(outcome) > 0
            for decision_target in range(node_count)
            if decision_target != outcome and world.path_exists(decision_target, outcome)
            for objective in ("minimize", "maximize")
        )
    return ()


def supports_query(world: WorldSpec, query_type: str) -> bool:
    if query_type not in QUERY_TYPES:
        return False
    return bool(legal_query_anchors(world, query_type))


def supports_task(query_type: str, task_head: str) -> bool:
    return (
        query_type in QUERY_TYPES
        and task_head in TASK_HEADS
        and query_task_compatible(query_type, task_head)
    )


def _normalize_hiding_modes(hiding: str | object) -> tuple[str, ...]:
    if isinstance(hiding, str):
        modes = (hiding,)
    elif isinstance(hiding, (tuple, list, set, frozenset)):
        modes = tuple(str(mode) for mode in hiding)
    else:
        raise TypeError("hiding must be a mode string or an iterable of mode strings")
    unknown = set(modes) - HIDING_MODES
    if unknown:
        raise ValueError(f"unsupported hiding modes: {sorted(unknown)}")
    return modes


def supports_hiding(world: WorldSpec, hiding: str | object) -> bool:
    del world
    try:
        _normalize_hiding_modes(hiding)
    except (TypeError, ValueError):
        return False
    return True


def default_manipulability(
    world: WorldSpec, query_type: str, anchors: Mapping[str, int | str]
) -> dict[str, bool]:
    """Derive the anchor-minimal manipulability mask.

    Every non-anchor variable defaults to manipulable and readable. Query
    outcome and collider anchors are readonly. For ATE and counterfactual
    transition bounds, the treatment is also readonly. For best intervention,
    the deployment target is readonly during experimentation: its candidate
    states are reserved for the terminal decision rather than being directly
    sampled.
    """

    manipulability = {name: True for name in world.variables}
    for anchor_name in ("outcome", "collider"):
        node = anchors.get(anchor_name)
        if isinstance(node, int):
            manipulability[world.variables[node]] = False
    if query_type in {"ate", "counterfactual_transition_bounds"}:
        treatment = anchors.get("treatment")
        if isinstance(treatment, int):
            manipulability[world.variables[treatment]] = False
    if query_type == "best_intervention":
        decision_target = anchors.get("decision_target")
        if isinstance(decision_target, int):
            manipulability[world.variables[decision_target]] = False
    return manipulability


def _randomized_interaction_surface(
    world: WorldSpec,
    query_type: str,
    anchors: Mapping[str, int | str],
    *,
    seed_id: str,
) -> tuple[dict[str, bool], int]:
    """Draw the main-pipeline manipulability width and observation bandwidth.

    Query anchors remain readonly.  Conditional on the remaining candidate
    variables, the width is uniform over every nonempty size and the subset is
    uniform within that size.  Observation bandwidth is an independent
    uniform draw over ``1..n``.  This is a private stage of the existing seed
    pipeline, not a second task sampler.
    """

    base = default_manipulability(world, query_type, anchors)
    candidates = tuple(name for name in world.variables if base[name])
    if not candidates:
        raise ValueError("query leaves no non-anchor intervention candidate")

    def rng(axis: str) -> random.Random:
        payload = f"cpt-world-interaction-surface-v1\0{axis}\0{seed_id}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))

    width_rng = rng("manipulability")
    width = width_rng.randint(1, len(candidates))
    selected = frozenset(width_rng.sample(candidates, width))
    manipulability = {name: name in selected for name in world.variables}
    observation_bandwidth = rng("observation-bandwidth").randint(1, len(world.variables))
    return manipulability, observation_bandwidth


def anonymize_world(
    world: WorldSpec, seed_id: str
) -> tuple[dict[str, str], tuple[Mapping[str, Any], ...]]:
    """Rename internal variables and states into opaque tokens."""

    labels = _opaque_labels(len(world.variables), seed_id)
    label_map = dict(zip(world.variables, labels, strict=True))
    visible_variables = tuple(
        {
            "label": label_map[name],
            "states": [f"state_{i}" for i in range(world.domains[i])],
        }
        for i, name in enumerate(world.variables)
    )
    return label_map, visible_variables


def hide_world(
    visible_variables: tuple[Mapping[str, Any], ...],
    query_visible: Mapping[str, Any],
    hiding_modes: tuple[str, ...],
) -> Mapping[str, Any]:
    """Build the model-visible object.

    This is hiding, not anonymization: anonymization already produced opaque
    labels and states; this function omits every non-visible field and only
    adds readout/action syntax flags that the renderer is allowed to surface.
    """

    modes = _normalize_hiding_modes(hiding_modes)
    visible: dict[str, Any] = {"variables": list(visible_variables), "query": query_visible}
    if "evidence_by_intervention_only" in modes:
        visible["initial_evidence"] = "none"
    if "no_full_joint" in modes:
        visible["readout"] = "measure_subset_only"
    if "manipulability_via_action_legality" in modes:
        visible["action_feedback"] = "legality_only"
    return visible


def assemble_seed(
    world: WorldSpec,
    hiding: str | object,
    query_type: str,
    task_head: str,
    *,
    seed_id: str,
    anchors: Mapping[str, int | str] | None = None,
    manipulability: Mapping[str, bool] | None = None,
    readable: Mapping[str, bool] | None = None,
    observation_bandwidth: int | None = None,
    answer_mode: str | None = None,
) -> Mapping[str, Any]:
    """Assemble an anonymous candidate seed or fail closed.

    ``anchors`` selects one of the structurally legal query assignments from
    :func:`legal_query_anchors`. If omitted, the first legal assignment is
    used. Masks default to the anchor-minimal rule from
    :func:`default_manipulability`; pinned seed masks can be passed explicitly.
    ``observation_bandwidth`` is optional for legacy/manual seeds and fixed in
    every seed emitted by the main sampler. ``answer_mode`` only selects one
    of the two registered terminal formats for a counterfactual-bound task.
    """

    if not isinstance(seed_id, str) or not seed_id:
        raise ValueError("seed_id must be a nonempty string")
    hiding_modes = _normalize_hiding_modes(hiding)
    if not legal_world(world):
        raise ValueError(f"{seed_id}: illegal world")
    if not supports_query(world, query_type):
        raise ValueError(f"{seed_id}: query does not match world")
    if not supports_task(query_type, task_head):
        raise ValueError(f"{seed_id}: task does not match query")
    if not supports_hiding(world, hiding_modes):
        raise ValueError(f"{seed_id}: unsupported hiding mode")
    if answer_mode is not None and query_type != "counterfactual_transition_bounds":
        raise ValueError("answer_mode is only valid for counterfactual transition bounds")

    legal_anchors = legal_query_anchors(world, query_type)
    if anchors is None:
        if not legal_anchors:
            raise ValueError(f"{seed_id}: query has no legal anchor assignment")
        selected_anchors = dict(legal_anchors[0])
    else:
        selected_anchors = dict(anchors)
        if selected_anchors not in [dict(item) for item in legal_anchors]:
            raise ValueError(f"{seed_id}: anchors are not structurally legal for query")

    label_map, visible_variables = anonymize_world(world, seed_id)

    def anchor_label(anchor_name: str) -> str:
        return label_map[world.variables[int(selected_anchors[anchor_name])]]

    def visible_state(node_name: str, state_index: int) -> str:
        node = world.variables.index(node_name)
        if state_index < 0 or state_index >= world.domains[node]:
            raise ValueError(f"state index {state_index} out of range for {node_name}")
        return f"state_{state_index}"

    query_visible: dict[str, Any] = {"type": query_type}
    if query_type in {
        "ate",
        "counterfactual_transition_bounds",
        "backadj_minimal_sets",
        "mediator_set",
    }:
        outcome_name = world.variables[int(selected_anchors["outcome"])]
        query_visible.update(
            {
                "treatment": anchor_label("treatment"),
                "outcome": anchor_label("outcome"),
            }
        )
        if query_type in {"ate", "counterfactual_transition_bounds", "mediator_set"}:
            query_visible["outcome_state"] = visible_state(outcome_name, 1)
        if query_type == "counterfactual_transition_bounds":
            treatment_name = world.variables[int(selected_anchors["treatment"])]
            query_visible["treatment_value"] = visible_state(treatment_name, 1)
            query_visible["baseline_value"] = visible_state(treatment_name, 0)
            if answer_mode is not None:
                query_visible["answer_mode"] = answer_mode
            query_visible["answer_mode"] = counterfactual_answer_mode(query_visible)
    elif query_type == "best_intervention":
        outcome_name = world.variables[int(selected_anchors["outcome"])]
        query_visible.update(
            {
                "decision_target": anchor_label("decision_target"),
                "outcome": anchor_label("outcome"),
                "objective": str(selected_anchors["objective"]),
                "outcome_state": visible_state(outcome_name, 1),
            }
        )

    manipulability_map = (
        default_manipulability(world, query_type, selected_anchors)
        if manipulability is None
        else dict(manipulability)
    )
    readable_map = {name: True for name in world.variables} if readable is None else dict(readable)
    for mask_name, mask in (("manipulability", manipulability_map), ("readable", readable_map)):
        if set(mask) != set(world.variables):
            raise ValueError(f"{seed_id}: {mask_name} variable set does not match world")
        if any(not isinstance(value, bool) for value in mask.values()):
            raise ValueError(f"{seed_id}: {mask_name} values must be bool")
    if observation_bandwidth is not None:
        if (
            isinstance(observation_bandwidth, bool)
            or not isinstance(observation_bandwidth, int)
            or not 1 <= observation_bandwidth <= len(world.variables)
        ):
            raise ValueError(
                f"{seed_id}: observation_bandwidth must lie in [1, {len(world.variables)}]"
            )

    visible = hide_world(visible_variables, query_visible, hiding_modes)
    assembled: dict[str, Any] = {
        "seed_id": seed_id,
        "hiding_modes": hiding_modes,
        "world_source": {
            "type": world.family,
            "family": world.family,
            "topology": world.topology,
            "variables": world.variables,
            "domains": world.domains,
            "state_names": {
                name: tuple(world.state_names[index]) for index, name in enumerate(world.variables)
            },
            "edges": [[world.variables[p], world.variables[c]] for p, c in world.edges],
            "cpt": {
                world.variables[node]: [[str(value) for value in row] for row in rows]
                for node, rows in world.cpt.items()
            },
        },
        "manipulability": manipulability_map,
        "readable": readable_map,
        "visible_schema": {
            "variable_labels": label_map,
            **visible,
        },
        "query": query_visible,
        "task_head": {"head": task_head},
    }
    if observation_bandwidth is not None:
        assembled["observation_bandwidth"] = observation_bandwidth
    return assembled


def render_seed_prompt(seed: Mapping[str, Any]) -> str:
    """Render a seed task prompt (backwards-compatible convenience wrapper)."""

    from .rendering import render_seed_task_prompt

    return render_seed_task_prompt(seed)


def iter_world_space(
    grammar: WorldGrammar,
    *,
    include_upstream: bool = True,
    start_seed: int = 0,
    count: int = 1,
) -> tuple[WorldSpec, ...]:
    """Enumerate upstream worlds first, then sampled worlds from the grammar."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    worlds = list(iter_upstream_worlds()) if include_upstream else []
    for seed in range(start_seed, start_seed + count):
        world = sample_world(grammar, seed)
        if not legal_world(world):
            raise ValueError(f"sampled world {seed} is not legal")
        worlds.append(world)
    return tuple(worlds)


def iter_sampled_seeds(
    grammar: WorldGrammar,
    *,
    query_types: tuple[str, ...],
    start_seed: int = 0,
    count: int = 1,
    hiding: str | object = "mechanism_hidden",
) -> tuple[Mapping[str, Any], ...]:
    """Yield generic task seeds from stable worlds through the main pipeline.

    Linear expansion step:

    ``structure -> hiding -> numerical query -> legal anchors -> task``.

    The caller explicitly names a nonempty subset of the five implemented
    query types. All use the same assembly path. Numerical ATE and decision
    tasks use the existing stable-CPT resampling owner; structural discovery
    tasks use the already sampled structural world directly. The main pipeline
    samples a nonempty manipulable subset of the eligible variables and an
    independent seed-fixed observation bandwidth.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    admitted_query_types = frozenset(
        {
            "ate",
            "counterfactual_transition_bounds",
            "backadj_minimal_sets",
            "best_intervention",
            "mediator_set",
        }
    )
    if not query_types:
        raise ValueError("query_types must not be empty")
    if len(set(query_types)) != len(query_types):
        raise ValueError("query_types must not contain duplicates")
    unknown_query_types = set(query_types) - admitted_query_types
    if unknown_query_types:
        raise ValueError(
            f"generic sampler does not admit query types: {sorted(unknown_query_types)}"
        )
    generated: list[tuple[WorldSpec, Mapping[str, Any]]] = []
    for sample_index in range(start_seed, start_seed + count):
        structural_world = sample_world(grammar, sample_index)
        for query_type in query_types:
            legal_anchors = legal_query_anchors(structural_world, query_type)
            if not legal_anchors:
                continue
            for anchor_index, anchors in enumerate(legal_anchors):
                for task_head in TASK_HEADS:
                    if not supports_task(query_type, task_head):
                        continue
                    try:
                        task_world = (
                            structural_world
                            if query_type in {"backadj_minimal_sets", "mediator_set"}
                            else sample_task_world(grammar, sample_index, query_type, anchors)
                        )
                        base_seed_id = (
                            f"SAMPLED-{sample_index}-{query_type}-{task_head}-a{anchor_index}"
                        )
                        manipulability, observation_bandwidth = _randomized_interaction_surface(
                            task_world,
                            query_type,
                            anchors,
                            seed_id=base_seed_id,
                        )
                        base_assembled = assemble_seed(
                            task_world,
                            hiding,
                            query_type,
                            task_head,
                            anchors=anchors,
                            seed_id=base_seed_id,
                            manipulability=manipulability,
                            observation_bandwidth=observation_bandwidth,
                        )
                        answer_modes: tuple[str | None, ...] = (
                            tuple(sorted(COUNTERFACTUAL_ANSWER_MODES))
                            if query_type == "counterfactual_transition_bounds"
                            else (None,)
                        )
                        for answer_mode in answer_modes:
                            seed_id = (
                                f"{base_seed_id}-mode-{answer_mode}"
                                if answer_mode is not None
                                else base_seed_id
                            )
                            if answer_mode is None:
                                assembled = base_assembled
                            else:
                                query = {
                                    **base_assembled["query"],
                                    "answer_mode": answer_mode,
                                }
                                assembled = {
                                    **base_assembled,
                                    "seed_id": seed_id,
                                    "query": query,
                                    "visible_schema": {
                                        **base_assembled["visible_schema"],
                                        "query": query,
                                    },
                                }
                            # Reuse the renderer's action-surface owner. Seeds with
                            # no legal non-endpoint experiment are not emitted.
                            render_seed_prompt(assembled)
                            generated.append((task_world, assembled))
                    except (ValueError, NotImplementedError):
                        continue
    answerability_candidates = tuple(generated)
    answerability = task_answerability(answerability_candidates) if answerability_candidates else {}
    return tuple(
        {
            **seed,
            "answerability": answerability[str(seed["seed_id"])],
            "answerability_scope": "sampled_candidate_family_full_query_surface",
        }
        for _, seed in generated
    )
