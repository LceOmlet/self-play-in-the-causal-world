"""Sampler for finite hidden CPT worlds.

The main sampler owns one declared distribution:

- node count is uniform over ``node_counts`` (8 through 16 by default);
- node cardinalities are independent and uniform over ``2..max_domain_size``;
- the node order is a uniform permutation;
- in the official 8--16-node distribution, each node's parent count is uniform
  over ``0..min(floor(node_count / 3), predecessor_count)``; smaller custom
  grammars retain the legacy ceiling of three;
- the parent subset is uniform conditional on that count;
- CPTs use a simplex-uniform base distribution.  A uniform undirected graph on
  each node's parents activates exactly the categorical functional-ANOVA score
  blocks indexed by its nonempty cliques; a simplex-uniform energy split is
  applied across those active blocks. ET-V2 maps the score table to
  probabilities by RMS-normalized exponential tilting.
  Every parent-bearing table receives the binary-preserving contextual
  parent-pair contrast correction declared by ADR 0026. The uniform amplitude
  has one unit of expected squared score energy before that geometric
  correction.

Generated worlds use ``float``/binary64 probabilities. Exact ``Fraction``
worlds remain valid fixed fixtures and reference inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

from .registry import (
    HIDING_MODES,
    QUERY_TYPES,
    TASK_FAMILY_QUERY_TYPES,
    TASK_HEADS,
    query_task_compatible,
)

_LABEL_POOL = "DEFGHIJKLMNOPQRSTUVW"

DEFAULT_NODE_COUNTS = tuple(range(8, 17))
_MAX_GRAMMAR_NODES = 16
_BACKDOOR_STRATIFIED_QUERY_TYPES = frozenset(TASK_FAMILY_QUERY_TYPES)
_CPT_VALIDITY_TOLERANCE = 1e-12
_ET_V2_STRENGTH_CEILING = math.sqrt(3.0)

Probability = float | Fraction


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
    cpt: Mapping[int, tuple[tuple[Probability, ...], ...]]

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
    """

    node_counts: tuple[int, ...] = DEFAULT_NODE_COUNTS
    max_domain_size: int = 5

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


def _simplex_uniform(domain_size: int, rng: random.Random) -> tuple[float, ...]:
    """Draw ``Dirichlet(1, ..., 1)`` without introducing a second RNG owner."""

    while True:
        draws = [rng.expovariate(1.0) for _ in range(domain_size)]
        total = math.fsum(draws)
        if total > 0.0 and all(value > 0.0 for value in draws):
            return tuple(value / total for value in draws)


def _project_joint_effect(
    row_count: int,
    domain_size: int,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    """Draw a unit direction in the row/column-centred joint-effect space."""

    while True:
        gaussian = [[rng.gauss(0.0, 1.0) for _ in range(domain_size)] for _ in range(row_count)]
        row_means = [math.fsum(row) / domain_size for row in gaussian]
        column_means = [
            math.fsum(gaussian[row][state] for row in range(row_count)) / row_count
            for state in range(domain_size)
        ]
        grand_mean = math.fsum(row_means) / row_count
        projected = [
            [
                gaussian[row][state] - row_means[row] - column_means[state] + grand_mean
                for state in range(domain_size)
            ]
            for row in range(row_count)
        ]
        norm = math.sqrt(math.fsum(value * value for row in projected for value in row))
        if math.isfinite(norm) and norm > 0.0:
            return tuple(tuple(value / norm for value in row) for row in projected)


def _parent_state_at_row(
    row_index: int,
    parent_domains: Sequence[int],
    parent_position: int,
) -> int:
    """Decode one parent state from the canonical mixed-radix CPT row index."""

    stride = math.prod(parent_domains[parent_position + 1 :])
    return (row_index // stride) % parent_domains[parent_position]


def _parent_subset_marginal_projection(
    table: Sequence[Sequence[float]],
    parent_domains: Sequence[int],
    parent_positions: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    """Project a table onto functions of one declared parent subset."""

    row_count = math.prod(parent_domains)
    if len(table) != row_count:
        raise ValueError("table row count does not match the parent domains")
    if not table or not table[0]:
        raise ValueError("table must have at least one row and one child state")
    domain_size = len(table[0])
    if any(len(row) != domain_size for row in table):
        raise ValueError("table rows must have a common child-state count")
    positions = tuple(parent_positions)
    if tuple(sorted(set(positions))) != positions:
        raise ValueError("parent_positions must be strictly increasing")
    if any(position < 0 or position >= len(parent_domains) for position in positions):
        raise ValueError("parent_positions contains an invalid parent position")

    totals: dict[tuple[int, ...], list[float]] = {}
    counts: dict[tuple[int, ...], int] = {}
    for row_index, row in enumerate(table):
        key = tuple(
            _parent_state_at_row(row_index, parent_domains, position) for position in positions
        )
        target = totals.setdefault(key, [0.0] * domain_size)
        counts[key] = counts.get(key, 0) + 1
        for child_state, value in enumerate(row):
            target[child_state] += value

    means = {key: tuple(value / counts[key] for value in values) for key, values in totals.items()}
    return tuple(
        means[
            tuple(
                _parent_state_at_row(row_index, parent_domains, position) for position in positions
            )
        ]
        for row_index in range(row_count)
    )


def _parent_interaction_projection(
    table: Sequence[Sequence[float]],
    parent_domains: Sequence[int],
    parent_positions: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    """Return one pure categorical ANOVA interaction component."""

    positions = tuple(parent_positions)
    if not positions:
        raise ValueError("an interaction requires at least one parent")
    if tuple(sorted(set(positions))) != positions:
        raise ValueError("parent_positions must be strictly increasing")

    marginals: list[tuple[int, tuple[tuple[float, ...], ...]]] = []
    for subset_size in range(len(positions) + 1):
        sign = -1 if (len(positions) - subset_size) % 2 else 1
        for subset in combinations(positions, subset_size):
            marginals.append(
                (sign, _parent_subset_marginal_projection(table, parent_domains, subset))
            )

    return tuple(
        tuple(
            math.fsum(sign * marginal[row_index][child_state] for sign, marginal in marginals)
            for child_state in range(len(table[0]))
        )
        for row_index in range(len(table))
    )


def _project_parent_subset_effect(
    parent_domains: Sequence[int],
    parent_positions: Sequence[int],
    domain_size: int,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    """Draw an isotropic unit direction for one pure parent-subset interaction."""

    positions = tuple(parent_positions)
    if not positions:
        raise ValueError("an interaction requires at least one parent")
    if tuple(sorted(set(positions))) != positions:
        raise ValueError("parent_positions must be strictly increasing")
    if any(position < 0 or position >= len(parent_domains) for position in positions):
        raise ValueError("parent_positions contains an invalid parent position")
    while True:
        joint = _project_joint_effect(math.prod(parent_domains), domain_size, rng)
        subset_projection = _parent_interaction_projection(
            joint,
            parent_domains,
            positions,
        )
        norm = math.sqrt(math.fsum(value * value for row in subset_projection for value in row))
        if math.isfinite(norm) and norm > 0.0:
            return tuple(tuple(value / norm for value in row) for row in subset_projection)


def _combine_effect_blocks(
    blocks: Sequence[Sequence[Sequence[float]]],
    energy_shares: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Combine orthonormal effect blocks while conserving their squared-energy budget."""

    if not blocks or len(blocks) != len(energy_shares):
        raise ValueError("blocks and energy_shares must have the same nonzero length")
    row_count = len(blocks[0])
    domain_size = len(blocks[0][0])
    if any(
        len(block) != row_count or any(len(row) != domain_size for row in block) for block in blocks
    ):
        raise ValueError("effect blocks must have one common shape")
    if any(not math.isfinite(share) or share < 0.0 for share in energy_shares):
        raise ValueError("energy shares must be finite and nonnegative")
    if not math.isclose(math.fsum(energy_shares), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("energy shares must sum to one")

    amplitudes = tuple(math.sqrt(share) for share in energy_shares)
    combined = tuple(
        tuple(
            math.fsum(
                amplitude * block[row_index][child_state]
                for amplitude, block in zip(amplitudes, blocks, strict=True)
            )
            for child_state in range(domain_size)
        )
        for row_index in range(row_count)
    )
    norm = math.sqrt(math.fsum(value * value for row in combined for value in row))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("combined effect direction has no finite positive norm")
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("effect blocks are not orthonormal under the declared geometry")
    return tuple(tuple(value / norm for value in row) for row in combined)


def _sample_parent_subset_balanced_effect(
    parent_domains: Sequence[int],
    domain_size: int,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    """Sample clique-supported parent interactions under one conserved energy budget.

    A uniformly sampled undirected graph on the parents activates singleton
    main effects and exactly its higher-order cliques.  Conditional on that
    support, the realized shares are one draw from the symmetric simplex
    distribution.  The one-parent law consumes no additional randomness.
    """

    parent_subsets = _sample_parent_interaction_cliques(len(parent_domains), rng)
    blocks = [
        _project_parent_subset_effect(parent_domains, positions, domain_size, rng)
        for positions in parent_subsets
    ]
    energy_shares = (1.0,) if len(blocks) == 1 else _simplex_uniform(len(blocks), rng)
    return _combine_effect_blocks(blocks, energy_shares)


def _sample_parent_interaction_cliques(
    parent_count: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    """Sample the nonempty cliques of a uniform graph on ``parent_count`` vertices.

    With at most three parents this gives the complete parameter-free support
    law directly: every pair-edge mask is equally likely, all singleton main
    effects are active, and a higher-order interaction is active exactly when
    all of its pair edges are present.
    """

    if isinstance(parent_count, bool) or not isinstance(parent_count, int) or parent_count < 1:
        raise ValueError("parent_count must be a positive integer")
    parent_pairs = tuple(combinations(range(parent_count), 2))
    if not parent_pairs:
        return ((0,),)
    edge_mask = rng.randrange(1 << len(parent_pairs))
    selected_edges = frozenset(
        pair for index, pair in enumerate(parent_pairs) if edge_mask & (1 << index)
    )
    return tuple(
        positions
        for subset_size in range(1, parent_count + 1)
        for positions in combinations(range(parent_count), subset_size)
        if subset_size == 1
        or all(tuple(sorted(pair)) in selected_edges for pair in combinations(positions, 2))
    )


def _finalize_cpt_row(values: Sequence[float]) -> tuple[float, ...]:
    """Apply the single accepted binary64 CPT-boundary rule."""

    cleaned: list[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("CPT entries must be finite")
        if value < -_CPT_VALIDITY_TOLERANCE:
            raise ValueError("CPT entry is below the numerical validity tolerance")
        cleaned.append(0.0 if value < 0.0 else value)
    total = math.fsum(cleaned)
    if not math.isfinite(total) or abs(total - 1.0) > _CPT_VALIDITY_TOLERANCE:
        raise ValueError("CPT row does not sum to one within the validity tolerance")
    if total <= 0.0:
        raise ValueError("CPT row has zero mass")
    return tuple(value / total for value in cleaned)


def _exponential_tilt_rows(
    base: Sequence[float],
    direction: Sequence[Sequence[float]],
    strength: float,
    *,
    score_scale: float = 1.0,
) -> tuple[tuple[float, ...], ...]:
    """Map one effect-score table to legal CPT rows under ET-V2.

    The direction is normalized to unit elementwise RMS before tilting. A rare
    base state therefore remains positive without imposing one shared additive
    boundary scale on every row and state.
    """

    if not base or any(not math.isfinite(value) or value <= 0.0 for value in base):
        raise ValueError("base probabilities must be finite and strictly positive")
    if not math.isclose(math.fsum(base), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("base probabilities must sum to one")
    if not direction or any(len(row) != len(base) for row in direction):
        raise ValueError("effect direction must be a nonempty table matching the base")
    if not math.isfinite(strength) or not 0.0 <= strength <= _ET_V2_STRENGTH_CEILING:
        raise ValueError(
            f"effect strength must be finite and lie in [0, {_ET_V2_STRENGTH_CEILING}]"
        )
    if not math.isfinite(score_scale) or score_scale <= 0.0:
        raise ValueError("score_scale must be finite and positive")

    squared_rms = math.fsum(value * value for row in direction for value in row) / (
        len(direction) * len(base)
    )
    if not math.isfinite(squared_rms) or squared_rms <= 0.0:
        raise ValueError("effect direction must have finite positive RMS")
    inverse_rms = 1.0 / math.sqrt(squared_rms)

    cpt_rows: list[tuple[float, ...]] = []
    for row in direction:
        log_weights = tuple(
            math.log(base[state]) + strength * score_scale * inverse_rms * value
            for state, value in enumerate(row)
        )
        maximum = max(log_weights)
        weights = tuple(math.exp(value - maximum) for value in log_weights)
        total = math.fsum(weights)
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("exponential tilt produced no finite positive mass")
        cpt_rows.append(_finalize_cpt_row(tuple(value / total for value in weights)))
    return tuple(cpt_rows)


def _sample_et_v2_strength(rng: random.Random) -> float:
    """Draw uniform amplitude with one unit of expected squared score energy."""

    return _ET_V2_STRENGTH_CEILING * rng.random()


def _single_parent_pairwise_score_scale(parent_domain_size: int) -> float:
    """Preserve the binary ET-V2 mean pairwise score contrast for ``k`` states.

    An RMS-normalized, parent-state-centred score table has mean squared
    elementwise row-pair contrast ``2k / (k - 1)``. The binary value is four,
    so this label-exchangeable correction is one for ``k=2`` and removes only
    the geometric attenuation introduced by additional parent states.
    """

    if (
        isinstance(parent_domain_size, bool)
        or not isinstance(parent_domain_size, int)
        or parent_domain_size < 2
    ):
        raise ValueError("parent_domain_size must be an integer >= 2")
    return math.sqrt(2.0 * (parent_domain_size - 1) / parent_domain_size)


def _contextual_parent_pair_score_scale(
    direction: Sequence[Sequence[float]],
    parent_domains: Sequence[int],
) -> float:
    """Normalize one-parent score changes to the binary ET-V2 reference.

    For each parent, compare every pair of its states while holding all other
    parents fixed. The mean squared elementwise score contrast is averaged
    symmetrically over parents, contexts, state pairs, and child states. The
    returned ``score_scale`` makes that contrast exactly four after ET-V2's
    elementwise RMS normalization. With one parent this is exactly the
    correction returned by :func:`_single_parent_pairwise_score_scale`.
    """

    domains = tuple(parent_domains)
    if not domains or any(
        isinstance(domain, bool) or not isinstance(domain, int) or domain < 2 for domain in domains
    ):
        raise ValueError("parent_domains must contain integers >= 2")
    row_count = math.prod(domains)
    if len(direction) != row_count or not direction or not direction[0]:
        raise ValueError("effect direction row count must match parent_domains")
    child_domain = len(direction[0])
    if any(
        len(row) != child_domain or any(not math.isfinite(value) for value in row)
        for row in direction
    ):
        raise ValueError("effect direction must be a finite rectangular table")

    parent_contrasts: list[float] = []
    for position, parent_domain in enumerate(domains):
        suffix_count = math.prod(domains[position + 1 :])
        prefix_count = math.prod(domains[:position])
        squared_contrast = math.fsum(
            (
                direction[(prefix * parent_domain + left) * suffix_count + suffix][state]
                - direction[(prefix * parent_domain + right) * suffix_count + suffix][state]
            )
            ** 2
            for prefix in range(prefix_count)
            for suffix in range(suffix_count)
            for left in range(parent_domain)
            for right in range(left + 1, parent_domain)
            for state in range(child_domain)
        )
        comparison_count = prefix_count * suffix_count * math.comb(parent_domain, 2) * child_domain
        parent_contrasts.append(squared_contrast / comparison_count)

    mean_squared_contrast = math.fsum(parent_contrasts) / len(parent_contrasts)
    squared_rms = math.fsum(value * value for row in direction for value in row) / (
        row_count * child_domain
    )
    if not math.isfinite(mean_squared_contrast) or mean_squared_contrast <= 0.0:
        raise ValueError("effect direction has no finite positive parent-state contrast")
    if not math.isfinite(squared_rms) or squared_rms <= 0.0:
        raise ValueError("effect direction must have finite positive RMS")
    return 2.0 * math.sqrt(squared_rms / mean_squared_contrast)


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
        nonce = index
        while True:
            digest = hashlib.sha256(
                f"cpt-world-space-labels-v1\0{seed_id}\0{index}\0{nonce}".encode()
            ).digest()
            token = "".join(_LABEL_POOL[byte % len(_LABEL_POOL)] for byte in digest[:3])
            if token not in labels:
                labels.append(token)
                break
            nonce += 1
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


def _sample_structure_at_size(
    grammar: WorldGrammar,
    node_count: int,
    rng: random.Random,
) -> _SampledStructure:
    domains = tuple(rng.randint(2, grammar.max_domain_size) for _ in range(node_count))
    order = list(range(node_count))
    rng.shuffle(order)
    parents_list: list[tuple[int, ...]] = [() for _ in range(node_count)]
    parent_count_ceiling = node_count // 3 if node_count >= 8 else 3
    for position, child in enumerate(order):
        parent_count = rng.randint(0, min(parent_count_ceiling, position))
        parents_list[child] = tuple(sorted(rng.sample(order[:position], parent_count)))
    edges = tuple(
        sorted((parent, child) for child, parents in enumerate(parents_list) for parent in parents)
    )
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


def _sample_structure(grammar: WorldGrammar, rng: random.Random) -> _SampledStructure:
    return _sample_structure_at_size(grammar, rng.choice(grammar.node_counts), rng)


def _build_world(
    structure: _SampledStructure,
    rng: random.Random,
) -> WorldSpec:
    cpt: dict[int, tuple[tuple[float, ...], ...]] = {}
    for node in range(structure.node_count):
        domain_size = structure.domains[node]
        node_parents = structure.parents[node]
        base = _simplex_uniform(domain_size, rng)
        if not node_parents:
            cpt[node] = (_finalize_cpt_row(base),)
        else:
            parent_domains = tuple(structure.domains[parent] for parent in node_parents)
            direction = _sample_parent_subset_balanced_effect(parent_domains, domain_size, rng)
            score_scale = (
                _single_parent_pairwise_score_scale(parent_domains[0])
                if len(parent_domains) == 1
                else _contextual_parent_pair_score_scale(
                    direction,
                    parent_domains,
                )
            )
            cpt[node] = _exponential_tilt_rows(
                base,
                direction,
                _sample_et_v2_strength(rng),
                score_scale=score_scale,
            )
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
    world = _build_world(structure, rng)
    if not legal_world(world):
        raise RuntimeError("internal error: main sampler produced an illegal WorldSpec")
    return world


def _task_target_metrics(
    world: WorldSpec, query_type: str, anchors: Mapping[str, int | str]
) -> Mapping[str, Probability]:
    """Task-relevant numerical targets for optional distribution diagnostics."""

    from .query_truth import (
        best_intervention_truth,
        categorical_treatment_effect,
        interventional_probability,
    )

    if query_type in {"backadj_minimal_sets", "mediator_set"}:
        raise NotImplementedError(
            f"revealing-quality metrics for {query_type} are not designed yet"
        )

    outcome = int(anchors["outcome"])
    baseline = interventional_probability(world, {}, outcome, 1)
    if query_type == "ate":
        effect = categorical_treatment_effect(
            world,
            int(anchors["treatment"]),
            outcome,
        )
        return {
            "target": sum((abs(component) for component in effect), start=0) / 2,
            "baseline": baseline,
        }
    if query_type == "individual_counterfactual_probability":
        raise NotImplementedError(
            "generic target profiling cannot substitute Frechet outer bounds "
            "for the exact individual-counterfactual target; use the exact "
            "counterfactual solver probe, which records timeouts as unresolved"
        )
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
            gap = min(others) - best_probability if others else 0.0
        else:
            others = [p for p in probabilities if p < best_probability]
            gap = best_probability - max(others) if others else 0.0
        return {"target": gap, "baseline": baseline}
    raise ValueError(f"unsupported query type {query_type}")


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
    CPT instances are drawn from the declared parameter distribution. This
    diagnostic never filters or changes the main sampling law.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = random.Random(seed)
    structure = _sample_structure(grammar, rng)
    structural_world = _build_world(structure, rng)
    legal_anchors = legal_query_anchors(structural_world, query_type)
    if dict(anchors) not in [dict(item) for item in legal_anchors]:
        raise ValueError("anchors are not structurally legal for the sampled world")
    targets: list[Probability] = []
    for _ in range(sample_count):
        world = _build_world(structure, rng)
        metrics = _task_target_metrics(world, query_type, anchors)
        targets.append(metrics["target"])
    return {
        "seed": seed,
        "query_type": query_type,
        "sample_count": sample_count,
        "zero_count": sum(1 for target in targets if target == 0),
        "negative_count": sum(1 for target in targets if target < 0),
        "positive_count": sum(1 for target in targets if target > 0),
        "target_min": float(min(targets, default=0.0)),
        "target_max": float(max(targets, default=0.0)),
    }


def sample_task_world(
    grammar: WorldGrammar,
    seed: int,
    query_type: str,
    anchors: Mapping[str, int | str] | None = None,
) -> WorldSpec:
    """Sample one world conditioned only on task structural eligibility.

    Node count is drawn once and held fixed while structures are redrawn. CPT
    parameters are sampled only after an eligible structure is found. Optional
    ``anchors`` validate a caller-selected role assignment without affecting
    the sampled mechanisms.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if query_type not in QUERY_TYPES:
        raise ValueError(f"unsupported query type: {query_type}")
    rng = random.Random(seed)
    node_count = rng.choice(grammar.node_counts)
    while True:
        structure = _sample_structure_at_size(grammar, node_count, rng)
        roles = _sampled_role_assignments(node_count, structure.edges, query_type, seed)
        if roles:
            break
    world = _build_world(structure, rng)
    if not legal_world(world):
        raise RuntimeError("internal error: task sampler produced an illegal WorldSpec")
    if anchors is not None:
        structural_anchors = {
            name: anchors.get(name) for name in QUERY_TYPES[query_type]["anchors"]
        }
        if structural_anchors not in [dict(item) for item in roles]:
            raise ValueError("anchors are not structurally legal for the sampled task world")
    return world


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
            if (
                not isinstance(effect, tuple)
                or len(effect) != surface.domains[surface.outcome]
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, (int, float, Fraction))
                    or not math.isfinite(float(component))
                    for component in effect
                )
            ):
                raise TypeError("ATE truth owner must return one finite effect per outcome state")
            return frozenset({("effect", effect)})
        lower = truth.get("lower")
        upper = truth.get("upper")
        if any(
            isinstance(endpoint, bool)
            or not isinstance(endpoint, (int, float, Fraction))
            or not math.isfinite(float(endpoint))
            for endpoint in (lower, upper)
        ):
            raise TypeError("counterfactual truth owner must return finite interval endpoints")
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
    """Diagnose candidate-family answerability on the full pre-K/M surface.

    Two candidates are observationally equivalent when they expose the same
    query-specific full experiment surface (up to opaque label spelling) and
    every legal passive or hard-do law is identical. Numerical candidates are
    answerable when that evidence determines one exact effect or interval.
    Decision candidates are answerable
    when every indistinguishable world shares at least one zero-regret terminal
    action. Discovery candidates are answerable when the evidence determines
    one exact structural answer.

    The classification is relative to the supplied task family. It is not
    applied as a generator label, filter, or admission check. Per-seed K and M
    are intentionally absent from this optional diagnostic.
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
        elif query_type == "individual_counterfactual_probability":
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

    No thresholds are applied and no planner is used. The optional
    :func:`task_answerability` diagnostic is computed separately and is not a
    generation gate.
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
        "individual_counterfactual_probability",
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
    if len(set(world.variables)) != len(world.variables):
        return False
    if len(world.state_names) != len(world.variables):
        return False
    if any(size <= 0 for size in world.domains):
        return False
    if any(
        len(names) != size for names, size in zip(world.state_names, world.domains, strict=True)
    ):
        return False
    if len(set(world.edges)) != len(world.edges):
        return False
    declared_edges: set[tuple[int, int]] = set()
    for child, parent_nodes in world.parents.items():
        if child < 0 or child >= len(world.variables):
            return False
        if len(parent_nodes) != len(set(parent_nodes)):
            return False
        if any(parent < 0 or parent >= len(world.variables) for parent in parent_nodes):
            return False
        declared_edges.update((parent, child) for parent in parent_nodes)
    if declared_edges != set(world.edges):
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
            if any(
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in row
            ):
                return False
            if all(isinstance(value, Fraction) for value in row):
                if sum(row, Fraction(0)) != 1:
                    return False
            elif abs(math.fsum(float(value) for value in row) - 1.0) > _CPT_VALIDITY_TOLERANCE:
                return False
    return True


def _legal_role_assignments(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    query_type: str,
) -> tuple[dict[str, int], ...]:
    if query_type not in QUERY_TYPES:
        return ()

    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for parent, child in edges:
        adjacency[parent].append(child)

    def path_exists(source: int, target: int, *, minimum_length: int = 1) -> bool:
        stack = [(source, 0)]
        seen_depth: set[tuple[int, int]] = {(source, 0)}
        while stack:
            node, depth = stack.pop()
            for child in adjacency[node]:
                next_depth = depth + 1
                if child == target and next_depth >= minimum_length:
                    return True
                marker = (child, next_depth)
                if marker not in seen_depth:
                    seen_depth.add(marker)
                    stack.append(marker)
        return False

    if query_type in {"ate", "individual_counterfactual_probability", "backadj_minimal_sets"}:
        return tuple(
            {"treatment": source, "outcome": target}
            for source in range(node_count)
            for target in range(node_count)
            if source != target and path_exists(source, target)
        )
    if query_type == "mediator_set":
        return tuple(
            {"treatment": source, "outcome": target}
            for source in range(node_count)
            for target in range(node_count)
            if source != target and path_exists(source, target, minimum_length=2)
        )
    if query_type == "best_intervention":
        return tuple(
            {
                "decision_target": decision_target,
                "outcome": outcome,
            }
            for decision_target in range(node_count)
            for outcome in range(node_count)
            if decision_target != outcome and path_exists(decision_target, outcome)
        )
    return ()


def _descendant_nodes(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    node: int,
) -> frozenset[int]:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for parent, child in edges:
        adjacency[parent].append(child)
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for child in adjacency[current]:
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return frozenset(seen)


def _backdoor_separated_structure(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    treatment: int,
    outcome: int,
    condition: frozenset[int],
) -> bool:
    """Test back-door separation from graph structure alone."""

    backdoor_edges = tuple(edge for edge in edges if edge[0] != treatment)
    parents: list[set[int]] = [set() for _ in range(node_count)]
    for parent, child in backdoor_edges:
        parents[child].add(parent)

    ancestors = {treatment, outcome, *condition}
    stack = list(ancestors)
    while stack:
        child = stack.pop()
        for parent in parents[child]:
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)

    moral_neighbors = {node: set() for node in ancestors}
    for parent, child in backdoor_edges:
        if parent in ancestors and child in ancestors:
            moral_neighbors[parent].add(child)
            moral_neighbors[child].add(parent)
    for child in ancestors:
        relevant_parents = sorted(parents[child] & ancestors)
        for left_index, left in enumerate(relevant_parents):
            for right in relevant_parents[left_index + 1 :]:
                moral_neighbors[left].add(right)
                moral_neighbors[right].add(left)

    reachable = {treatment}
    stack = [treatment]
    while stack:
        current = stack.pop()
        for neighbor in moral_neighbors[current]:
            if neighbor in condition or neighbor in reachable:
                continue
            if neighbor == outcome:
                return False
            reachable.add(neighbor)
            stack.append(neighbor)
    return outcome not in reachable


def _minimum_backdoor_adjustment_size(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    treatment: int,
    outcome: int,
) -> int:
    """Return the exact minimum back-door set size for one ordered pair.

    The treatment's parents always form a valid set, so exact search only needs
    subsets smaller than that instance-specific upper bound.
    """

    treatment_parents = tuple(parent for parent, child in edges if child == treatment)
    upper_bound = len(treatment_parents)
    descendants = _descendant_nodes(node_count, edges, treatment)
    allowed = tuple(
        node
        for node in range(node_count)
        if node not in {treatment, outcome} and node not in descendants
    )
    for size in range(upper_bound):
        for subset in combinations(allowed, size):
            if _backdoor_separated_structure(
                node_count,
                edges,
                treatment,
                outcome,
                frozenset(subset),
            ):
                return size
    return upper_bound


def _sampled_backdoor_complexity(node_count: int, seed: int, query_type: str) -> int:
    """Draw the task's minimum adjustment-set size uniformly over ``0..floor(n/3)``."""

    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 2:
        raise ValueError("node_count must be an integer >= 2")
    seed_id = f"SAMPLED-{seed}-{query_type}"
    return _axis_rng(seed_id, "backdoor-complexity").randrange(node_count // 3 + 1)


def _backdoor_role_endpoints(
    query_type: str,
    role: Mapping[str, int],
) -> tuple[int, int]:
    """Return the ordered intervention/outcome endpoints for one task role."""

    treatment_key = "decision_target" if query_type == "best_intervention" else "treatment"
    return role[treatment_key], role["outcome"]


def _sampled_role_assignments(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    query_type: str,
    seed: int,
) -> tuple[dict[str, int], ...]:
    """Return the role population used by the formal task sampler.

    Every official 8--16-node task family first draws its minimum adjustment-set
    size uniformly from ``0..floor(node_count / 3)``, then conditions only on
    structural roles having that size. Smaller custom grammars retain the
    unstratified role population used by unit tests and explicit diagnostics.
    """

    roles = _legal_role_assignments(node_count, edges, query_type)
    if query_type not in _BACKDOOR_STRATIFIED_QUERY_TYPES or node_count < 8:
        return roles
    target = _sampled_backdoor_complexity(node_count, seed, query_type)
    return tuple(
        role
        for role in roles
        if _minimum_backdoor_adjustment_size(
            node_count,
            edges,
            *_backdoor_role_endpoints(query_type, role),
        )
        == target
    )


def legal_query_anchors(world: WorldSpec, query_type: str) -> tuple[dict[str, int], ...]:
    """Return every structurally legal variable-role assignment for a query.

    Numerical states and decision objectives are sampled only after one role
    assignment is chosen; they are not part of this structural set.
    """

    return _legal_role_assignments(len(world.variables), world.edges, query_type)


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
    """Make exactly the non-anchor variables eligible for K sampling."""

    manipulability = {name: True for name in world.variables}
    for anchor_name in QUERY_TYPES[query_type]["anchors"]:
        node = anchors.get(anchor_name)
        if isinstance(node, int):
            manipulability[world.variables[node]] = False
    return manipulability


def _axis_rng(seed_id: str, axis: str) -> random.Random:
    payload = f"cpt-world-task-v1\0{axis}\0{seed_id}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))


def _sample_task_attributes(
    world: WorldSpec,
    query_type: str,
    roles: Mapping[str, int],
    *,
    seed_id: str,
) -> dict[str, int | str]:
    """Sample only the already-declared post-role task attributes."""

    anchors: dict[str, int | str] = dict(roles)
    rng = _axis_rng(seed_id, "task-attributes")
    if query_type in {"ate", "individual_counterfactual_probability"}:
        treatment = int(roles["treatment"])
        outcome = int(roles["outcome"])
        ordered_pairs = tuple(
            (reference, comparison)
            for reference in range(world.domains[treatment])
            for comparison in range(world.domains[treatment])
            if reference != comparison
        )
        reference, comparison = rng.choice(ordered_pairs)
        if query_type == "ate":
            anchors.update(
                {
                    "baseline_value": reference,
                    "treatment_value": comparison,
                }
            )
        else:
            from .query_truth import interventional_probability

            probabilities = tuple(
                float(
                    interventional_probability(
                        world,
                        {treatment: reference},
                        outcome,
                        state,
                    )
                )
                for state in range(world.domains[outcome])
            )
            draw = rng.random()
            cumulative = 0.0
            factual_outcome_state = len(probabilities) - 1
            for state, probability in enumerate(probabilities):
                cumulative += probability
                if draw < cumulative:
                    factual_outcome_state = state
                    break
            anchors.update(
                {
                    "factual_value": reference,
                    "counterfactual_value": comparison,
                    "factual_outcome_state": factual_outcome_state,
                    "outcome_state": rng.randrange(world.domains[outcome]),
                }
            )
    elif query_type == "best_intervention":
        outcome = int(roles["outcome"])
        anchors.update(
            {
                "objective": rng.choice(("minimize", "maximize")),
                "outcome_state": rng.randrange(world.domains[outcome]),
            }
        )
    return anchors


def _best_intervention_is_observationally_discordant(
    world: WorldSpec,
    anchors: Mapping[str, int | str],
) -> bool:
    """Return whether observational and interventional optimal actions are disjoint."""

    from .query_truth import (
        best_intervention_states,
        worldspec_projected_interventional_distribution,
    )

    decision = int(anchors["decision_target"])
    outcome = int(anchors["outcome"])
    outcome_state = int(anchors["outcome_state"])
    objective = str(anchors["objective"])
    causal_states, _ = best_intervention_states(
        world,
        outcome,
        objective,
        decision,
        outcome_state=outcome_state,
    )

    law = worldspec_projected_interventional_distribution(
        world,
        {},
        (decision, outcome),
    )
    action_mass = [0.0] * world.domains[decision]
    target_mass = [0.0] * world.domains[decision]
    for (action, observed_outcome), raw_probability in law:
        probability = float(raw_probability)
        action_mass[action] += probability
        if observed_outcome == outcome_state:
            target_mass[action] += probability
    observational_values = tuple(
        target_mass[action] / action_mass[action]
        for action in range(world.domains[decision])
    )
    observational_best = (
        min(observational_values)
        if objective == "minimize"
        else max(observational_values)
    )
    observational_states = frozenset(
        action
        for action, value in enumerate(observational_values)
        if value == observational_best
    )
    return observational_states.isdisjoint(causal_states)


def _balanced_proposal_seed(slot: int, attempt: int) -> int:
    """Injectively map a balanced-task slot and rejection attempt to one seed."""

    if slot < 0 or attempt < 0:
        raise ValueError("slot and attempt must be nonnegative")
    diagonal = slot + attempt
    return diagonal * (diagonal + 1) // 2 + attempt


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

    width_rng = _axis_rng(seed_id, "manipulability")
    width = width_rng.randint(1, len(candidates))
    selected = frozenset(width_rng.sample(candidates, width))
    manipulability = {name: name in selected for name in world.variables}
    observation_bandwidth = _axis_rng(seed_id, "observation-bandwidth").randint(
        1, len(world.variables)
    )
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
) -> Mapping[str, Any]:
    """Assemble an anonymous candidate seed or fail closed.

    ``anchors`` selects one of the structurally legal query assignments from
    :func:`legal_query_anchors`. If omitted, the first legal assignment is
    used. Masks default to the anchor-minimal rule from
    :func:`default_manipulability`; pinned seed masks can be passed explicitly.
    ``observation_bandwidth`` is optional for legacy/manual seeds and fixed in
    every seed emitted by the main sampler.
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
    legal_anchors = legal_query_anchors(world, query_type)
    if anchors is None:
        if not legal_anchors:
            raise ValueError(f"{seed_id}: query has no legal anchor assignment")
        selected_anchors = dict(legal_anchors[0])
    else:
        selected_anchors = dict(anchors)
        structural_anchors = {
            name: selected_anchors.get(name) for name in QUERY_TYPES[query_type]["anchors"]
        }
        if structural_anchors not in [dict(item) for item in legal_anchors]:
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
        "individual_counterfactual_probability",
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
        if query_type == "ate":
            treatment_name = world.variables[int(selected_anchors["treatment"])]
            baseline_value = int(selected_anchors.get("baseline_value", 0))
            treatment_value = int(selected_anchors.get("treatment_value", 1))
            if baseline_value == treatment_value:
                raise ValueError(f"{seed_id}: treatment and baseline values must differ")
            query_visible["treatment_value"] = visible_state(treatment_name, treatment_value)
            query_visible["baseline_value"] = visible_state(treatment_name, baseline_value)
        if query_type == "individual_counterfactual_probability":
            treatment_name = world.variables[int(selected_anchors["treatment"])]
            factual_value = int(selected_anchors.get("factual_value", 0))
            counterfactual_value = int(selected_anchors.get("counterfactual_value", 1))
            factual_outcome_state = int(selected_anchors.get("factual_outcome_state", 0))
            outcome_state = int(selected_anchors.get("outcome_state", 1))
            if factual_value == counterfactual_value:
                raise ValueError(
                    f"{seed_id}: factual and counterfactual treatment values must differ"
                )
            query_visible.update(
                {
                    "factual_value": visible_state(treatment_name, factual_value),
                    "counterfactual_value": visible_state(treatment_name, counterfactual_value),
                    "factual_outcome_state": visible_state(outcome_name, factual_outcome_state),
                    "outcome_state": visible_state(outcome_name, outcome_state),
                }
            )
    elif query_type == "best_intervention":
        outcome_name = world.variables[int(selected_anchors["outcome"])]
        outcome_state = int(selected_anchors.get("outcome_state", 1))
        query_visible.update(
            {
                "decision_target": anchor_label("decision_target"),
                "outcome": anchor_label("outcome"),
                "objective": str(selected_anchors.get("objective", "minimize")),
                "outcome_state": visible_state(outcome_name, outcome_state),
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


def assemble_sampled_anchor_tasks(
    grammar: WorldGrammar,
    sample_index: int,
    query_type: str,
    anchor_index: int,
    *,
    hiding: str | object = "mechanism_hidden",
) -> tuple[tuple[WorldSpec, Mapping[str, Any]], ...]:
    """Build the existing main-pipeline task for one legal query anchor.

    This is the single-item composition seam used by both exhaustive seed
    expansion and evaluation schedules.  World, CPT, interaction-surface,
    renderer, and task semantics remain owned by their existing functions.
    """

    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("sample_index must be a nonnegative integer")
    if isinstance(anchor_index, bool) or not isinstance(anchor_index, int) or anchor_index < 0:
        raise ValueError("anchor_index must be a nonnegative integer")
    if query_type not in {
        "ate",
        "individual_counterfactual_probability",
        "backadj_minimal_sets",
        "best_intervention",
        "mediator_set",
    }:
        raise ValueError(f"generic sampler does not admit query type: {query_type}")

    task_world = sample_task_world(grammar, sample_index, query_type)
    legal_anchors = _sampled_role_assignments(
        len(task_world.variables),
        task_world.edges,
        query_type,
        sample_index,
    )
    if anchor_index >= len(legal_anchors):
        raise ValueError("anchor_index is outside the legal query anchors")
    roles = legal_anchors[anchor_index]
    compatible_heads = tuple(
        task_head for task_head in TASK_HEADS if supports_task(query_type, task_head)
    )
    if len(compatible_heads) != 1:
        raise ValueError("query type must resolve to exactly one task head")
    task_head = compatible_heads[0]
    base_seed_id = f"SAMPLED-{sample_index}-{query_type}-{task_head}-a{anchor_index}"
    anchors = _sample_task_attributes(
        task_world,
        query_type,
        roles,
        seed_id=base_seed_id,
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
    render_seed_prompt(base_assembled)
    return ((task_world, base_assembled),)


def iter_sampled_seeds(
    grammar: WorldGrammar,
    *,
    query_types: tuple[str, ...],
    start_seed: int = 0,
    count: int = 1,
    hiding: str | object = "mechanism_hidden",
    best_intervention_balance_start: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Yield one sampled task per requested family and output slot.

    Linear expansion step:

    ``node count -> eligible world -> one uniform role -> states -> K/M``.

    The caller explicitly names a nonempty subset of the five implemented
    query types. Node count stays fixed during structural eligibility sampling.
    The main pipeline samples one legal variable-role assignment uniformly,
    then the declared state anchors, a nonempty K-subset, and an independent
    seed-fixed M. Four families never filter numerical answers. Best
    intervention uses deterministic rejection over complete task proposals so
    the first output slot in every five-slot block is observationally
    concordant and the other four are observationally discordant. By default,
    ``start_seed`` also owns that output-slot phase. Streaming callers that
    share world seeds across task families may provide
    ``best_intervention_balance_start`` to keep the best-intervention slot
    independent of skipped or resampled rows. This makes every consecutive
    aligned five-slot block exactly 1:4 without changing either conditional
    proposal distribution.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if best_intervention_balance_start is not None and best_intervention_balance_start < 0:
        raise ValueError("best_intervention_balance_start must be nonnegative")
    admitted_query_types = frozenset(TASK_FAMILY_QUERY_TYPES)
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
    for output_offset, sample_index in enumerate(range(start_seed, start_seed + count)):
        for query_type in query_types:
            attempt = 0
            while True:
                proposal_index = (
                    _balanced_proposal_seed(sample_index, attempt)
                    if query_type == "best_intervention"
                    else sample_index
                )
                task_world = sample_task_world(grammar, proposal_index, query_type)
                legal_roles = _sampled_role_assignments(
                    len(task_world.variables),
                    task_world.edges,
                    query_type,
                    proposal_index,
                )
                role_seed_id = f"SAMPLED-{proposal_index}-{query_type}"
                anchor_index = _axis_rng(role_seed_id, "variable-role").randrange(
                    len(legal_roles)
                )
                if query_type == "best_intervention":
                    task_head = "decision"
                    anchors = _sample_task_attributes(
                        task_world,
                        query_type,
                        legal_roles[anchor_index],
                        seed_id=(
                            f"SAMPLED-{proposal_index}-{query_type}-{task_head}"
                            f"-a{anchor_index}"
                        ),
                    )
                    discordant = _best_intervention_is_observationally_discordant(
                        task_world,
                        anchors,
                    )
                    balance_slot = (
                        sample_index
                        if best_intervention_balance_start is None
                        else best_intervention_balance_start + output_offset
                    )
                    desired_discordant = balance_slot % 5 != 0
                    if discordant != desired_discordant:
                        attempt += 1
                        continue
                generated.extend(
                    assemble_sampled_anchor_tasks(
                        grammar,
                        proposal_index,
                        query_type,
                        anchor_index,
                        hiding=hiding,
                    )
                )
                break
    return tuple(seed for _, seed in generated)
