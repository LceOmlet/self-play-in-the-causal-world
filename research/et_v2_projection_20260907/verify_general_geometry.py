"""Verify the general mixed-cardinality ET-V2 geometry and density Jacobians."""

import json
import math
import random
import sys
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cpt_world import world_space as w  # noqa: E402


def log_effect_density(radii, dimensions, weights, parents, cells):
    """Density on the active score subspace, in orthonormal table coordinates."""
    k = len(radii)
    radius = math.sqrt(sum(value * value for value in radii))
    weighted_radius = math.sqrt(
        sum(weight * value * value for weight, value in zip(weights, radii, strict=True))
    )
    assert 0 < weighted_radius < math.sqrt(6 * parents * cells)
    assert all(value > 0 for value in radii)
    log_areas = [
        math.log(2) + dimension / 2 * math.log(math.pi) - math.lgamma(dimension / 2)
        for dimension in dimensions
    ]
    return (
        math.lgamma(k)
        + (k - 1) * math.log(2)
        - 0.5 * math.log(6 * parents * cells)
        - sum(log_areas)
        + math.log(weighted_radius)
        - 2 * k * math.log(radius)
        + sum(
            (2 - dimension) * math.log(value)
            for dimension, value in zip(dimensions, radii, strict=True)
        )
    )


def integer_contrasts(domain):
    return [
        tuple(1 if state < j else -j if state == j else 0 for state in range(domain))
        for j in range(1, domain)
    ]


# Exact Gram and contextual pair-contrast identities on integer tensor bases.
exact_checks = []
for domains, child in (((2, 3), 3), ((2, 3, 2), 4)):
    contexts = tuple(product(*(range(domain) for domain in domains)))
    m, cells = len(domains), math.prod(domains) * child
    blocks = []
    for size in range(1, m + 1):
        for support in combinations(range(m), size):
            basis = []
            for vectors in product(
                *(integer_contrasts(domains[i]) for i in support), integer_contrasts(child)
            ):
                vector = tuple(
                    math.prod(vectors[j][context[i]] for j, i in enumerate(support))
                    * vectors[-1][state]
                    for context in contexts
                    for state in range(child)
                )
                basis.append(vector)
            assert len(basis) == (child - 1) * math.prod(domains[i] - 1 for i in support)
            weight = sum((F(domains[i], domains[i] - 1) for i in support), F(0))
            for vector in basis:
                norm = sum(value * value for value in vector)
                contrasts = F(0)
                for axis, domain in enumerate(domains):
                    total = 0
                    for row, context in enumerate(contexts):
                        for state in range(context[axis] + 1, domain):
                            right = list(context)
                            right[axis] = state
                            other_row = contexts.index(tuple(right))
                            total += sum(
                                (vector[row * child + c] - vector[other_row * child + c]) ** 2
                                for c in range(child)
                            )
                    count = math.prod(domains) // domain * math.comb(domain, 2) * child
                    contrasts += F(total, count * m)
                assert contrasts == F(2, m * cells) * weight * norm
            blocks.append((support, basis))
    flattened = [(support, vector) for support, basis in blocks for vector in basis]
    for left in range(len(flattened)):
        for right in range(left):
            assert (
                sum(a * b for a, b in zip(flattened[left][1], flattened[right][1], strict=True))
                == 0
            )
    exact_checks.append(
        {
            "domains": domains,
            "child": child,
            "basis_vectors": len(flattened),
            "mutually_orthogonal": True,
            "weighted_contrast_identity": True,
        }
    )

# A direct symbolic Cartesian Jacobian for unequal block dimensions (1, 2).
t, e, angle = sp.symbols("t e angle", positive=True)
weight = 2 * e + sp.Rational(3, 2) * (1 - e)
radius = sp.sqrt(48) * t / sp.sqrt(weight)
vector = sp.Matrix(
    [
        radius * sp.sqrt(e),
        radius * sp.sqrt(1 - e) * sp.cos(angle),
        radius * sp.sqrt(1 - e) * sp.sin(angle),
    ]
)
determinant = sp.trigsimp(vector.jacobian((t, e, angle)).det())
expected = sp.sqrt(48) / sp.sqrt(weight) * radius**2 / (2 * sp.sqrt(e))
assert sp.simplify(determinant**2 / expected**2) == 1

# Three active scalar blocks are the independent binary special case.
e1, e2 = sp.symbols("e1 e2", positive=True)
shares = (e1, e2, 1 - e1 - e2)
weight3 = 2 * e1 + 2 * e2 + 4 * shares[2]
radius3 = sp.sqrt(32) * t / sp.sqrt(weight3)
vector3 = sp.Matrix([radius3 * sp.sqrt(share) for share in shares])
determinant3 = vector3.jacobian((t, e1, e2)).det()
expected3 = sp.sqrt(32) / sp.sqrt(weight3) * radius3**2 / (4 * sp.sqrt(sp.prod(shares)))
assert sp.simplify(determinant3**2 / expected3**2) == 1

# Base-simplex density in orthonormal CLR coordinates: sqrt(child)*prod(base).
base_checks = []
for child in (2, 3, 4, 5):
    probabilities = sp.Matrix([sp.Rational(i + 1, child * (child + 1) // 2) for i in range(child)])
    helmert = sp.Matrix.hstack(
        *(
            sp.Matrix(vector) / sp.sqrt(sum(v * v for v in vector))
            for vector in integer_contrasts(child)
        )
    )
    derivative = (sp.diag(*probabilities) - probabilities * probabilities.T) * helmert
    assert (
        sp.simplify(derivative[: child - 1, :].det() ** 2 - child * sp.prod(probabilities) ** 2)
        == 0
    )
    base_checks.append(child)

# Compare the full declared geometry to the actual ET-V2 functions.
actual_checks, max_scale_error, max_logscore_error = 0, 0.0, 0.0
for domains in ((2, 3), (3, 4), (2, 3, 2), (2, 2, 2)):
    for child in (2, 3, 5):
        for full_support in (False, True):
            support = tuple(
                s
                for size in range(1, len(domains) + 1)
                for s in combinations(range(len(domains)), size)
                if full_support or size == 1
            )
            for seed in range(4):
                rng = random.Random(713 + seed)
                blocks = [w._project_parent_subset_effect(domains, s, child, rng) for s in support]
                shares = w._simplex_uniform(len(blocks), rng)
                direction = w._combine_effect_blocks(blocks, shares)
                weights = [sum(domains[i] / (domains[i] - 1) for i in s) for s in support]
                cells = math.prod(domains) * child
                scale = w._contextual_parent_pair_score_scale(direction, domains)
                expected_scale = math.sqrt(
                    2 * len(domains) / sum(k * e for k, e in zip(weights, shares, strict=True))
                )
                max_scale_error = max(max_scale_error, abs(scale - expected_scale))
                assert abs(scale - expected_scale) < 1e-12
                amplitude = 0.2 + 0.3 * seed
                base = w._simplex_uniform(child, rng)
                cpt = w._exponential_tilt_rows(base, direction, amplitude, score_scale=scale)
                centered_logs = [
                    tuple(math.log(p) - sum(math.log(q) for q in row) / child for p in row)
                    for row in cpt
                ]
                intercept = [sum(row[j] for row in centered_logs) / len(cpt) for j in range(child)]
                for row, effect in zip(centered_logs, direction, strict=True):
                    for j in range(child):
                        error = abs(
                            row[j] - intercept[j] - amplitude * scale * math.sqrt(cells) * effect[j]
                        )
                        max_logscore_error = max(max_logscore_error, error)
                        assert error < 1e-11
                actual_checks += 1

# Recover the earlier density in binary Walsh-logit coordinates exactly in scale.
for m in (1, 2, 3):
    support = tuple(s for size in range(1, m + 1) for s in combinations(range(m), size))
    k = len(support)
    a = [0.05 * (i + 1) for i in range(k)]
    cells = 2 ** (m + 1)
    b = [math.sqrt(cells) / 2 * value for value in a]
    transformed = log_effect_density(
        b, [1] * k, [2 * len(s) for s in support], m, cells
    ) + k * math.log(math.sqrt(cells) / 2)
    r2 = sum(value * value for value in a)
    weighted = math.sqrt(sum(len(s) * value * value for s, value in zip(support, a, strict=True)))
    earlier = (
        math.lgamma(k)
        - math.log(4 * math.sqrt(3 * m))
        + math.log(weighted)
        + sum(math.log(value) for value in a)
        - k * math.log(r2)
    )
    assert abs(transformed - earlier) < 1e-12

report = {
    "passed": True,
    "exact_basis_checks": exact_checks,
    "symbolic_cartesian_jacobians": [[1, 2], [1, 1, 1]],
    "base_clr_jacobian_dimensions": base_checks,
    "actual_kernel_cases": actual_checks,
    "max_scale_error": max_scale_error,
    "max_logscore_error": max_logscore_error,
    "binary_density_reduction": True,
    "scope": (
        "General density proven by change of variables; finite symbolic/exact and "
        "actual-kernel checks support it. No new sampler law."
    ),
}
(ROOT / "general-geometry-acceptance.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
