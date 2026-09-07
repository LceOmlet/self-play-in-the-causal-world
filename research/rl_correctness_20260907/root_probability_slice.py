"""Certified strong-reversal slices in a binary root CPT parameter.

Offline mathematical reference, not a production sampler or training policy.
All input coefficients are rational. SymPy isolates polynomial roots exactly;
the unclassified root boxes have an explicitly bounded total probability mass.
"""
from fractions import Fraction as F
from itertools import combinations

import sympy as sp

RHO = sp.Symbol('rho')


def affine(endpoints):
    left, right = map(F, endpoints)
    return left, right - left


def value(line, x):
    return line[0] + line[1] * x


def multiply(left, right):
    return (left[0] * right[0], left[0] * right[1] + left[1] * right[0], left[1] * right[1])


def subtract(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def relation(causal, joint, mass, rho, objective, threshold):
    mu = [value(line, rho) for line in causal]
    nu = [value(c, rho) / value(a, rho) for c, a in zip(joint, mass, strict=True)]
    optimum = max if objective == 'maximize' else min
    mu_best, nu_best = optimum(mu), optimum(nu)
    causal_actions = {i for i, v in enumerate(mu) if v == mu_best}
    observed_actions = {i for i, v in enumerate(nu) if v == nu_best}
    observed_causal = optimum(mu[i] for i in observed_actions)
    regret = abs(mu_best - observed_causal)
    return causal_actions.isdisjoint(observed_actions) and regret >= threshold


def certified_slice(causal_endpoints, joint_endpoints, mass_endpoints, *, threshold, objective='maximize', digits=24):
    """Enclose the exact Lebesgue mass of a root's admissible parameter set."""
    threshold = F(threshold)
    assert threshold > 0 and objective in ('maximize', 'minimize')
    assert len(causal_endpoints) == len(joint_endpoints) == len(mass_endpoints)
    assert all(F(p) > 0 for endpoints in mass_endpoints for p in endpoints)
    causal, joint, mass = [list(map(affine, endpoints)) for endpoints in
                           (causal_endpoints, joint_endpoints, mass_endpoints)]
    polynomials = []
    for left, right in combinations(range(len(causal)), 2):
        delta = subtract(causal[left], causal[right])
        polynomials += [delta, (delta[0] - threshold, delta[1]), (delta[0] + threshold, delta[1])]
        polynomials.append(subtract(multiply(joint[left], mass[right]), multiply(joint[right], mass[left])))
    boxes = []
    effective = 0
    for coefficients in polynomials:
        expression = sum(sp.Rational(c.numerator, c.denominator) * RHO**i for i, c in enumerate(coefficients))
        if expression == 0:
            continue  # Permanent ties are handled by exact classification.
        effective += 1
        for (lower, upper), _multiplicity in sp.Poly(expression, RHO).intervals(eps=sp.Rational(1, 10**digits)):
            lo, hi = max(F(0), F(lower)), min(F(1), F(upper))
            if lo <= hi:
                boxes.append((lo, hi))
    merged = []
    for lo, hi in sorted(boxes):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    accepted, rejected = [], []
    cursor = F(0)
    for lo, hi in [*merged, (F(1), F(1))]:
        if cursor < lo:
            interval = (cursor, lo)
            middle = (cursor + lo) / 2
            destination = accepted if relation(causal, joint, mass, middle, objective, threshold) else rejected
            destination.append(interval)
        cursor = hi
    lower_mass = sum((hi-lo for lo, hi in accepted), F(0))
    uncertain = sum((hi-lo for lo, hi in merged), F(0))
    return {'accepted': accepted, 'rejected': rejected, 'root_boxes': merged,
            'mass_lower': lower_mass, 'mass_upper': lower_mass + uncertain,
            'uncertain_mass': uncertain, 'polynomials': effective,
            'isolator_width': F(1, 10**digits), 'includes_boundary_atoms': False}
