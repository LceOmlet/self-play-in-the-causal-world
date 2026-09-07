"""Check the binary coefficient change of variables against actual ET-V2."""
from itertools import combinations, product
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
project = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent/'rl_algorithm_fix_20260906/project'
package = types.ModuleType('cpt_world')
package.__path__ = [str(project/'src/cpt_world')]
sys.modules['cpt_world'] = package
from cpt_world import world_space as w

checks = []
for parents in (1, 2, 3):
    supports = [tuple((p,) for p in range(parents))]
    if parents >= 2:
        supports.append(tuple(combinations(range(parents), 1)) + ((0, 1),))
        supports.append(tuple(subset for order in range(1, parents+1) for subset in combinations(range(parents), order)))
    for support in dict.fromkeys(supports):
        contexts = list(product((-1, 1), repeat=parents))
        blocks = [tuple(tuple(child*math.prod(context[p] for p in subset)/math.sqrt(2**(parents+1))
                              for child in (-1, 1)) for context in contexts) for subset in support]
        for index in range(1, 26):
            shares = [1 + (index*(i+3)) % 17 for i in range(len(support))]
            shares = [weight/sum(shares) for weight in shares]
            signs = [(-1)**(index+i) for i in range(len(support))]
            signed = [tuple(tuple(sign*v for v in row) for row in block) for sign, block in zip(signs, blocks, strict=True)]
            direction = w._combine_effect_blocks(signed, shares)
            scale = w._contextual_parent_pair_score_scale(direction, (2,)*parents)
            energy_order = sum(len(subset)*share for subset, share in zip(support, shares, strict=True))
            assert abs(scale-math.sqrt(parents/energy_order)) < 1e-13
            amplitude = math.sqrt(3)*index/26
            base = (index/27, 1-index/27)
            rows = w._exponential_tilt_rows(base, direction, amplitude, score_scale=scale)
            logits = [math.log(row[1]/row[0])-math.log(base[1]/base[0]) for row in rows]
            coefficients = [sum(logit*math.prod(context[p] for p in subset) for logit, context in zip(logits, contexts, strict=True))/len(contexts) for subset in support]
            expected = [2*math.sqrt(parents)*amplitude*sign*math.sqrt(share)/math.sqrt(energy_order) for share, sign in zip(shares, signs, strict=True)]
            error = max(abs(a-b) for a, b in zip(coefficients, expected, strict=True))
            assert error < 1e-12
            weighted_norm_squared = sum(len(subset)*a*a for subset, a in zip(support, coefficients, strict=True))
            assert abs(weighted_norm_squared-4*parents*amplitude**2) < 1e-11
            checks.append({'parents': parents, 'active_blocks': len(support), 'max_coefficient_error': error})

# Exact Jacobian identities for the two smallest nontrivial coefficient spaces.
import sympy as sp
jacobians = []
for dimension in (2, 3):
    a = sp.symbols(f'a0:{dimension}', positive=True)
    radius = sp.sqrt(sum(v*v for v in a))
    transformed = sp.Matrix([radius, *(v*v/radius**2 for v in a[:-1])])
    determinant = sp.simplify(transformed.jacobian(a).det())
    expected = 2**(dimension-1)*sp.prod(a)/radius**(2*dimension-1)
    assert sp.simplify(determinant**2-expected**2) == 0
    jacobians.append(dimension)
result = {'passed': True, 'actual_kernel_checks': len(checks),
          'max_coefficient_error': max(c['max_coefficient_error'] for c in checks),
          'symbolic_jacobian_dimensions': jacobians, 'integrated_into_sampler': False}
(ROOT/'et-v2-binary-density-acceptance.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
