"""Build the proof-backed exact structural owner from the accepted baseline."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'rl_algorithm_fix_20260906/project'
DEST = ROOT / 'diagonal_candidate/src/cpt_world'
shutil.copytree(PROJECT / 'src/cpt_world', DEST, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
source = (ROOT.parent / 'kernel_integration_20260907/production_snapshot/src/cpt_world/counterfactual_solver.py').read_text(encoding='utf-8')
helper = (ROOT / 'indirect_diagonal_solver.py').read_text(encoding='utf-8')
helper = helper[helper.index('def _indirect_mediator_joint_bounds('):]
helper = helper.replace('    states = world.domains[mediator]', '''    from math import prod

    from .query_truth import worldspec_projected_interventional_distribution

    states = world.domains[mediator]''', 1)
helper = helper.replace('s._', '_').replace('s.CounterfactualBoundsResult', 'CounterfactualBoundsResult')
helper = helper.replace('q.worldspec_projected_interventional_distribution', 'worldspec_projected_interventional_distribution')
helper = helper.replace('    world,\n    treatment,\n    outcome,',
                        '    world: WorldSpec,\n    treatment: int,\n    outcome: int,', 1)
helper = helper.replace('    mediator,\n', '    mediator: int,\n', 1)
helper = helper.replace('    baseline_value,\n', '    baseline_value: int,\n', 1)
helper = helper.replace('    treatment_value,\n', '    treatment_value: int,\n', 1)
helper = helper.replace('    outcome_events,\n', '    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],\n', 1)
helper = helper.replace('    time_limit_seconds,\n):', '    time_limit_seconds: float | None,\n) -> CounterfactualBoundsResult | None:', 1)
source = source.replace('class _ExactPricedResponseLP:', helper + '\n\nclass _ExactPricedResponseLP:')
needle = '''    mediator = affected[0]
    if world.parents[mediator] != (treatment,):'''
assert source.count(needle) == 1
source = source.replace(needle, '''    mediator = affected[0]
    if treatment not in world.parents[outcome] and mediator in world.parents[outcome]:
        return _indirect_mediator_joint_bounds(
            world,
            treatment,
            outcome,
            mediator=mediator,
            baseline_value=baseline_value,
            treatment_value=treatment_value,
            outcome_events=outcome_events,
            time_limit_seconds=time_limit_seconds,
        )
    if world.parents[mediator] != (treatment,):''')
source = source.replace('''    The supported twin graph is ``X -> M -> Y`` with the direct ``X -> Y``
''', '''    Pure indirect chains first use the proven diagonal-transport owner when
    its structural conditions hold. The direct case is ``X -> M -> Y`` with
    the direct ``X -> Y``
''')
(DEST / 'counterfactual_solver.py').write_text(source, encoding='utf-8')
print(DEST)
