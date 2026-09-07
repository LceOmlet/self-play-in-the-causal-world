"""Assemble an isolated kernel candidate; never writes the production tree."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'rl_algorithm_fix_20260906/project'
DEST = ROOT / 'candidate/src/cpt_world'
shutil.copytree(PROJECT / 'src/cpt_world', DEST, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
source = (ROOT.parent / 'kernel_integration_20260907/production_snapshot/src/cpt_world/counterfactual_solver.py').read_text(encoding='utf-8')
source = source.replace('from dataclasses import dataclass\n', 'from dataclasses import dataclass, replace\n')
helper = (ROOT / 'response_envelope.py').read_text(encoding='utf-8')
helper = helper[helper.index('@dataclass(frozen=True)'):]
helper = helper.replace('Envelope', '_ResponseTransportEnvelope')
helper = helper.replace('def response_envelope(', 'def _indirect_response_transport_envelope(')
helper = helper.replace('    affected = q._descendants', '''    """Outer bounds from complete terminal responses, not pairwise tables.

    For a fixed terminal response r, a mediator transport contributes a
    rectangle event. Its extrema are max(0,u(A_r)+v(B_r)-1) and
    min(u(A_r),v(B_r)). Average these valid bounds over one feasible joint
    terminal response distribution; aggregate shared assignments before the
    LP when they read the same terminal contexts. Independent per-response
    transports are a relaxation only. Common-transport feasibility can then
    construct independent SCM mechanisms attaining these bounds. Otherwise
    the original SCM optimizer must supply its globally certified endpoints.
    """
    from math import prod

    from .query_truth import worldspec_projected_interventional_distribution

    affected = q._descendants''')
helper = helper.replace('q._descendants', '_descendants').replace('q._ancestors', '_ancestors')
helper = helper.replace('q.worldspec_projected_interventional_distribution', 'worldspec_projected_interventional_distribution')
helper = helper.replace('s._', '_')
source = source.replace('class _ExactPricedResponseLP:', helper + '\n\nclass _ExactPricedResponseLP:')
needle = '    partial_terminal = _partially_attainable_terminal_bounds(\n'
assert source.count(needle) == 1
replacement = '''    envelope = _indirect_response_transport_envelope(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_events=resolved_events,
        time_limit_seconds=time_limit_seconds,
    )
    envelope_build_seconds = 0.0
    envelope_solve_seconds = 0.0
    if envelope is not None:
        if envelope.attained_lower is not None and envelope.attained_upper is not None:
            return CounterfactualBoundsResult(
                lower=envelope.attained_lower,
                upper=envelope.attained_upper,
                build_seconds=envelope.build_seconds,
                solve_seconds=envelope.solve_seconds,
                affected_nodes=2,
                pair_kernel_entries=0,
                generated_columns=0,
                response_blocks=envelope.response_blocks,
                dynamic_response_blocks=0,
                max_response_contexts=next(
                    world.domains[node]
                    for node in (_descendants(world, treatment) & _ancestors(world, outcome))
                ),
                auxiliary_variables=0,
                backend="one_mediator_response_envelope_attainment",
            )
        outward = 10.0 * _SCIP_NUMERICAL_TOLERANCE
        target_outer_bounds = (
            max(target_outer_bounds[0], envelope.lower - outward),
            min(target_outer_bounds[1], envelope.upper + outward),
        )
        if target_outer_bounds[0] > target_outer_bounds[1]:
            raise RuntimeError("response envelope contradicts existing outer bounds")
        envelope_build_seconds = envelope.build_seconds
        envelope_solve_seconds = envelope.solve_seconds
        if time_limit_seconds is not None:
            time_limit_seconds -= envelope_build_seconds + envelope_solve_seconds
            if time_limit_seconds <= 0.0:
                raise RuntimeError("response envelope exhausted the endpoint time limit")

''' + needle
source = source.replace(needle, replacement)
source = source.replace('''    if partial_terminal is not None:
        return partial_terminal
''', '''    if partial_terminal is not None:
        return replace(
            partial_terminal,
            build_seconds=partial_terminal.build_seconds + envelope_build_seconds,
            solve_seconds=partial_terminal.solve_seconds + envelope_solve_seconds,
        )
''')
source = source.replace('''        build_seconds=lower_build_seconds + restart_seconds,
        solve_seconds=lower_seconds + upper_seconds,
''', '''        build_seconds=lower_build_seconds + restart_seconds + envelope_build_seconds,
        solve_seconds=lower_seconds + upper_seconds + envelope_solve_seconds,
''')
(DEST / 'counterfactual_solver.py').write_text(source, encoding='utf-8')
print(DEST)
