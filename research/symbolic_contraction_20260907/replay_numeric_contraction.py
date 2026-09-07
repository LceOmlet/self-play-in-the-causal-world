"""Read-only production-owner replay. Timers do not change any solver options."""
import dataclasses
import functools
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('CPT_CF_AUDIT_ROOT', '/home/chen/runs/symbolic-contraction-20260907'))
INPUTS = Path(os.environ.get('CPT_CF_INPUT_ROOT', '/home/chen/runs/kernel-root-cause-20260907'))
OLD = Path('/home/chen/runs/environment-validation-20260907')


def emit(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, default=str) + '\n')


def worker(index, variant):
    import faulthandler
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    from cpt_world import query_truth as q
    from cpt_world import counterfactual_solver as s
    world, seed = pickle.loads((INPUTS / f'task-{index}.pkl').read_bytes())
    events = ROOT / variant / f'{index}.events.jsonl'
    start = time.perf_counter()
    totals = {}
    models = []
    stack = (ROOT / variant / f'{index}.stack.txt').open('w')
    if variant == 'baseline':
        faulthandler.dump_traceback_later(2.0, repeat=True, file=stack)

    def wrap(owner, name, verbose=True):
        original = getattr(owner, name)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            t = time.perf_counter()
            if verbose:
                event = {'event': 'enter', 'name': name, 't': t-start}
                if name == '_eliminate_factor_tokens':
                    w, _, factors, tokens, local = args[:5]
                    union = {v for f in [*factors, local] if any(k in f.scope for k in tokens) for v in f.scope} - set(tokens)
                    event['tokens'] = tokens
                    event['cells_upper'] = __import__('math').prod(w.domains[v[0]] for v in union)
                emit(events, event)
            try:
                result = original(*args, **kwargs)
                if verbose and isinstance(result, s.CounterfactualBoundsResult):
                    emit(events, {'event': 'certificate', 'name': name, **dataclasses.asdict(result)})
                return result
            finally:
                elapsed = time.perf_counter()-t
                count, seconds = totals.get(name, (0, 0.0))
                totals[name] = (count+1, seconds+elapsed)
                if verbose:
                    extra = {}
                    if name in {'__init__', 'optimize'} and args and isinstance(args[0], s._SparseResponseModel):
                        m = args[0]
                        if hasattr(m, 'model'):
                            extra = {'vars': m.model.getNVars(), 'cons': m.model.getNConss()}
                        if hasattr(m, 'auxiliary_values'):
                            extra['auxiliaries'] = len(m.auxiliary_values)
                            extra['kernels'] = len(m.kernel_cache)
                            extra['blocks'] = len(m.pricing_blocks)
                            extra['dynamic_blocks'] = len(m.dynamic_pricing_blocks)
                            extra['response_contexts'] = [len(b.contexts) for b in m.pricing_blocks]
                        if name == 'optimize':
                            extra.update(primal=m.model.getPrimalbound(), dual=m.model.getDualbound(), nodes=m.model.getNNodes())
                    emit(events, {'event': 'exit', 'name': name, 't': time.perf_counter()-start, 'seconds': elapsed, **extra})
        setattr(owner, name, observed)

    for name in ['_variable_elimination_distribution']:
        wrap(q, name)
    for name in ['_one_mediator_joint_bounds', '_two_mediator_joint_bounds',
                 '_direct_treatment_terminal_bounds', '_root_separator_bounds',
                 '_partially_attainable_terminal_bounds', '_eliminate_factor_tokens',
                 '_solve_sparse_two_world_event_bounds']:
        wrap(s, name)
    for name in ['_exact_pairwise_map', '_exact_pairwise_min_sum']:
        wrap(s, name, verbose=False)
    for name in ['__init__', '_prepare_context_graphs', '_build_response_couplings', '_build_twin_probability', '_compile_numeric_terminal_factors', 'optimize']:
        if hasattr(s._SparseResponseModel, name):
            wrap(s._SparseResponseModel, name)
    try:
        truth = q.compute_query_truth(world, seed, counterfactual_endpoint_time_limit_seconds=5.0)
        result = {'status': 'ok', 'truth': truth}
    except Exception as e:
        result = {'status': type(e).__name__, 'error': str(e)}
    finally:
        faulthandler.cancel_dump_traceback_later()
    result.update(index=index, seconds=time.perf_counter()-start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, totals=totals)
    emit(events, {'event': 'result', **result})


def prepare():
    from cpt_world import world_space as w
    candidates = [json.loads(line) for cohort in ['cohort-a', 'cohort-b'] for line in (OLD / cohort / 'counterfactual-candidates.jsonl').read_text().splitlines()]
    for i, c in enumerate(candidates):
        sid = c['seed_id']
        idx, anchor = int(sid.split('-')[1]), int(sid.rsplit('-a', 1)[1])
        task_path = ROOT / f'task-{i}.pkl'
        if not task_path.exists():
            ((world, seed),) = w.assemble_sampled_anchor_tasks(w.WorldGrammar(), idx, 'individual_counterfactual_probability', anchor)
            task_path.write_bytes(pickle.dumps((world, seed)))
        if i % 10 == 0:
            print('prepared', i, flush=True)
    (ROOT / 'candidates.json').write_text(json.dumps(candidates, indent=2))
    print('prepared all', len(candidates), flush=True)


def run(variant):
    (ROOT / variant).mkdir(exist_ok=True)
    candidates = json.loads((INPUTS / 'candidates.json').read_text())
    indices = set(map(int, os.environ['CASE_INDICES'].split(','))) if os.environ.get('CASE_INDICES') else None
    for i, c in enumerate(candidates):
        if indices is not None and i not in indices:
            continue
        if (ROOT / variant / f'{i}.parent.json').exists():
            continue
        start = time.perf_counter()
        log = (ROOT / variant / f'{i}.stderr.txt').open('w')
        child = subprocess.Popen([sys.executable, __file__, 'worker', str(i), variant], stdout=log, stderr=log)
        rss = 0
        while child.poll() is None and time.perf_counter()-start < 11.0:
            try:
                for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                    if line.startswith('VmHWM:'):
                        rss = max(rss, int(line.split()[1]))
            except FileNotFoundError:
                pass
            time.sleep(.05)
        timeout = child.poll() is None
        if timeout:
            child.terminate()
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        result = {'index': i, 'seed_id': c['seed_id'], 'prior_status': c['status'], 'hard_timeout': timeout, 'seconds': time.perf_counter()-start, 'rss_kib': rss, 'exit_code': child.returncode}
        (ROOT / variant / f'{i}.parent.json').write_text(json.dumps(result))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ROOT.mkdir(exist_ok=True)
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'worker':
        worker(int(sys.argv[2]), sys.argv[3])
    else:
        run(sys.argv[2])
