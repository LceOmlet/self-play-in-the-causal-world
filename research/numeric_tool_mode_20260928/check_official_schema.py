"""CPU-only schema check with the actual pinned Transformers and causal owner.

No model is loaded and no Trainer method is copied or replaced.
"""
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
from transformers.utils import get_json_schema
from cpt_world import CPTWorldEnvironment

# Explicit project adapter module, not a package shadow/overlay.
import importlib.util
spec = importlib.util.spec_from_file_location('numeric_tool_adapter', Path(__file__).with_name('tool_mode.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def unused_factory():
    raise RuntimeError('Schema checks must not execute Python code')
env_class = module.make_numeric_environment(CPTWorldEnvironment, unused_factory)
env = env_class()
methods = [method for name, method in inspect.getmembers(env, inspect.ismethod)
           if not name.startswith('_') and name not in {'reset', 'get_reward'}]
schemas = [get_json_schema(method) for method in methods]
assert {s['function']['name'] for s in schemas} == {'act', 'numerics'}
assert {s['function']['name']: s['function']['parameters']['required'] for s in schemas} == {
    'act': ['command'], 'numerics': ['code']}
print(json.dumps({'passed': True, 'schemas': schemas,
                  'versions': {n: importlib.metadata.version(n) for n in ['trl', 'transformers']},
                  'owner_source': inspect.getfile(CPTWorldEnvironment),
                  'official_schema_source': inspect.getfile(get_json_schema),
                  'python': sys.executable, 'trainer_integration': 'not executed',
                  'gpu_used': False}, indent=2))
