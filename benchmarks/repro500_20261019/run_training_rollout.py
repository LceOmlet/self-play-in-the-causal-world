"""Evaluate frozen cases with the installed training rollout implementation.

GRPOTrainer's generation, token concatenation, response parsing, tool loop and
completion truncation methods are called unchanged. No optimizer is created.
Each case gets one rollout; batches contain four distinct evaluation cases.
The training group-of-four advantage calculation is not an evaluation step.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

SUITE = Path(__file__).resolve().parent
TRAIN = Path('/home/chen/runs/official-numerics-group-from-zero-20260929/run-02')
BASE = '/home/chen/models/Qwen/Qwen3.5-9B'
ADAPTER = TRAIN/'checkpoints/checkpoint-2350'
METHODS = ('_generate', '_generate_single_turn', '_tokenize_prompts',
           '_tool_call_loop', '_get_tool_suffix_ids')
CONFIG_KEYS = ('max_completion_length', 'vllm_max_model_length', 'max_tool_calling_iterations',
    'temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'generation_kwargs',
    'chat_template_kwargs', 'num_generations', 'generation_batch_size', 'seed',
    'vllm_model_impl', 'vllm_enable_prefix_caching', 'vllm_speculative_config', 'model_init_kwargs')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n',
                         encoding='utf-8', newline='\n')
    temporary.replace(path)


def event(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps({'unix':time.time(), **value}, ensure_ascii=False, default=str)+'\n')
        out.flush()
        os.fsync(out.fileno())


def training_processor():
    from transformers import AutoProcessor, ProcessorMixin
    from trl.chat_template_utils import (
        add_response_schema, get_training_chat_template, is_chat_template_prefix_preserving)
    processing = AutoProcessor.from_pretrained(BASE, truncation_side='left', padding_side='left',
                                               local_files_only=True)
    tokenizer = processing.tokenizer if isinstance(processing, ProcessorMixin) else processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not (getattr(tokenizer, 'response_template', None) or getattr(tokenizer, 'response_schema', None)):
        processing = add_response_schema(processing)
    template = None if is_chat_template_prefix_preserving(processing) else get_training_chat_template(processing)
    return processing, tokenizer, template


def capture_contract():
    import trl
    import transformers
    import vllm
    from trl import GRPOTrainer
    from trl.generation.vllm_generation import VLLMGeneration
    configuration = read(TRAIN/'configuration.json')
    assert configuration['max_completion_length'] == 31497
    assert configuration['vllm_max_model_length'] == 32768
    assert configuration['generation_kwargs'] is None
    assert configuration['chat_template_kwargs'] == {'enable_thinking':False}
    processing, tokenizer, template = training_processor()
    prepared = read(TRAIN/'prepared.json')
    protocol_sources = {}
    for name in ('tool_mode.py','secure_backend.py','nsjail-secure.cfg.in'):
        source = str(TRAIN.parent/'source'/name)
        actual = digest((SUITE/'frozen/protocol'/name).read_bytes())
        original = prepared['source_sha256'][source]
        assert digest(Path(source).read_bytes()) == original, 'Training source changed: '+name
        # The later, explicitly frozen v1.1 on-demand task interface is retained.
        # It is intentionally different from training's old automatic-feedback adapter.
        if name != 'tool_mode.py':
            assert actual == original,name
        protocol_sources[name] = {'evaluation_sha256':actual,'training_sha256':original,
                                  'identical':actual == original}
    sources = {name:digest(inspect.getsource(getattr(GRPOTrainer,name)).encode()) for name in METHODS}
    sources['VLLMGeneration.generate'] = digest(inspect.getsource(VLLMGeneration.generate).encode())
    sources['GRPOTrainer.__init__'] = digest(inspect.getsource(GRPOTrainer.__init__).encode())
    return {
        'version':'training-rollout-v3-no-8k-20261021',
        'training_configuration':str(TRAIN/'configuration.json'),
        'training_configuration_sha256':digest((TRAIN/'configuration.json').read_bytes()),
        'training_settings':{k:configuration.get(k) for k in CONFIG_KEYS},
        'engine_seed':0,  # Actual colocated TRL backend: process_index // tensor_parallel_size.
        'extra_per_turn_generation_cap':None,
        'task_interface':'Frozen on-demand-data-v1.1, retained for all fixed eval100 cases',
        'training_task_interface_identical':False,
        'completion_budget_includes_tool_feedback':True,
        'rollouts_per_evaluation_case':1, 'evaluation_cases_per_model':100,
        'inference_batch_size':4, 'generation_methods_unmodified':list(METHODS),
        'method_source_sha256':sources, 'protocol_source_sha256':protocol_sources,
        'processor_class':type(processing).__name__,
        'tokenizer_class':type(tokenizer).__name__,
        'training_chat_template_sha256':digest((template or processing.chat_template).encode()),
        'response_template_sha256':digest(str(getattr(tokenizer,'response_template',None)).encode()),
        'versions':{'trl':trl.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__},
        'checkpoint_adapter_sha256':digest((ADAPTER/'adapter_model.safetensors').read_bytes()),
        'checkpoint_adapter_config_sha256':digest((ADAPTER/'adapter_config.json').read_bytes()),
        'base_model':BASE, 'checkpoint_adapter':str(ADAPTER),
        'inference_only_differences':[
            'No optimizer, loss, reward normalization, weight updates or repeated training samples.',
            'One rollout per fixed evaluation case; batches contain four distinct cases rather than four copies of a training question.',
            'Standalone inference engine keeps weights resident; no training/inference sleep cycle or external distributed launcher.',
            'Checkpoint is an independently saved PEFT-merged BF16 snapshot using the training merge recipe; CPU merge arithmetic need not be bitwise identical to training GPU arithmetic.',
            'GPU memory reservation and eager execution are inference resource settings, not token caps.'
        ]}


def validate_contract(path):
    expected = read(path)
    actual = capture_contract()
    assert actual == expected, 'Training implementation/configuration changed since contract capture'
    return actual


def make_trainer(processing, tokenizer, template, backend, environments, config):
    """Initialize generation-only state; all executed rollout methods stay upstream."""
    import torch
    from transformers import ProcessorMixin
    from trl import GRPOTrainer
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.processing_class = processing
    trainer._tokenizer = tokenizer
    trainer._is_vlm = isinstance(processing, ProcessorMixin)
    trainer.chat_template = template
    trainer.chat_template_kwargs = config['chat_template_kwargs']
    trainer.environment_factories = {None:None}
    trainer._batch_environments = [None]*len(environments)
    tools = [environments[0].act, environments[0].numerics]
    trainer._env_tools = {None:tools}
    trainer.tools = tools
    trainer._sync_tool_dicts = [e.dispatch for e in environments]
    trainer._async_tool_dicts = [{} for _ in environments]
    trainer.max_completion_length = config['max_completion_length']
    trainer.max_tool_calling_iterations = config['max_tool_calling_iterations'] or sys.maxsize
    trainer.use_vllm = True
    trainer.vllm_mode = 'colocate'
    trainer.vllm_generation = backend
    trainer._last_loaded_step = 0
    trainer.state = SimpleNamespace(global_step=0, num_input_tokens_seen=0)
    trainer.model = SimpleNamespace(training=True)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.accelerator = SimpleNamespace(device=torch.device('cpu'), is_main_process=True,
                                         gather=lambda value:value)
    trainer.num_generations = config['num_generations']
    trainer.num_generations_eval = config['num_generations']
    trainer.rollout_func = None
    trainer._metrics = {'train':defaultdict(list),'eval':defaultdict(list)}
    for name in METHODS:
        assert name not in vars(trainer)
        assert getattr(trainer,name).__func__ is getattr(GRPOTrainer,name)
    return trainer


def make_backend(llm, configuration):
    import torch
    from trl.generation.vllm_generation import VLLMGeneration
    backend = VLLMGeneration.__new__(VLLMGeneration)
    # Backend only checks PEFT/distributed status here; inference weights live in LLM.
    backend.model = torch.nn.Module()
    backend.llm = llm
    backend.accelerator = SimpleNamespace(process_index=0, local_process_index=0,
                                          is_main_process=True, num_processes=1)
    backend.mode = 'colocate'
    backend.tensor_parallel_size = 1
    backend.enable_sleep_mode = False
    backend.rollout_residency = True
    backend._rollout_session_depth = 0
    backend._llm_weights_sleeping = False
    backend._llm_kv_cache_sleeping = False
    backend.structured_outputs_regex = None
    backend.generation_kwargs = configuration['generation_kwargs'] or {}
    backend.logprobs = 0
    for field in ('temperature','top_p','top_k','repetition_penalty','max_completion_length'):
        setattr(backend,field,configuration[field])
    backend.min_p = configuration['min_p'] or 0.0
    return backend


class LoggedEnvironment:
    def __init__(self, env, task_id, folder):
        self.env, self.task_id, self.folder = env, task_id, folder
        self.dispatch = {'act':self.logged_act,'numerics':self.logged_numerics}
        self.infra_errors = []
        self.calls = 0
        self.initial_ids = None
        self.generation_calls = 0
        # Schemas must come from the original environment methods, not logging wrappers.
        self.act, self.numerics = env.act, env.numerics

    def execute(self, name, arguments):
        self.calls += 1
        event(self.folder/'events.jsonl', {'kind':'tool_request','tool':name,'arguments':arguments})
        try:
            result = getattr(self.env,name)(**arguments)
        except Exception as error:
            event(self.folder/'events.jsonl',{'kind':'tool_exception','tool':name,
                                            'error_type':type(error).__name__,'message':str(error)})
            if not isinstance(error,(ValueError,TypeError)):
                self.infra_errors.append(type(error).__name__)
            raise
        detail = {'kind':'tool_response','tool':name,'output':result,
                  'episode_completed':bool(self.env.episode.completed),
                  'remaining_budget':self.env.episode.remaining_budget}
        if name == 'act' and self.env._visible['observations']:
            detail['stored_observation'] = self.env._visible['observations'][-1]
        event(self.folder/'events.jsonl',detail)
        return result

    def logged_act(self, command):
        return self.execute('act',{'command':command})

    def logged_numerics(self, code):
        return self.execute('numerics',{'code':code})


class LoggedLLM:
    """Observe the original LLM.generate call without changing its inputs/results."""
    def __init__(self, llm):
        self.llm = llm
        self.active = []

    def __getattr__(self, name):
        return getattr(self.llm,name)

    def identify(self, ids):
        matches = [e for e in self.active if ids[:len(e.initial_ids)] == e.initial_ids]
        assert len(matches) == 1, 'Ambiguous generation trace attribution'
        return matches[0]

    def generate(self, prompts, sampling_params, **kwargs):
        assert sampling_params.max_tokens == 31497, 'Extra per-turn cap is forbidden'
        records = [self.identify(p['prompt_token_ids']) for p in prompts]
        for environment,prompt in zip(records,prompts,strict=True):
            environment.generation_calls += 1
            event(environment.folder/'events.jsonl', {'kind':'generation_request',
                'turn':environment.generation_calls,'prompt_token_ids':prompt['prompt_token_ids'],
                'sampling_params':str(sampling_params),'max_tokens':sampling_params.max_tokens})
        outputs = self.llm.generate(prompts,sampling_params=sampling_params,**kwargs)
        assert len(outputs) == len(records)
        for environment, output in zip(records,outputs,strict=True):
            choices = []
            for choice in output.outputs:
                values = [next(iter(step.values())).logprob for step in choice.logprobs or []]
                assert len(values) == len(choice.token_ids) and all(math.isfinite(v) for v in values)
                choices.append({'token_ids':list(choice.token_ids),'text':choice.text,
                    'finish_reason':choice.finish_reason,'stop_reason':choice.stop_reason,
                    'sampled_token_logprobs':values})
            event(environment.folder/'events.jsonl',{'kind':'generation_response',
                  'turn':environment.generation_calls,'prompt_token_ids':list(output.prompt_token_ids),
                  'outputs':choices})
        return outputs


def load_capsule(task_id, entry):
    path = SUITE/'data'/entry['file']
    assert digest(path.read_bytes()) == entry['compressed_sha256']
    with gzip.open(path,'rt',encoding='utf-8') as source:
        capsule = json.load(source)
    from capsule import canonical
    assert digest(canonical(capsule)) == entry['sha256']
    assert capsule['task_id'] == task_id
    return capsule


def merge_checkpoint(output):
    """Create a derivative; neither original base weights nor adapter is modified."""
    import torch
    from peft import PeftModel
    from trl.trainer.utils import create_model_from_path
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    model = create_model_from_path(BASE,dtype='bfloat16',device_map='cpu',local_files_only=True)
    model = PeftModel.from_pretrained(model,str(ADAPTER),is_trainable=False,autocast_adapter_dtype=True)
    assert all(p.dtype == torch.float32 for n,p in model.named_parameters() if 'lora_' in n)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(output,safe_serialization=True,max_shard_size='4GB')
    write(output/'merge-receipt.json',{'source_adapter':str(ADAPTER),
        'source_adapter_sha256':digest((ADAPTER/'adapter_model.safetensors').read_bytes()),
        'source_base':BASE,'base_dtype':'bfloat16','adapter_dtype':'float32',
        'merge':'PEFT merge_and_unload; same merge recipe as training weight synchronization',
        'merge_device':'cpu','sources_modified':False,
        'files':{p.name:digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}})


def run_model(args):
    contract = validate_contract(args.contract)
    if args.engine_profile == 'graph':
        evidence = read(args.tolerance_report)
        assert evidence['passed'] and all(evidence['models'][m]['passed']
            for m in ('base','checkpoint2350')), 'Both model gates must pass before evaluation'
        contract = copy.deepcopy(contract)
        contract['version'] = 'training-rollout-v4-graphs-no-8k-20261021'
        contract['engine_execution'] = {'enforce_eager':False,
            'tolerance_report_sha256':digest(args.tolerance_report.read_bytes()),
            'same_profile_both_models':True}
    config = contract['training_settings']
    runtime = args.runtime
    runtime.mkdir(parents=True,exist_ok=False)
    write(runtime/'contract.json',contract)
    write(runtime/'runner-source.json',{'run_training_rollout.py_sha256':digest(Path(__file__).read_bytes()),
        'training_methods':contract['method_source_sha256'],'no_method_overrides':True})
    write(runtime/'progress.json',{'model':args.model,'total':100,'started':0,'completed':0,'submitted':0,'stage':'loading'})
    sys.path.insert(0,str(SUITE/'frozen/protocol'))
    from secure_backend import NsjailSessionFactory
    factory = NsjailSessionFactory(runtime/'numeric-sessions')
    import tool_mode
    from capsule import FrozenEnvironment
    from transformers import set_seed
    from transformers.utils import get_json_schema
    from vllm import LLM
    set_seed(config['seed'])
    env_type = tool_mode.make_numeric_environment(FrozenEnvironment,factory)
    manifest = read(SUITE/'data/manifest.json')
    for name, expected in manifest['owner']['files'].items():
        assert digest((SUITE/name).read_bytes()) == expected,name
    evaluation = read(SUITE/'data/eval100.json')
    entries = {entry['task_id']:entry for entry in manifest['tasks']}
    assert len(evaluation['task_ids']) == len(set(evaluation['task_ids'])) == 100
    model_path = BASE if args.model == 'base' else str(args.merged_checkpoint)
    if args.model != 'base':
        receipt = read(args.merged_checkpoint/'merge-receipt.json')
        assert receipt['source_adapter_sha256'] == contract['checkpoint_adapter_sha256']
        for name, expected in receipt['files'].items():
            assert digest((args.merged_checkpoint/name).read_bytes()) == expected
    llm = LLM(model=model_path,tokenizer=BASE,dtype='bfloat16',
        max_model_len=config['vllm_max_model_length'],max_num_seqs=4,
        gpu_memory_utilization=0.75,max_num_batched_tokens=4096,
        seed=contract['engine_seed'],enable_prefix_caching=True,
        model_impl=config['vllm_model_impl'],enforce_eager=args.engine_profile == 'eager',
        logprobs_mode='processed_logprobs',limit_mm_per_prompt={'image':0,'video':0})
    write(runtime/'engine-execution.json',{'profile':args.engine_profile,
        'compilation_config':str(llm.llm_engine.vllm_config.compilation_config),
        'max_model_len':32768,'max_completion_length':31497})
    proxy = LoggedLLM(llm)
    backend = make_backend(proxy,config)
    processing,tokenizer,template = training_processor()
    completed = submitted = started = 0
    summaries = []
    try:
        for offset in range(0,100,4):
            group = evaluation['task_ids'][offset:offset+4]
            environments, prompts = [], []
            for task in group:
                capsule = load_capsule(task,entries[task])
                env = env_type()
                env.reset(capsule=capsule,**capsule['row'])
                folder = runtime/'cases'/task
                folder.mkdir(parents=True,exist_ok=False)
                write(folder/'prompt.json',{'messages':capsule['row']['prompt'],
                    'tools':[get_json_schema(env.act),get_json_schema(env.numerics)],
                    'capsule_sha256':entries[task]['sha256']})
                environments.append(LoggedEnvironment(env,task,folder))
                prompts.append(copy.deepcopy(capsule['row']['prompt']))
                started += 1
            trainer = make_trainer(processing,tokenizer,template,backend,environments,config)
            initial_ids,_,_ = trainer._tokenize_prompts(copy.deepcopy(prompts))
            for environment,ids in zip(environments,initial_ids,strict=True):
                environment.initial_ids = ids
            proxy.active = environments
            write(runtime/'progress.json',{'model':args.model,'total':100,'started':started,
                'completed':completed,'submitted':submitted,'stage':'running','active_task_ids':group})
            output = trainer._generate(prompts)
            prompt_ids,completion_ids,tool_masks,completions,logprobs = output[:5]
            assert all(not environment.infra_errors for environment in environments), 'Executor infrastructure failure'
            for i,environment in enumerate(environments):
                ep = environment.env.episode
                assert prompt_ids[i] == environment.initial_ids
                assert len(completion_ids[i]) <= config['max_completion_length']
                assert len(completion_ids[i]) == len(tool_masks[i]) == len(logprobs[i])
                result = {'task_id':environment.task_id,'family':entries[environment.task_id]['family'],
                    'model':args.model,'valid_terminated':True,'submitted':bool(ep.completed),
                    'reward':environment.env.get_reward(),'terminal_score':ep.terminal_score,
                    'terminal_reward_exact':str(ep.terminal_reward),
                    'budget':ep.budget.max_observations,'observations_used':ep.observations_used,
                    'remaining_budget':ep.remaining_budget,'queries_used':ep.queries_used,
                    'tool_calls':environment.calls,'generation_calls':environment.generation_calls,
                    'initial_prompt_tokens':len(prompt_ids[i]),'rollout_completion_tokens':len(completion_ids[i]),
                    'rollout_model_tokens':sum(tool_masks[i]),
                    'rollout_tool_feedback_tokens':len(tool_masks[i])-sum(tool_masks[i]),
                    'reason':'terminal_answer' if ep.completed else 'training_rollout_ended_without_terminal_answer'}
                write(environment.folder/'rollout.json',{'prompt_ids':prompt_ids[i],
                    'completion_ids':completion_ids[i],'tool_mask':tool_masks[i],
                    'logprobs':logprobs[i],'completions':completions[i]})
                write(environment.folder/'private-result.json',result)
                completed += 1
                submitted += int(result['submitted'])
                summaries.append({'task_id':environment.task_id,'valid_terminated':True,
                                  'submitted':result['submitted']})
                write(runtime/'progress.json',{'model':args.model,'total':100,'started':started,
                    'completed':completed,'submitted':submitted,'stage':'running'})
                print(json.dumps({'model':args.model,'completed':completed,'total':100}),flush=True)
        write(runtime/'run-summary.json',{'model':args.model,'tasks':summaries})
        write(runtime/'progress.json',{'model':args.model,'total':100,'started':100,
            'completed':completed,'submitted':submitted,'stage':'complete'})
    except BaseException as error:
        write(runtime/'failure.json',{'kind':'infrastructure_or_harness_failure',
            'error_type':type(error).__name__,'message':str(error),'valid_completed':completed,
            'retry_or_resample_permitted':False})
        raise
    finally:
        factory.close()


def configure_platform():
    # Same process-local NonNvmlCudaPlatform fallback as the existing evaluator.
    import vllm.platforms as platforms
    platforms.builtin_platform_plugins['cuda'] = lambda:'vllm.platforms.cuda.NonNvmlCudaPlatform'
    platforms._current_platform = None


# Spawned vLLM workers re-import this module, so the local platform fallback must
# also run on import, as in the existing inference launcher.
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
configure_platform()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-contract',type=Path)
    parser.add_argument('--merge-checkpoint',type=Path)
    parser.add_argument('--model',choices=('base','checkpoint2350'))
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--merged-checkpoint',type=Path)
    parser.add_argument('--engine-profile',choices=('eager','graph'),default='eager')
    parser.add_argument('--tolerance-report',type=Path)
    arguments = parser.parse_args()
    if arguments.capture_contract:
        write(arguments.capture_contract,capture_contract())
        print('Training generation contract captured; no inference performed.')
    elif arguments.merge_checkpoint:
        merge_checkpoint(arguments.merge_checkpoint)
    elif arguments.model and arguments.contract and arguments.runtime:
        configure_platform()
        run_model(arguments)
    else:
        parser.error('Choose capture, merge, or model evaluation mode')
