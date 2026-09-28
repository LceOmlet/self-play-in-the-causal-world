"""Explicit production wiring: official GRPOTrainer + CPT owner + numeric sandbox.

No Trainer subclass, patched instance methods, scripted rollout or checkpoint
resume. Uses the already audited and explicitly versioned scheduling build.
"""
from __future__ import annotations
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
from datasets import IterableDataset
from transformers import TrainerCallback, set_seed
from trl import GRPOTrainer
from cpt_world import CPTWorldEnvironment, build_cpt_world_advantage_utility, iter_random_balanced_training_rows
from cpt_world.official_train import configuration, lora_configuration, require_contract

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT/'run-01'
MODEL = os.environ.get('CPT_MODEL_DIR', '/models/Qwen/Qwen3.5-9B')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def load(name):
    spec = importlib.util.spec_from_file_location('cpt_numeric_'+name, ROOT/'source'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Progress(TrainerCallback):
    """Observe/save via public callbacks; never change a loss, reward or rollout."""
    def __init__(self):
        self.started = time.monotonic()
        self.before = None
        self.first_change = None
        self.last_saved = 0
        self.status = {'phase':'initializing', 'completed_optimizer_steps':0,
                       'last_complete_checkpoint_step':0, 'first_parameter_change':None}

    def save_status(self, **fields):
        self.status.update(fields, checked_utc=utc())
        write_json(RUN/'status.json', self.status)

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        assert state.global_step == 0
        assert len(optimizer.state) == 0
        trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert trainable and all(p.dtype == torch.float32 for _,p in trainable)
        assert all('lora_' in n for n,_ in trainable)
        assert all(torch.count_nonzero(p).item() == 0 for n,p in trainable if 'lora_B' in n)
        write_json(RUN/'fresh-start.json', {'utc':utc(), 'initial_global_step':0,
            'optimizer_state_entries':0, 'resume_from_checkpoint':None,
            'base_model':MODEL, 'fresh_lora_B_all_zero':True,
            'trainable_dtype':'float32', 'trainable_parameters':sum(p.numel() for _,p in trainable)})
        self.save_status(phase='training', started_utc=utc(), target_steps=args.max_steps)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.monotonic()
        self.save_status(phase='generating_or_updating', active_step=state.global_step+1)

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.first_change is not None:
            return
        self.before = None
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and torch.count_nonzero(parameter.grad).item():
                self.before = (name, parameter, parameter.detach().clone())
                break
        self.current_lr = [float(group['lr']) for group in optimizer.param_groups]

    def on_step_end(self, args, state, control, **kwargs):
        observation = None
        if self.first_change is None:
            observation = {'step':state.global_step, 'learning_rates':self.current_lr,
                           'observed_nonzero_gradient':self.before is not None,
                           'observed_parameter_change':False}
            if self.before is not None:
                name, parameter, before = self.before
                delta = parameter.detach() - before
                observation.update(parameter=name, changed_elements=int(torch.count_nonzero(delta)),
                                   max_abs_change=float(delta.abs().max()))
                observation['observed_parameter_change'] = observation['changed_elements'] > 0
                if observation['observed_parameter_change']:
                    self.first_change = observation
                    write_json(RUN/'first-parameter-change.json', {'utc':utc(), **observation})
                    control.should_save = True
            with (RUN/'initial-update-observations.jsonl').open('a') as stream:
                stream.write(json.dumps({'utc':utc(), **observation})+'\n')
            self.before = None
        # Save the first completed Adam step as well as the official 50-step cadence.
        if state.global_step == 1:
            control.should_save = True
        self.save_status(phase='optimizer_step_complete', completed_optimizer_steps=state.global_step,
                         last_step_seconds=time.monotonic()-self.step_start,
                         elapsed_training_seconds=time.monotonic()-self.started,
                         first_parameter_change=self.first_change)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            record = {'utc':utc(), 'step':state.global_step, **logs}
            with (RUN/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                stream.flush()
            self.save_status(last_metrics=record)

    def on_save(self, args, state, control, optimizer=None, **kwargs):
        checkpoint = Path(args.output_dir)/f'checkpoint-{state.global_step}'
        required = ['adapter_model.safetensors','adapter_config.json','optimizer.pt',
                    'scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin']
        assert all((checkpoint/name).is_file() and (checkpoint/name).stat().st_size > 0 for name in required)
        saved_state = json.loads((checkpoint/'trainer_state.json').read_text())
        assert saved_state['global_step'] == state.global_step
        assert all(v.dtype == torch.float32 for values in optimizer.state.values()
                   for v in values.values() if isinstance(v, torch.Tensor))
        self.last_saved = state.global_step
        self.save_status(last_complete_checkpoint_step=state.global_step,
                         last_complete_checkpoint=str(checkpoint))

    def on_train_end(self, args, state, control, **kwargs):
        self.save_status(phase='completed' if state.global_step == args.max_steps else 'ended_early',
                         completed_optimizer_steps=state.global_step)


def main():
    assert sys.flags.isolated and not os.environ.get('PYTHONPATH')
    preflight = json.loads((ROOT/'preflight-passed.json').read_text())
    assert preflight['passed'] and preflight['production_training_updates'] == 0
    contract = require_contract()
    RUN.mkdir(exist_ok=False)
    callback = Progress()
    callback.save_status(phase='loading_model')
    source_files = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT/'source').glob('*') if p.is_file()}
    write_json(RUN/'process.json', {'utc':utc(),'pid':os.getpid(),'cwd':str(Path.cwd()),
        'argv':sys.argv,'python':sys.executable,'systemd_unit':'cpt-numerics-training.service'})
    adapter, backend = load('tool_mode'), load('secure_backend')
    factory = backend.NsjailSessionFactory(RUN/'sandbox-sessions')
    environment = adapter.make_numeric_environment(CPTWorldEnvironment, factory)
    args = configuration(str(RUN/'checkpoints'))
    args.vllm_rollout_residency = True
    args.vllm_enable_prefix_caching = True
    args.disable_tqdm = True
    args.log_completions = True  # Upstream local Parquet writer; no external logger.
    assert args.max_steps == 10000 and args.max_completion_length == 31497
    assert args.vllm_max_model_length == 32768 and args.generation_kwargs is None
    set_seed(args.seed)
    write_json(RUN/'prepared.json', {'utc':utc(), 'from_rl_step':0,'base_model':MODEL,
        'resume_from_checkpoint':None,'fresh_lora':True,'tools':['act','numerics'],
        'runtime_contract':contract,'source_sha256':source_files,
        'infinite_balanced_task_stream':True,'context_limit':32768,
        'additional_checkpoint_saves':'first completed optimizer step and first observed parameter change',
        'algorithm':'TRL GRPOTrainer with official loss_type=dapo; explicit scheduling build 1.14.0+cpt.rollout1',
        'numeric_limits':{'cpu':1,'memory_bytes':1073741824,'call_seconds':30,'output_utf8_bytes':8192},
        'preflight_result':preflight['result']})
    (RUN/'configuration.json').write_text(args.to_json_string())
    try:
        trainer = GRPOTrainer(model=MODEL, args=args,
            train_dataset=IterableDataset.from_generator(iter_random_balanced_training_rows),
            reward_funcs=build_cpt_world_advantage_utility(), peft_config=lora_configuration(),
            environment_factory=environment, callbacks=[callback])
        assert type(trainer) is GRPOTrainer and trainer.rollout_func is None
        assert {f.__name__ for f in trainer.tools} == {'act','numerics'}
        assert trainer.reward_func_names == ['CPTWorldAdvantageUtility','CPTWorldNumericEnvironment']
        assert trainer.reward_weights.tolist() == [0.0,1.0]
        methods = ['train','training_step','compute_loss','_compute_loss','_generate',
                   '_generate_and_score_completions','_calculate_rewards','_tool_call_loop',
                   '_get_per_token_logps_and_entropies']
        for name in methods:
            assert name not in vars(trainer)
            assert getattr(trainer,name).__func__ is getattr(GRPOTrainer,name)
        write_json(RUN/'trainer-wiring.json', {'utc':utc(), 'trainer_source':inspect.getfile(GRPOTrainer),
            'methods_unmodified':methods,'reward_names':trainer.reward_func_names,
            'reward_weights':trainer.reward_weights.tolist(),'rollout_func':None,
            'tools':[f.__name__ for f in trainer.tools]})
        trainer.train(resume_from_checkpoint=None)
        trainer.save_model(str(RUN/'final-adapter'))
        trainer.save_state()
    except BaseException as error:
        callback.save_status(phase='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        factory.close()


if __name__ == '__main__':
    main()
