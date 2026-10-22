"""Single-GPU actor/value PPO with pinned upstream math and the v1.1 tool owner.

TRL is used only for its unchanged multi-turn generation/parser/tool loop.
All advantages, policy losses and value losses are the pinned VERL PPO functions.
There is no GRPO/group-reward advantage or GRPO optimizer in this program.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO/'benchmarks/repro500_20261019'
BASE = '/home/chen/models/Qwen/Qwen3.5-9B'
BASELINE = Path('/home/chen/runs/official-numerics-group-from-zero-20260929/run-02/configuration.json')
VERL_COMMIT = 'd1390eea0c94467d4ae1387eeda9e2f78f8d73ea'
LOAD = dict(context_length=32768, max_completion_length=31497, generation_batch_size=4,
            num_generations=4, per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=1e-6, max_steps=10000, warmup_steps=0, num_iterations=1,
            lora_rank=8, lora_alpha=32, temperature=1., top_p=1., top_k=0,
            repetition_penalty=1., beta=0., max_grad_norm=1.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8', newline='\n')
    temp.replace(path)


def event(path, value):
    opener = gzip.open(path,'at',encoding='utf-8',compresslevel=6) if path.suffix == '.gz' else path.open('a', encoding='utf-8')
    with opener as f:
        f.write(json.dumps({'unix':time.time(), **value}, ensure_ascii=False, default=str)+'\n')


def setup(root):
    os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
    os.environ.setdefault('VERL_USE_EXTERNAL_PLUGINS', 'none')
    os.environ.setdefault('VLLM_PLUGINS', 'cpt_ppo_cuda')
    assert os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] == '0'
    actual = subprocess.check_output(['git','-C',str(root/'verl-source'),'rev-parse','HEAD'], text=True).strip()
    assert actual == VERL_COMMIT
    for p in (root/'deps', root/'kernel-compat', root/'verl-source', REPO/'runtime_plugins', REPO/'src', REPO/'scripts', SUITE, SUITE/'frozen/protocol'):
        sys.path.insert(0, str(p))
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from transformers import set_seed
    set_seed(42)


def contract():
    old = read(BASELINE)
    pairs = {'context_length':'vllm_max_model_length'}
    for key in ('max_completion_length','generation_batch_size','num_generations',
                'per_device_train_batch_size','gradient_accumulation_steps','learning_rate',
                'max_steps','warmup_steps','num_iterations','temperature','top_p','top_k',
                'repetition_penalty','beta','max_grad_norm'):
        pairs[key] = key
    assert all(LOAD[k] == old[v] for k,v in pairs.items())
    assert old['bf16'] and old['gradient_checkpointing'] and old['use_liger_kernel']
    assert old['chat_template_kwargs'] == {'enable_thinking':False}
    return {'algorithm':'PPO actor/value with GAE; pinned VERL vanilla clipped policy and clipped value losses',
            'load':LOAD, 'baseline_configuration_sha256':sha(BASELINE), 'preserved_fields':pairs,
            'ppo':{'gamma':1.,'lam':.95,'clip_ratio':.2,'clip_ratio_c':3.,'cliprange_value':.2,
                   'vf_coef':.1,'ppo_epochs':1,'loss_agg_mode':'token-mean','kl_coef':0.},
            'optimizer':{k:old[k] for k in ('optim','weight_decay','adam_beta1','adam_beta2','adam_epsilon','lr_scheduler_type')},
            'precision':'BF16 base/compute, FP32 trainable LoRA/value master parameters and log-softmax/PPO statistics; BF16 reduced-precision reductions disabled',
            'environment':'on-demand-data-v1.1; full legal observations automatically stored; act returns receipt only',
            'sampling':'unchanged infinite balanced five-family training stream; four independent environments for each repeated prompt',
            'model':BASE,'fresh_lora':True,'resume_checkpoint':None,'initial_global_step':0,
            'seed':42,'engine_seed':0,'thinking':False,'verl_commit':VERL_COMMIT,
            'rollout_only_methods':'TRL GRPOTrainer._generate/_generate_single_turn/_tokenize_prompts/_tool_call_loop/_get_tool_suffix_ids',
            'grpo_learning_or_group_advantages_used':False}


def load_actor():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
    from grpo_kernel_check import enable_local_fla_kernels, require_gdn_kernels_active
    actor = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16,
                attn_implementation='sdpa', device_map='cpu')
    actor.config.use_cache = False
    _apply_liger_kernel_to_instance(actor, cross_entropy=False, fused_linear_cross_entropy=False)
    actor = get_peft_model(actor, LoraConfig(r=8,lora_alpha=32,lora_dropout=0.,bias='none',
                target_modules='all-linear',exclude_modules=r'.*(?:visual|vision|aligner|multi_token_predictor|mtp|lm_head).*',
                task_type='CAUSAL_LM'))
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.enable_input_require_grads()
    actor.to('cuda')
    enable_local_fla_kernels(actor, Path('/home/chen/kernels/fla-v1-398dfa8c'))
    require_gdn_kernels_active(actor)
    assert all(torch.count_nonzero(p).item() == 0 for n,p in actor.named_parameters() if '.lora_B.' in n)
    head = torch.nn.Linear(actor.config.text_config.hidden_size,1,dtype=torch.float32,device='cuda')
    return actor, head


def logprobs(hidden, weight, targets, chunk_size):
    """Checkpoint ordinary PyTorch logits/CE; never materialize T × vocabulary.

    Weight storage and matmul stay BF16, exactly as the original training model.
    Loss/normalization stay FP32. No custom low-precision tolerance is introduced.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    def forward(x, y):
        return -F.cross_entropy(F.linear(x,weight).float(), y, reduction='none')
    outputs = []
    for i in range(0, targets.numel(), chunk_size):
        x,y = hidden[i:i+chunk_size],targets[i:i+chunk_size]
        outputs.append(checkpoint(forward,x,y,use_reentrant=False) if torch.is_grad_enabled() else forward(x,y))
    return torch.cat(outputs)


def forward_case(actor, head, case, chunk_size):
    import torch
    ids = case['prompt_ids']+case['completion_ids']
    assert len(ids) <= LOAD['context_length'], 'Actual sequence escaped the fixed 32K context'
    x = torch.tensor([ids],device='cuda',dtype=torch.long)
    # This is the same text/multimodal backbone, including LoRA modules; no logits head allocation.
    backbone = actor.get_base_model()
    start = len(case['prompt_ids'])-1
    targets = x[0,start+1:]
    mask = torch.tensor(case['tool_mask'],device='cuda',dtype=torch.bool)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        h = backbone.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True).last_hidden_state
        states = h[0,start:-1]
        active = states[mask]
        lp = logprobs(active,backbone.get_output_embeddings().weight,targets[mask],chunk_size)
        values = head(active).squeeze(-1).float()
    return lp,values


def prepare_ppo(actor, head, cases, chunk_size):
    import torch
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    old = []
    actor.eval();head.eval()
    with torch.no_grad():
        for case in cases:
            p,v = forward_case(actor,head,case,chunk_size)
            old.append((p.detach().cpu(),v.detach().cpu()))
    # Retain the original response/observation layout for upstream multi-turn GAE.
    # Observation values are immaterial by the upstream exact-equality test.
    width = max(len(c['completion_ids']) for c in cases)
    values = torch.zeros((4,width),dtype=torch.float32)
    mask = torch.zeros_like(values);rewards = torch.zeros_like(values)
    for i,((p,v),case) in enumerate(zip(old,cases,strict=True)):
        n = p.numel();assert n > 0
        positions=torch.tensor(case['tool_mask'],dtype=torch.bool).nonzero().flatten()
        assert positions.numel() == n
        values[i,positions]=v;mask[i,positions]=1;rewards[i,positions[-1]]=case['reward']
    advantages,returns = compute_gae_advantage_return(rewards,values,mask,1.,.95)
    for i,(p,v) in enumerate(old):
        positions=torch.tensor(cases[i]['tool_mask'],dtype=torch.bool).nonzero().flatten()
        cases[i].update(old_logprobs=p,old_values=v,advantages=advantages[i,positions],returns=returns[i,positions])
    return int(mask.sum())


def optimizer_for(actor, head):
    import torch
    params = [p for p in actor.parameters() if p.requires_grad]+list(head.parameters())
    return torch.optim.AdamW(params, lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True)


def update(actor, head, cases, opt, chunk_size, global_tokens):
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla,compute_value_loss
    info = {'dp_size':1,'batch_num_tokens':global_tokens,'global_batch_size':4}
    cfg = OmegaConf.create({'clip_ratio':.2,'clip_ratio_low':None,'clip_ratio_high':None,
                           'clip_ratio_c':3.,'global_batch_info':info})
    actor.train();head.train();opt.zero_grad(set_to_none=True)
    metrics=[]
    before = next(p for n,p in actor.named_parameters() if '.lora_B.' in n).detach().clone()
    for case in cases:
        lp,v = forward_case(actor,head,case,chunk_size)
        row = lambda key:case[key].to('cuda').unsqueeze(0)
        mask = torch.ones_like(lp).unsqueeze(0)
        pg,pm = compute_policy_loss_vanilla(row('old_logprobs'),lp[None],row('advantages'),mask,'token-mean',cfg)
        vf,clip = compute_value_loss(v[None],row('returns'),row('old_values'),mask,.2,'token-mean',**info)
        loss = pg+.1*vf
        assert torch.isfinite(loss)
        loss.backward()  # Upstream global denominator already accounts for all four micro-batches.
        metrics.append({'policy_loss':pg.item(),'value_loss':vf.item(),**pm,'value_clipfrac':clip.item()})
        del lp,v,pg,vf,loss
    params=[p for p in actor.parameters() if p.requires_grad]+list(head.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    norm=torch.nn.utils.clip_grad_norm_(params,1.)
    assert torch.isfinite(norm)
    opt.step();opt.zero_grad(set_to_none=True)
    after=next(p for n,p in actor.named_parameters() if '.lora_B.' in n)
    return {'micro_batches':metrics,'grad_norm':norm.item(),'first_lora_B_changed':not torch.equal(before,after),
            'active_policy_tokens':global_tokens,'optimizer_steps':1,'gradient_accumulation_steps':4}


def nonmutating_params(self):
    """Use PEFT's own delta calculation while keeping frozen BF16 weights intact."""
    import torch
    from peft.tuners.lora.layer import LoraLayer
    modules=dict(self.model.named_modules())
    for name,param in self.model.named_parameters():
        if self.model.prefix in name or 'original_module' in name:
            continue
        value=param.data
        if name.endswith('.base_layer.weight'):
            layer=modules[name.removesuffix('.base_layer.weight')]
            assert isinstance(layer,LoraLayer) and not layer.merged
            for adapter in layer.active_adapters:
                if adapter in layer.lora_A:
                    with torch.no_grad():
                        value=(value+layer.get_delta_weight(adapter)).to(param.dtype)
        target=name.removeprefix('base_model.model.').replace('.base_layer','')
        target=self._fix_param_name_to_vllm(target,extra_prefixes=['modules_to_save.default.'])
        yield target,value


class GenerationTrace:
    def __init__(self,llm):
        self.llm=llm;self.path=None
    def __getattr__(self,n):
        return getattr(self.llm,n)
    def generate(self,prompts,sampling_params,**kwargs):
        assert sampling_params.max_tokens == 31497
        started=time.time()
        event(self.path,{'kind':'generation_request','prompt_lengths':[len(p['prompt_token_ids']) for p in prompts],
                         'prompts':prompts,'sampling_params':str(sampling_params),'max_tokens':sampling_params.max_tokens})
        outputs=self.llm.generate(prompts,sampling_params=sampling_params,**kwargs)
        event(self.path,{'kind':'generation_response','seconds':time.time()-started,
             'outputs':[{'prompt_token_ids':list(o.prompt_token_ids),'choices':[{'token_ids':list(c.token_ids),
                 'text':c.text,'finish_reason':c.finish_reason,'stop_reason':c.stop_reason,
                 'sampled_token_logprobs':[next(iter(p.values())).logprob for p in c.logprobs or []]} for c in o.outputs]} for o in outputs]})
        return outputs


def save(actor,head,opt,runtime,step):
    import torch
    folder=runtime/'checkpoints'/f'checkpoint-{step}'
    actor.save_pretrained(folder)
    torch.save({'value_head':head.state_dict(),'optimizer':opt.state_dict(),'global_step':step},folder/'ppo_state.pt')
    write(folder/'training-state.json',{'global_step':step,'algorithm':'PPO','context_length':32768,'batch_size':4})


def train(root,runtime,chunk_size):
    import torch
    from accelerate import Accelerator
    from trl.generation.vllm_generation import DistributedBackend
    from transformers.utils import get_json_schema
    from vllm import LLM
    from cpt_world import CPTWorldEnvironment,iter_random_balanced_training_rows
    import tool_mode
    from secure_backend import delegated_cgroup,NsjailSessionFactory
    import run_training_rollout as rollout
    delegated_cgroup()
    assert not runtime.exists(), 'A fresh run cannot silently overwrite or resume prior training'
    runtime.mkdir(parents=True)
    configuration=contract();write(runtime/'configuration.json',configuration)
    write(runtime/'status.json',{'stage':'loading_from_original','global_step':0})
    actor,head=load_actor()
    write(runtime/'fresh-start.json',{'model':BASE,'global_step':0,'lora_B_all_zero':True,'resumed':False,
                                     'runner_sha256':sha(Path(__file__))})
    actor.to('cpu');head.to('cpu');gc.collect();torch.cuda.empty_cache()
    llm=LLM(model=BASE,tokenizer=BASE,dtype='bfloat16',max_model_len=32768,
            max_num_seqs=4,max_num_batched_tokens=4096,gpu_memory_utilization=.75,
            seed=0,enable_prefix_caching=True,enable_sleep_mode=True,model_impl='vllm',
            enforce_eager=False,compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'},
            logprobs_mode='processed_logprobs',limit_mm_per_prompt={'image':0,'video':0},disable_log_stats=True)
    proxy=GenerationTrace(llm)
    cfg={**LOAD,'min_p':None,'generation_kwargs':None,'chat_template_kwargs':{'enable_thinking':False},'max_tool_calling_iterations':None}
    backend=rollout.make_backend(proxy,cfg)
    backend.model=actor;backend.accelerator=Accelerator(mixed_precision='bf16')
    backend._dist=DistributedBackend(backend.accelerator)
    backend._iter_named_params=MethodType(nonmutating_params,backend)
    backend.enable_sleep_mode=True;backend.rollout_residency=True
    processor,tokenizer,template=rollout.training_processor()
    factory=NsjailSessionFactory(runtime/'numeric-sessions')
    env_type=tool_mode.make_numeric_environment(CPTWorldEnvironment,factory)
    rows=iter(iter_random_balanced_training_rows())
    opt=None
    try:
        for step in range(1,LOAD['max_steps']+1):
            folder=runtime/'rollouts'/f'step-{step:06d}';folder.mkdir(parents=True)
            row=tool_mode.prepare_row(next(rows))
            # Same prompt/group-of-four workload; learning is PPO, not group-standardized GRPO.
            envs=[]
            for i in range(4):
                e=env_type();e.reset(**copy.deepcopy(row))
                case=folder/f'case-{i}';case.mkdir()
                envs.append(rollout.LoggedEnvironment(e,f'{step}-{i}',case))
            prompts=[copy.deepcopy(row['prompt']) for _ in range(4)]
            write(folder/'task-row.json',row)
            write(folder/'public-prompt.json',{'messages':row['prompt'],'tools':[get_json_schema(envs[0].act),get_json_schema(envs[0].numerics)]})
            if step > 1:
                t=time.time();backend.sync_weights()
                write(folder/'weight-sync.json',{'seconds':time.time()-t,'source':'updated PPO actor','frozen_weights_merged_in_place':False})
                actor.to('cpu');head.to('cpu');gc.collect();torch.cuda.empty_cache()
            proxy.path=folder/'generation-batches.jsonl.gz'
            trainer=rollout.make_trainer(processor,tokenizer,template,backend,envs,cfg)
            write(runtime/'status.json',{'stage':'rollout','global_step':step-1,'active_step':step,'batch_size':4,'context_length':32768})
            start=time.time()
            with backend.rollout_session():
                result=trainer._generate(prompts)
            prompt_ids,completion_ids,tool_masks=result[:3]
            assert all(not e.infra_errors for e in envs)
            cases=[]
            for i,e in enumerate(envs):
                c={'prompt_ids':prompt_ids[i],'completion_ids':completion_ids[i],'tool_mask':tool_masks[i],
                   'reward':e.env.get_reward(),'submitted':bool(e.env.episode.completed),
                   'tool_calls':e.calls,'observations_used':e.env.episode.observations_used}
                assert len(c['completion_ids']) <= 31497
                write(folder/f'case-{i}'/'rollout.json',c);cases.append(c)
            rollout_seconds=time.time()-start
            write(runtime/'status.json',{'stage':'ppo_update','global_step':step-1,'active_step':step})
            actor.to('cuda');head.to('cuda')
            if opt is None:opt=optimizer_for(actor,head)
            for group in opt.param_groups:group['lr']=1e-6*(1-(step-1)/10000)
            torch.cuda.reset_peak_memory_stats();start=time.time()
            tokens=prepare_ppo(actor,head,cases,chunk_size)
            metrics=update(actor,head,cases,opt,chunk_size,tokens)
            torch.cuda.synchronize()
            metrics.update(step=step,rollout_seconds=rollout_seconds,update_seconds=time.time()-start,
                    rewards=[c['reward'] for c in cases],submitted=sum(c['submitted'] for c in cases),
                    peak_cuda_allocated=torch.cuda.max_memory_allocated(),logprob_chunk_size=chunk_size,
                    learning_rate=opt.param_groups[0]['lr'],task_family=row['query_type'])
            event(runtime/'metrics.jsonl',metrics)
            write(runtime/'status.json',{'stage':'running','global_step':step,'latest_metrics':metrics})
            if step == 1 or step % 50 == 0:save(actor,head,opt,runtime,step)
    except BaseException as exc:
        write(runtime/'failure.json',{'error_type':type(exc).__name__,'message':str(exc),'automatic_resample':False})
        raise
    finally:
        factory.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chen/runs/ppo-ondemand-from-base-20261022'))
    p.add_argument('--runtime',type=Path)
    p.add_argument('--logprob-chunk-size',type=int,default=512)
    a=p.parse_args();setup(a.root)
    train(a.root,a.runtime or a.root/'training',a.logprob_chunk_size)
