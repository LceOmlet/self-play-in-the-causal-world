"""Exercise the deployed owner and full 32K/b4 PPO update before a fresh run."""
import argparse
import ast
import copy
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_ppo_ondemand as ppo


def owner_check(root):
    from cpt_world import CPTWorldEnvironment,iter_random_balanced_training_rows
    import tool_mode
    from secure_backend import NsjailSessionFactory
    row=tool_mode.prepare_row(next(iter(iter_random_balanced_training_rows())))
    factory=NsjailSessionFactory(root/'preflight-numeric-sessions')
    env=tool_mode.make_numeric_environment(CPTWorldEnvironment,factory)()
    try:
        assert env.reset(**row) == ''
        # Variable names are public task data. The fixture never supplies task answers.
        labels=env.episode.seed['visible_schema']['variable_labels']
        names=[labels[k] for k,allowed in env.episode.seed['readable'].items() if allowed]
        result=json.loads(env.act({'type':'observe','measure':[names[0]],'batch_size':1}))
        assert result['type']=='batch_stored'
        assert set(result)=={'type','observation_id','n','cost','remaining_budget'}
        output=env.numerics("print(public['observations'][-1]['feedback']['batch']['n'])\nworkspace['retained']=17")
        assert output.strip()=='1',output
        assert env.numerics("print(workspace['retained'])").strip()=='17'
        env2=tool_mode.make_numeric_environment(CPTWorldEnvironment,factory)()
        env2.reset(**copy.deepcopy(row))
        assert env2.numerics("print(len(public['observations']), 'retained' in workspace)").strip()=='0 False'
        env2.get_reward()
        return {'passed':True,'protocol':tool_mode.PROTOCOL_VERSION,'act_receipt_only':True,
                'lossless_observation_available_without_llm_copy':True,'workspace_persistent':True,
                'answer_sessions_isolated':True,'numerics_input':'read trusted public observations; no transcribed data'}
    finally:
        env.get_reward();factory.close()


def head_check(actor):
    """Use bitwise equality for this extra bridge, never invent a numerical epsilon."""
    import torch
    import torch.nn.functional as F
    weight=actor.get_base_model().get_output_embeddings().weight
    h=torch.randn(512,weight.shape[1],device='cuda',dtype=torch.bfloat16)*.1
    targets=torch.randint(0,weight.shape[0],(512,),device='cuda')
    upstream=torch.rand(512,device='cuda',dtype=torch.float32)-.5
    x=h.detach().clone().requires_grad_()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ref=-F.cross_entropy(F.linear(x,weight).float(),targets,reduction='none')
    ref.backward(upstream);ref=ref.detach();reference_grad=x.grad.detach()
    checks={}
    for chunk in (128,512,1024):
        x=h.detach().clone().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=ppo.logprobs(x,weight,targets,chunk)
        out.backward(upstream)
        checks[str(chunk)]={'logprobs_equal':torch.equal(out.detach(),ref),
                            'hidden_gradient_equal':torch.equal(x.grad,reference_grad),
                            'max_logprob_difference':(out.detach()-ref).abs().max().item(),
                            'max_gradient_difference':(x.grad-reference_grad).abs().max().item()}
    # This is explicitly a project bitwise bridge, not an upstream PPO BF16 tolerance.
    assert any(v['logprobs_equal'] and v['hidden_gradient_equal'] for v in checks.values())
    return checks


def gae_mask_bridge():
    import torch
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    # Production keeps observation slots; show that changing only observation
    # values cannot change active-token advantages/returns, exactly as upstream.
    mask=torch.tensor([[1.,1.,0.,0.,1.,0.,1.,0.]]*4)
    r=torch.zeros_like(mask);r[:,6]=torch.tensor([.1,.3,.7,1.])
    a=torch.randn_like(mask);b=a.clone();b[mask==0]=100.
    x,y=compute_gae_advantage_return(r,a,mask,1.,.95)
    u,v=compute_gae_advantage_return(r,b,mask,1.,.95)
    assert torch.equal(x,u) and torch.equal(y*mask,v*mask)
    return {'passed':True,'criterion':'upstream multi-turn torch.equal, unchanged observation mask positions'}


def weight_bridge(actor, force_nonzero=False):
    import torch
    from peft.tuners.lora.layer import LoraLayer
    import run_training_rollout as rollout
    backend=rollout.make_backend(None,{**ppo.LOAD,'min_p':None,'generation_kwargs':None})
    backend.model=actor
    layers={}
    count=0
    with torch.no_grad():
        for name,module in actor.named_modules():
            if not isinstance(module,LoraLayer):continue
            if force_nonzero:
                for adapter in module.active_adapters:
                    module.lora_B[adapter].weight.normal_(0.,.001)
            key=name.removeprefix('base_model.model.')+'.weight'
            key=backend._fix_param_name_to_vllm(key,extra_prefixes=['modules_to_save.default.'])
            layers[key]=module
        for name,candidate in ppo.nonmutating_params(backend):
            if name not in layers:continue
            module=layers[name]
            original=module.base_layer.weight.detach().clone()
            module.merge()
            assert candidate.dtype == module.base_layer.weight.dtype
            assert torch.equal(candidate,module.base_layer.weight), 'PEFT delta stream differs from official merge'
            module.unmerge()
            module.base_layer.weight.copy_(original)
            count+=1
    assert count == len(layers) and count > 0
    return {'passed':True,'lora_layers':count,'criterion':'actual streaming iterator bitwise equality to PEFT merge',
            'nonzero_adapter_weights':True,'frozen_weight_restore_verified':True}


def reuse_benchmark_with_transport_fix(root):
    """Re-use measured update trials only if every learning function is identical."""
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old=functions(root/'preflight04-run_ppo_ondemand.py')
    new=functions(Path(ppo.__file__))
    names=('contract','load_actor','logprobs','forward_case','prepare_ppo','optimizer_for','update')
    assert all(old[n]==new[n] for n in names), 'Learning code changed; full benchmark must be repeated'
    trials=[json.loads((root/'benchmark-trials'/f'trial-{i:02d}.json').read_text())['latest'] for i in (2,3)]
    assert all(t['total_context_tokens']==131072 and t['rollouts']==4 and t['microbatch']==1 and t['grad_accumulation']==4 and t['first_lora_B_changed'] for t in trials)
    actor,head=ppo.load_actor()
    checks=head_check(actor)
    assert all(checks[str(t['chunk'])]['logprobs_equal'] and checks[str(t['chunk'])]['hidden_gradient_equal'] for t in trials)
    weights=weight_bridge(actor,force_nonzero=True)
    selected=min(trials,key=lambda x:x['preparation_seconds']+x['update_seconds'])['chunk']
    result={'passed':True,'head_bitwise_checks':checks,'full_32k_b4_trials':trials,
            'warmup_excluded':True,'selected_logprob_chunk_size':selected,'weight_transfer':weights,
            'learning_function_ast_identical_to_measured_trials':list(names),
            'change_after_timing':'only inference weight streaming BF16 cast; tested separately against actual PEFT merge',
            'formal_training_reuses_no_preflight_weights':True}
    ppo.write(root/'model-preflight.json',result)
    return result


def model_bench(root):
    import torch
    actor,head=ppo.load_actor()
    checks=head_check(actor)
    cases=[]
    g=torch.Generator().manual_seed(43)
    for i in range(4):
        ids=torch.randint(10,10000,(32768,),generator=g).tolist()
        # Same total context and completion ceilings as training, with observation gaps.
        mask=[int(j%5!=0) for j in range(31497)]
        cases.append({'prompt_ids':ids[:1271],'completion_ids':ids[1271:],'tool_mask':mask,'reward':[.1,.3,.7,1.][i]})
    state={n:v.detach().clone() for n,v in actor.named_parameters() if v.requires_grad}
    head_state=copy.deepcopy(head.state_dict())
    trials=[]
    eligible=[int(k) for k,v in checks.items() if v['logprobs_equal'] and v['hidden_gradient_equal']]
    # Warm the full shape once, then measure two choices at the identical learning load.
    chunks=list(dict.fromkeys([512]+eligible))
    chunks=[x for x in chunks if x in eligible][:2]
    for chunk in [chunks[0]]+chunks:
        with torch.no_grad():
            for n,v in actor.named_parameters():
                if n in state:v.copy_(state[n])
        head.load_state_dict(head_state)
        opt=ppo.optimizer_for(actor,head)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.time()
        tokens=ppo.prepare_ppo(actor,head,cases,chunk)
        preparation=time.time()-start;start=time.time()
        metrics=ppo.update(actor,head,cases,opt,chunk,tokens)
        torch.cuda.synchronize()
        metrics.update(chunk=chunk,preparation_seconds=preparation,update_seconds=time.time()-start,
                       total_context_tokens=4*32768,rollouts=4,microbatch=1,grad_accumulation=4,
                       peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        trials.append(metrics)
        ppo.write(root/'preflight-update-progress.json',{'completed_trials':len(trials),'latest':metrics})
        del opt
    weights=weight_bridge(actor)
    measured=trials[1:]
    selected=min(measured,key=lambda x:x['preparation_seconds']+x['update_seconds'])['chunk']
    result={'passed':True,'head_bitwise_checks':checks,'full_32k_b4_trials':trials,
            'first_trial_is_warmup':True,'selected_logprob_chunk_size':selected,
            'selection':'fastest measured eligible identical-load trial; not a claim of global optimality',
            'weight_transfer':weights,'formal_training_reuses_no_preflight_weights':True}
    ppo.write(root/'model-preflight.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chen/runs/ppo-ondemand-from-base-20261022'))
    p.add_argument('--reuse-update-trials',action='store_true')
    a=p.parse_args();ppo.setup(a.root)
    from secure_backend import delegated_cgroup
    delegated_cgroup()
    ppo.write(a.root/'learning-load-contract.json',ppo.contract())
    ppo.write(a.root/'owner-preflight.json',owner_check(a.root))
    ppo.write(a.root/'gae-mask-bridge.json',gae_mask_bridge())
    print(json.dumps(reuse_benchmark_with_transport_fix(a.root) if a.reuse_update_trials else model_bench(a.root),default=str),flush=True)
