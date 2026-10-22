"""Run unchanged upstream tests, then the production update at 32 x 32768.

This is a load check. It does not implement alternative PPO math or assert a
new numerical tolerance. All learning calls execute run_ppo_ondemand directly.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_ppo_ondemand as ppo

root=Path(sys.argv[1]);ppo.setup(root)
import torch
import pytest
from secure_backend import delegated_cgroup
delegated_cgroup()
assert not subprocess.check_output(['git','-C',str(root/'verl-source'),'diff','--name-only'],text=True).strip()
tests=[
    'tests/trainer/ppo/test_core_algos_on_cpu.py::test_multi_turn_compute_gae_advantage_return',
    'tests/trainer/ppo/test_value_loss_normalization_on_cpu.py',
    'tests/trainer/ppo/test_loss_aggregation_on_cpu.py',
    'tests/trainer/ppo/test_rollout_corr_integration.py',
]
assert pytest.main(['-q',*[str(root/'verl-source'/p) for p in tests],
    '--junitxml='+str(root/'batch32-official-tests.xml')])==0
ppo.LOAD['gradient_accumulation_steps']=32
ppo.write(root/'batch32-preflight-progress.json',{'stage':'loading_actor'})
actor,head=ppo.load_actor()
cases=[];g=torch.Generator().manual_seed(43)
for i in range(32):
    ids=torch.randint(10,10000,(32768,),generator=g).tolist()
    cases.append({'prompt_ids':ids[:1271],'completion_ids':ids[1271:],
        'tool_mask':[int(j%5!=0) for j in range(31497)],'reward':[.1,.3,.7,1.][i%4]})
opt=ppo.optimizer_for(actor,head)
torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
ppo.write(root/'batch32-preflight-progress.json',{'stage':'official_gae_and_old_policy','batch_size':32})
tokens=ppo.prepare_ppo(actor,head,cases,1024)
torch.cuda.synchronize();preparation=time.time()-start;start=time.time()
ppo.write(root/'batch32-preflight-progress.json',{'stage':'production_update','batch_size':32,'preparation_seconds':preparation})
metrics=ppo.update(actor,head,cases,opt,1024,tokens)
torch.cuda.synchronize()
assert len(metrics['micro_batches'])==32 and metrics['gradient_accumulation_steps']==32
assert metrics['optimizer_steps']==1 and metrics['first_lora_B_changed']
assert all(int(v['step'])==1 for v in opt.state.values())
metrics.update(passed=True,batch_size=32,microbatch=1,total_context_tokens=32*32768,
    preparation_seconds=preparation,update_seconds=time.time()-start,
    peak_allocated=torch.cuda.max_memory_allocated(),runner_sha256=ppo.sha(Path(ppo.__file__)),
    method='Production runner directly calls pinned unchanged VERL functions; unchanged official tests and assertions',
    numerical_tolerance_added=False,formal_training_reuses_no_preflight_weights=True)
ppo.write(root/'batch32-preflight.json',metrics)
print(json.dumps(metrics),flush=True)
