"""Launch the explicitly authorized checkpoint-1 continuation with batch 32."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

root=Path(sys.argv[1]);project=root/'project-b32';runtime=root/'training-b32'
sys.path.insert(0,str(project/'scripts'))
from run_ppo_ondemand import read,write,sha
assert not runtime.exists(), 'Never overwrite a continuation or retry it silently'
assert subprocess.check_output(['systemctl','--user','show',
    'cpt-numerics-ppo-ondemand-from-base-20261022.service','-p','MainPID','--value'],text=True).strip()=='0'
assert subprocess.check_output(['systemctl','--user','show',
    'cpt-numerics-ppo-batch32-preflight-20261022.service','-p','MainPID','--value'],text=True).strip()=='0'
suite=ET.parse(root/'batch32-official-tests.xml').getroot().findall('.//testsuite')
assert sum(int(s.get('tests','0')) for s in suite)==45
assert all(int(s.get(k,'0'))==0 for s in suite for k in ('failures','errors','skipped'))
preflight=read(root/'batch32-preflight.json')
assert preflight['passed'] and preflight['batch_size']==32 and preflight['optimizer_steps']==1
assert preflight['total_context_tokens']==32*32768 and preflight['gradient_accumulation_steps']==32
measured_runner=root/'batch32-preflight-runner.py'
assert preflight['runner_sha256']==sha(measured_runner)
def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,ast.FunctionDef)}
measured,current=functions(measured_runner),functions(project/'scripts/run_ppo_ondemand.py')
learning_functions=('load_actor','logprobs','forward_case','prepare_ppo','optimizer_for','update')
assert all(measured[n]==current[n] for n in learning_functions)
assert preflight['formal_training_reuses_no_preflight_weights']
checkpoint=root/'training/checkpoints/checkpoint-1'
validation=read(root/'checkpoint-validation.json')
assert validation['passed'] and validation['global_step']==1
assert validation['value_head_finite_fp32'] and validation['adapter_all_finite']
assert validation['optimizer_states_at_step_one']==498 and validation['nonzero_lora_B_layers']==248
assert all(sha(root/name)==entry['sha256'] for name,entry in validation['checkpoint_files'].items())
assert sha(checkpoint/'adapter_model.safetensors')=='d4a98c5bf7680b07d752da0260b62e552778b3a107a9f3129e3781a95845a843'
assert read(checkpoint/'training-state.json')['global_step']==1
head=subprocess.check_output(['git','-C',str(root/'verl-source'),'rev-parse','HEAD'],text=True).strip()
assert head=='d1390eea0c94467d4ae1387eeda9e2f78f8d73ea'
assert not subprocess.check_output(['git','-C',str(root/'verl-source'),'diff','--name-only'],text=True).strip()
hashes={str(p.relative_to(project)):sha(p) for p in project.rglob('*')
        if p.is_file() and (p.suffix in ('.py','.in') or 'runtime_plugins' in p.parts)
        and '__pycache__' not in p.parts}
unit='cpt-numerics-ppo-ondemand-b32-20261022'
receipt={'unix':time.time(),'source_sha256':hashes,'official_ppo_tests_passed':45,
    'verl_commit':head,'preflight_sha256':sha(root/'batch32-preflight.json'),
    'learning_functions_ast_identical_to_full_load_preflight':learning_functions,
    'post_preflight_training_change':'Resume CUDA weight sync; batch status fields; assert and record actual upstream function paths/hashes',
    'resume_checkpoint':str(checkpoint),'resume_step':1,
    'user_authorized_discard_unsaved_updates':True,
    'prior_run_status_at_switch':read(root/'training/status.json'),
    'discarded_updates_do_not_enter_new_series':True,
    'optimizer_batch_size':32,'generation_batch_size':4,'microbatch':1,'gradient_accumulation_steps':32,
    'checkpoint_every':1,'checkpoint_retention':'last three plus step 1 and every 50th',
    'prior_source_receipt_sha256':sha(root/'launch-receipt.json'),
    'rng_limitation':'Legacy checkpoint has no RNG state. Row position resumes; engine sampling RNG restarts.',
    'automatic_restart':False}
write(root/'launch-b32-receipt.json',receipt)
args=['systemd-run','--user','--unit='+unit,'--property=Delegate=yes',
    '--property=WorkingDirectory='+str(project),
    '--property=StandardOutput=append:'+str(root/'training-b32.log'),
    '--property=StandardError=append:'+str(root/'training-b32.log'),
    '--setenv=TOKENIZERS_PARALLELISM=false','--setenv=HF_HUB_OFFLINE=1',
    '--setenv=VLLM_ENABLE_V1_MULTIPROCESSING=0','--setenv=PYTHONUNBUFFERED=1',
    sys.executable,'-I',str(project/'scripts/run_ppo_ondemand.py'),'--root',str(root),
    '--runtime',str(runtime),'--optimizer-batch-size','32','--resume-checkpoint',str(checkpoint),
    '--checkpoint-every','1','--logprob-chunk-size','1024']
subprocess.run(args,check=True)
active={'project':project.name,'runtime':runtime.name,'receipt':'launch-b32-receipt.json',
    'service':unit+'.service','optimizer_batch_size':32,'resumed_from_step':1,
    'prior_run_runtime':'training','prior_run_discarded_after_step':1}
write(root/'active-run.json',active)
print(json.dumps(active))
