"""Launch fresh PPO only after pinned official tests and real-load checks pass."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def read(path):
    return json.loads(path.read_text())


def main(root):
    project=root/'project'
    assert not (root/'training').exists(), 'Refuse implicit restart/resampling of a prior run'
    tests={}
    for name,count in (('official-ppo-tests.xml',34),('official-ppo-policy-tests.xml',11)):
        suites=ET.parse(root/name).getroot().findall('.//testsuite')
        assert sum(int(s.get('tests','0')) for s in suites)==count
        assert all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
        tests[name]=count
    head=subprocess.check_output(['git','-C',str(root/'verl-source'),'rev-parse','HEAD'],text=True).strip()
    assert head=='d1390eea0c94467d4ae1387eeda9e2f78f8d73ea'
    assert not subprocess.check_output(['git','-C',str(root/'verl-source'),'diff','--name-only'],text=True).strip()
    owner=read(root/'owner-preflight.json');model=read(root/'model-preflight.json')
    assert owner['passed'] and model['passed'] and model['weight_transfer']['passed']
    assert model['formal_training_reuses_no_preflight_weights']
    for trial in model['full_32k_b4_trials']:
        assert trial['total_context_tokens']==131072 and trial['rollouts']==4
        assert trial['microbatch']==1 and trial['grad_accumulation']==4 and trial['first_lora_B_changed']
    selected=model['selected_logprob_chunk_size']
    load=read(root/'learning-load-contract.json')
    assert load['load']['context_length']==32768 and load['load']['generation_batch_size']==4
    assert load['resume_checkpoint'] is None and load['fresh_lora']
    generation=Path('/home/chen/runs/repro-benchmark500-20261019/vllm-fixed-schedule-graphs-20261021/official-report.json')
    g=read(generation)
    assert g['passed'] and g['engine_profile']=='decode_graph_sync'
    assert g['compilation_config']=={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'}
    assert g['environment']=={'VLLM_ENABLE_V1_MULTIPROCESSING':'0'}
    hashes={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in project.rglob('*') if p.is_file() and (p.suffix in ('.py','.in') or 'runtime_plugins' in p.parts) and '__pycache__' not in p.parts}
    receipt={'official_ppo_tests_passed':tests,'verl_commit':head,'model_and_owner_preflights_passed':True,
             'selected_logprob_chunk_size':selected,'fresh_base_model':load['model'],
             'learning_load':load['load'],'source_sha256':hashes,
             'decoding_evidence_sha256':hashlib.sha256(generation.read_bytes()).hexdigest(),
             'decoding_evidence_scope':'prior matching fixed-schedule base-weight greedy graph tests; not a PPO numerical tolerance',
             'no_preflight_parameters_reused':True,'automatic_resume':False}
    (root/'launch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    args=['systemd-run','--user','--unit=cpt-numerics-ppo-ondemand-from-base-20261022',
          '--property=Delegate=yes',f'--property=WorkingDirectory={project}',
          f'--property=StandardOutput=append:{root}/training.log',f'--property=StandardError=append:{root}/training.log',
          '--setenv=TOKENIZERS_PARALLELISM=false','--setenv=HF_HUB_OFFLINE=1',
          '--setenv=VLLM_ENABLE_V1_MULTIPROCESSING=0','--setenv=PYTHONUNBUFFERED=1',
          sys.executable,'-I',str(project/'scripts/run_ppo_ondemand.py'),
          '--root',str(root),'--logprob-chunk-size',str(selected)]
    subprocess.run(args,check=True)
    print(json.dumps({'started_service':'cpt-numerics-ppo-ondemand-from-base-20261022.service',
                      'official_ppo_tests_passed':45,'chunk':selected,'runtime':str(root/'training')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chen/runs/ppo-ondemand-from-base-20261022'))
    main(p.parse_args().root)
