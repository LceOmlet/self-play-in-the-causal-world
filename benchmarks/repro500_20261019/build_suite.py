"""Freeze 100 native outputs per family; never regenerate a frozen task at reset."""
import argparse
import concurrent.futures
import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen/owner'))
sys.path.insert(0,str(ROOT/'frozen/protocol'))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from cpt_world import world_space as ws,query_truth as qt
from cpt_world.world import OutcomeTape
from cpt_world.world_runtime import WorldSpecEpisode
from cpt_world.counterfactual_isolation import compute_counterfactual_truth_isolated
import tool_mode

FAMILIES=('ate','individual_counterfactual_probability','backadj_minimal_sets','best_intervention','mediator_set')
START=202610210000
FORMAT='cpt-world-capsule-v1'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def native(family,index,ordinal):
    captured=[];original=ws.assemble_sampled_anchor_tasks
    def capture(*args,**kwargs):
        result=original(*args,**kwargs);captured.extend(result);return result
    ws.assemble_sampled_anchor_tasks=capture
    try:
        (seed,)=ws.iter_sampled_seeds(ws.WorldGrammar(),query_types=(family,),start_seed=index,count=1,
            best_intervention_balance_start=ordinal if family=='best_intervention' else None)
    finally:ws.assemble_sampled_anchor_tasks=original
    assert len(captured)==1 and captured[0][1]==seed
    return captured[0]


def one(family,ordinal,output):
    output=Path(output);task_id=f'{family}-{ordinal:03d}';target=output/'tasks'/f'{task_id}.json.gz'
    if target.exists():
        with gzip.open(target,'rt',encoding='utf-8') as f:capsule=json.load(f)
        return {'task_id':task_id,'family':family,'ordinal':ordinal,'status':'existing','sha256':digest(capsule)}
    started=time.monotonic();base=START+FAMILIES.index(family)*10000;rejections=[]
    attempt=0
    while True:
        input_seed=base+ordinal+100*attempt
        world,seed=native(family,input_seed,ordinal)
        if family=='individual_counterfactual_probability':
            try:
                truth=compute_counterfactual_truth_isolated(world,seed,endpoint_time_limit_seconds=5.,
                    diagnostic_dir=output/'generation-diagnostics')
            except RuntimeError as exc:
                rejections.append({'input_seed':input_seed,'reason':str(exc)})
                attempt+=1
                if attempt>=10000:raise RuntimeError('Native CF admission exhausted; do not substitute an uncertified task')
                continue
        else:truth=qt.compute_query_truth(world,seed)
        break
    proposal=int(seed['seed_id'].split('-',2)[1]);anchor=int(seed['seed_id'].rsplit('-a',1)[1])
    tape_key=f'repro500-v1:{task_id}:outcome-tape-20261019'
    episode=WorldSpecEpisode(world,seed,OutcomeTape(tape_key),terminal_truth=truth,max_graph_nodes=16)
    row=tool_mode.prepare_row({'prompt':list(episode.initial_messages()),'sample_index':proposal,
        'query_type':family,'anchor_index':anchor,'tape_key':tape_key,
        'terminal_truth_json':json.dumps(truth,ensure_ascii=False,sort_keys=True), 'task_id':task_id})
    world_data=json.loads(json.dumps(dataclasses.asdict(world),allow_nan=False))
    seed=json.loads(json.dumps(seed,allow_nan=False))
    truth=json.loads(json.dumps(truth,allow_nan=False))
    capsule={'format':FORMAT,'task_id':task_id,'family':family,'ordinal':ordinal,
        'world':world_data,'seed':seed,'tape_key':tape_key,'terminal_truth':truth,
        'max_graph_nodes':16,'row':row,
        'sampling':{'input_seed':input_seed,'family_start_seed':base,'ordinal':ordinal,'attempt':attempt,
            'native_cf_rejections':rejections,'best_balance_slot':ordinal if family=='best_intervention' else None},
        'components_sha256':{'world':digest(world_data),'seed':digest(seed),'terminal_truth':digest(truth),
            'prompt':digest(row['prompt']),'tape_key':digest(tape_key)}}
    blob=canonical(capsule);temp=target.with_suffix('.tmp')
    with temp.open('wb') as out:
        with gzip.GzipFile(fileobj=out,mode='wb',mtime=0,filename='') as gz:gz.write(blob)
    temp.replace(target)
    return {'task_id':task_id,'family':family,'ordinal':ordinal,'status':'frozen',
        'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'seconds':time.monotonic()-started,
        'native_cf_rejections':len(rejections)}


def source_manifest():
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'frozen').rglob('*') if p.is_file() and p.suffix in ('.py','.in')}
    versions={}
    for name in ('numpy','scipy','pyscipopt','highspy','torch','transformers','trl','peft','vllm','jupyter_client','ipykernel'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'files':files,'sha256':digest(files),'python':sys.version,'packages':versions,
        'protocol':tool_mode.PROTOCOL_VERSION,'reward_version':'terminal-quality-v10'}


def finalize(output):
    entries=[]
    for family in FAMILIES:
        for ordinal in range(100):
            task_id=f'{family}-{ordinal:03d}';path=output/'tasks'/f'{task_id}.json.gz'
            with gzip.open(path,'rt',encoding='utf-8') as f:capsule=json.load(f)
            assert capsule['task_id']==task_id and capsule['family']==family
            entries.append({'task_id':task_id,'family':family,'ordinal':ordinal,
                'file':f'tasks/{task_id}.json.gz','sha256':digest(capsule),
                'compressed_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    selection={}
    rank=lambda family,ordinal:hashlib.sha256(f'repro500-v1:eval100:{family}:{ordinal}'.encode()).hexdigest()
    for family in FAMILIES:
        if family=='best_intervention':
            concordant=sorted([i for i in range(100) if i%5==0],key=lambda i:rank(family,i))[:4]
            reversal=sorted([i for i in range(100) if i%5!=0],key=lambda i:rank(family,i))[:16]
            ids=sorted(concordant+reversal)
        else:ids=sorted(sorted(range(100),key=lambda i:rank(family,i))[:20])
        selection[family]=[f'{family}-{i:03d}' for i in ids]
    evaluation={'format':'cpt-world-eval100-v1','selection_rule':'SHA256 rank of family and ordinal, fixed independently of task content/outcomes; best intervention retains 4 concordant and 16 strong reversal.',
        'task_ids_by_family':selection,'task_ids':[t for f in FAMILIES for t in selection[f]],'per_family':20,'total':100}
    manifest={'format':'cpt-world-suite-v1','tasks':entries,'per_family':100,'total':500,
        'tasks_merkle_input_sha256':digest(entries),'eval100_sha256':digest(evaluation),
        'owner':source_manifest(),'sampling':{'start_seed':START,'family_stride':10000,'count':100,
        'cf_candidate_stride':100,'only_native_cf_certification_rejections':True,'effect_filtering':False,
        'best_mix':{'concordant':20,'strong_reversal':80}},
        'restore_contract':'Restore explicit world, exact round-trip CPT floats, query/labels/masks/budget, OutcomeTape key, prepared prompt, native truth and frozen owner. No generation or timed CF admission runs during restore.'}
    for name,value in [('manifest.json',manifest),('eval100.json',evaluation)]:
        (output/name).write_bytes(canonical(value)+b'\n')
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data');p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);(args.output/'tasks').mkdir(exist_ok=True)
    (args.output/'sampling-plan.json').write_bytes(canonical({'family_count':100,'families':FAMILIES,'start_seed':START,
        'eval_rule':'SHA256 rank on family and ordinal; no scores/difficulty in selection','protocol':tool_mode.PROTOCOL_VERSION})+b'\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool,(args.output/'generation.jsonl').open('a',encoding='utf-8') as out:
        futures=[pool.submit(one,f,i,str(args.output)) for i in range(100) for f in FAMILIES]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();out.write(json.dumps(result,separators=(',',':'))+'\n');out.flush();print(json.dumps(result),flush=True)
    manifest=finalize(args.output);print(json.dumps({'event':'suite_frozen','total':manifest['total'],'dataset_sha256':manifest['tasks_merkle_input_sha256']}),flush=True)


if __name__=='__main__':main()
