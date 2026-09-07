"""Check the analytic three-node case and all eligible frozen worlds."""
import dataclasses
from fractions import Fraction as F
import json
import pickle
import time
from pathlib import Path

from cpt_world import counterfactual_solver as s
from cpt_world import query_truth as q
from cpt_world.world_space import WorldSpec
from response_envelope import response_envelope

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/chen/runs/kernel-root-cause-20260907')
world = WorldSpec(family='test_dag',topology='odd-mediator-consistency',variables=('X','M','Y'),
    domains=(2,3,2),state_names=(('0','1'),('0','1','2'),('0','1')),
    parents={0:(),1:(0,),2:(1,)},edges=((0,1),(1,2)),
    cpt={0:((F(1,2),F(1,2)),),1:((F(1,3),)*3,)*2,2:((F(1,2),F(1,2)),)*3})
envelope = response_envelope(world,0,2,baseline_value=0,treatment_value=1,
                             outcome_events=((0,),(1,)),time_limit_seconds=5)
assert abs(envelope.lower) < 1e-9 and abs(envelope.upper-1/3) < 1e-9
records = [{'index':'analytic-m3',**dataclasses.asdict(envelope)}]
for index in range(74):
    world,seed=pickle.loads((DATA/f'task-{index}.pkl').read_bytes())
    query=seed['query']
    x=q._resolve_seed_node(world,seed,query['treatment'])
    y=q._resolve_seed_node(world,seed,query['outcome'])
    affected=q._descendants(world,x)&(q._ancestors(world,y)|{y})
    if len(affected)!=2 or x in world.parents[y]:
        continue
    a=q._state_index_for_node(world,x,query['factual_value'])
    b=q._state_index_for_node(world,x,query['counterfactual_value'])
    left=q._state_index_for_node(world,y,query['factual_outcome_state'])
    right=q._state_index_for_node(world,y,query['outcome_state'])
    events=((left,),(right,))
    quotient=s._coarsen_terminal_event_outcome(world,y,events)
    if quotient is not None:
        world,events=quotient
    begin=time.perf_counter()
    envelope=response_envelope(world,x,y,baseline_value=a,treatment_value=b,
                               outcome_events=events,time_limit_seconds=5)
    record={'index':index,'seconds':time.perf_counter()-begin,
            'envelope':None if envelope is None else dataclasses.asdict(envelope)}
    records.append(record)
    print(json.dumps(record),flush=True)
    (ROOT/'response-envelope-evidence.json').write_text(json.dumps(records,indent=2))
