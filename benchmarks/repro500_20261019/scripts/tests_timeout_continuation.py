"""Synthetic saved-response/tool replay and censored-request continuation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import resume_timeouts as recovery
import run_9b


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return {'input_ids':[0]*100,'attention_mask':[1]*100}


def main():
    profile=json.loads(Path(run_9b.__file__).with_name('model_profile.json').read_text())
    prompt=[{'role':'user','content':'Synthetic only'}]
    def assistant(index,op,arguments):
        return {'role':'assistant','content':None,'tool_calls':[{'id':'call-'+str(index),
            'type':'function','function':{'name':op,'arguments':json.dumps(arguments)}}]}
    first=assistant(0,'numerics',{'code':'print(2)'})
    final=assistant(1,'act',{'command':{'answer':'synthetic'}})
    def payload(messages):
        return {'model':profile['models']['checkpoint2350'],'messages':copy.deepcopy(messages),
            'tools':[],'tool_choice':'auto','temperature':profile['temperature'],
            'seed':profile['seed'],'stream':False,'max_tokens':8192,
            'chat_template_kwargs':{'enable_thinking':False}}
    response={'choices':[{'message':first,'finish_reason':'tool_calls'}],
        'usage':{'prompt_tokens':100,'completion_tokens':10}}
    receipt={'output':'2','ended':False}
    messages=prompt+[first,{'role':'tool','tool_call_id':'call-0','content':'2'}]
    source={'task_id':'synthetic','model':'checkpoint2350','profile':profile,'tools':[],
        'messages':messages,'ended_unix':10,'termination_reason':'model_api_failure',
        'events':[{'kind':'completion','request':payload(prompt),'response':response},
            {'kind':'tool','request':{'op':'numerics','task_id':'synthetic','request_id':'old-uuid',
                'code':'print(2)'},'response':receipt},
            {'kind':'completion','request':payload(messages),'error_type':'TimeoutError'}]}
    assert recovery.classification(source)=='timeout'
    invalid=copy.deepcopy(source);invalid['events'][-1]['error_type']='HTTPError'
    try:recovery.classification(invalid)
    except AssertionError:pass
    else:raise AssertionError('Unrelated infrastructure failures must stay rejected')
    for mode in ('success','feedback_mismatch','payload_mismatch'):
        operations=[];calls=[]
        fixture=copy.deepcopy(source)
        if mode=='payload_mismatch':fixture['events'][-1]['request']['seed']+=1
        def owner(url,request,wait):
            operations.append(request['op'])
            if request['op']=='prompt':return {'messages':prompt,'tools':[],'task_id':'synthetic'}
            if request['op']=='numerics':
                return {'output':'different' if mode=='feedback_mismatch' else '2','ended':False}
            return {'output':'closed','ended':True}
        def completion(url,request,wait):
            assert mode=='success' and wait==3600
            assert request==source['events'][-1]['request']
            calls.append(copy.deepcopy(request))
            return {'choices':[{'message':final,'finish_reason':'tool_calls'}]}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);old=root/'old';out=root/'out';old.mkdir();out.mkdir()
            path=old/(hashlib.sha256(b'synthetic').hexdigest()+'.json')
            path.write_text(json.dumps(fixture),encoding='utf-8')
            recovery.continue_task('synthetic',path,fixture,tokenizer=Tokenizer(),profile=profile,
                owner_url='http://secret',model_api='http://synthetic',output_dir=out,
                owner=owner,completion=completion)
            record=json.loads((out/path.name).read_text())
            if mode=='success':
                assert len(calls)==1 and record['api_requests_this_run']==1
                assert record['reused_responses']==1 and record['termination_reason']=='owner_ended'
                assert operations==['prompt','numerics','act']
                assert record['continuation_source']['censored_request_reissued_identically']
                assert record['events'][0]['response']==response
            else:
                assert not calls and record['termination_reason'] in recovery.INFRA_REASONS
            assert 'http://secret' not in (out/path.name).read_text()
        print(mode+': PASS')
    print('Only TimeoutError without a response is eligible; unrelated failures rejected: PASS')


if __name__=='__main__':main()
