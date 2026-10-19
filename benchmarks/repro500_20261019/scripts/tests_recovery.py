"""Synthetic token-container/string-argument and exact first-response recovery."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_9b import count_tokens,run_task


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['return_dict'] is True
        for message in messages:
            for call in message.get('tool_calls',[]):
                assert isinstance(call['function']['arguments'],dict)
        return {'input_ids':list(range(101)), 'attention_mask':[1]*101}


def main():
    root=Path(__file__).resolve().parents[1]
    profile=json.loads((root/'model_profile.json').read_text())
    prompt=[{'role':'user','content':'Synthetic only'}]
    tools=[]
    assistant={'role':'assistant','content':None,'tool_calls':[{'id':'synthetic-call','type':'function',
        'function':{'name':'act','arguments':'{"command":{"type":"answer","value":"state_0"}}'}}]}
    before=copy.deepcopy(assistant)
    assert count_tokens(Tokenizer(),prompt+[assistant],tools,generation=False)==101
    assert assistant==before
    payload={'model':'checkpoint2350','messages':prompt,'tools':tools,'tool_choice':'auto',
        'temperature':profile['temperature'],'seed':profile['seed'],'stream':False,
        'max_tokens':8192,'chat_template_kwargs':{'enable_thinking':False}}
    response={'choices':[{'message':assistant,'finish_reason':'tool_calls'}],
        'usage':{'prompt_tokens':101,'completion_tokens':10}}
    original={'task_id':'synthetic','model':'checkpoint2350','profile':profile,'model_requests':1,
        'error_type':'TypeError','termination_reason':'runner_or_owner_failure',
        'messages':prompt+[assistant],'tools':tools,'events':[{'kind':'completion','request':payload,
        'response':response,'started_unix':1,'ended_unix':2}]}
    operations=[]
    def owner(url,request,timeout):
        operations.append(request)
        if request['op']=='prompt':
            return {'task_id':'synthetic','messages':prompt,'tools':tools}
        assert request['op']=='act'
        return {'output':'synthetic terminal','ended':True}
    def no_resampling(*args):
        raise AssertionError('A preserved first response must not be sampled again')
    with tempfile.TemporaryDirectory() as temporary:
        source,out=Path(temporary)/'old',Path(temporary)/'new'
        source.mkdir();out.mkdir()
        name=hashlib.sha256(b'synthetic').hexdigest()+'.json'
        (source/name).write_text(json.dumps(original),encoding='utf-8')
        result=run_task('synthetic',tokenizer=Tokenizer(),profile=profile,model='checkpoint2350',
            owner_url='http://synthetic-secret',model_api='http://synthetic',output_dir=out,
            owner=owner,completion=no_resampling,resume_first_responses=source)
        recovered=json.loads((out/name).read_text())
        assert result['termination_reason']=='owner_ended'
        assert [r['op'] for r in operations]==['prompt','act']
        assert recovered['api_requests_this_run']==0 and recovered['reused_responses']==1
        assert recovered['model_requests']==1 and recovered['messages'][1]==assistant
        assert recovered['usage']==response['usage']
        assert 'synthetic-secret' not in (out/name).read_text()
    print('BatchEncoding counts input_ids, string arguments counted without mutation: PASS')
    print('saved first response dispatched once; zero new API requests/resampling: PASS')


if __name__=='__main__':
    main()
