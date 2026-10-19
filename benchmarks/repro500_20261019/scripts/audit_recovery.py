"""Validate all 200 saved first replies for deterministic recovery; no API calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_9b import count_tokens

suite=Path(__file__).resolve().parents[1]
root=suite.parent
profile=json.loads((suite/'model_profile.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(profile['base_model_path'],local_files_only=True)
report={'api_calls':0,'tool_dispatches':0,'checked':0,'models':{}}
for model,folder in [('checkpoint2350','qwen-transcripts'),('base','base-transcripts')]:
    records={}
    for path in sorted((root/folder).glob('*.json')):
        if path.name=='run_summary.json':
            continue
        raw=path.read_bytes();record=json.loads(raw)
        assert record['model']==model and record['model_requests']==1
        assert record['error_type']=='TypeError' and record['termination_reason']=='runner_or_owner_failure'
        assert not any(e['kind']=='tool' for e in record['events'])
        turns=[e for e in record['events'] if e['kind']=='completion']
        assert len(turns)==1 and 'response' in turns[0]
        first=turns[0]
        initial=record['messages'][:-1]
        n=count_tokens(tokenizer,initial,record['tools'],generation=True)
        assert n>2 and n+first['request']['max_tokens']<=profile['max_model_len']
        expected={'model':profile['models'][model],'messages':initial,'tools':record['tools'],
            'tool_choice':'auto','temperature':profile['temperature'],'seed':profile['seed'],
            'stream':False,'max_tokens':min(8192,32768-n),'chat_template_kwargs':{'enable_thinking':False}}
        assert first['request']==expected
        assert record['messages'][-1]==first['response']['choices'][0]['message']
        after=count_tokens(tokenizer,record['messages'],record['tools'],generation=False)
        assert after<=32768
        records[record['task_id']]={'source_sha256':hashlib.sha256(raw).hexdigest(),
            'correct_initial_tokens':n,'after_reply_tokens':after,'original_counter':first['prompt_tokens_tokenizer']}
    assert len(records)==100
    report['models'][model]=records
    report['checked']+=len(records)
report['status']='all_saved_responses_recoverable_without_resampling'
print(json.dumps(report,sort_keys=True))
