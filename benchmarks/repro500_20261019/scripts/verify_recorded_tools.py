"""Replay one censored trajectory's recorded tools, with zero inference calls.

Uses a separate disposable owner. Never reads grades or emits tool contents.
The probe is excluded from all formal model counts and comparison scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import resume_timeouts
from run_9b import owner_request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--probe-runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    controls=json.loads((args.probe_runtime/'private-control.json').read_text())
    role=next(r for r in controls['roles'] if r['kind']=='qwen')
    url='http://127.0.0.1:'+str(controls['port'])+'/'+role['token']
    selected=None
    for path in sorted(args.source.glob('*.json')):
        if path.name=='run_summary.json':continue
        record=json.loads(path.read_text())
        if 'ended_unix' not in record:continue
        if resume_timeouts.classification(record)=='timeout':
            tools=[e for e in record['events'] if e['kind']=='tool']
            if tools:selected=(path,record,tools);break
    assert selected is not None,'No closed timeout trajectory with recorded tools'
    path,source,tools=selected
    report={'task_id':source['task_id'],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'formal_evaluation_case':False,'model_api_requests':0,'grades_read':0,
        'recorded_tool_count':len(tools),'tool_comparisons':[],'all_feedback_identical':False}
    opened=False
    try:
        prompt=owner_request(url,{'op':'prompt','task_id':source['task_id']},60)
        opened=True
        assert prompt['tools']==source['tools']
        assert prompt['messages']==source['messages'][:len(prompt['messages'])]
        assert prompt['ended'] is False
        for index,event in enumerate(tools):
            result=owner_request(url,event['request'],120)
            old=json.dumps(event['response'],sort_keys=True,ensure_ascii=False).encode()
            new=json.dumps(result,sort_keys=True,ensure_ascii=False).encode()
            report['tool_comparisons'].append({'index':index,'tool':event['request']['op'],
                'old_receipt_sha256':hashlib.sha256(old).hexdigest(),
                'new_receipt_sha256':hashlib.sha256(new).hexdigest(),
                'identical':result==event['response']})
            assert result==event['response'],'Recorded tool feedback did not reproduce'
        report['all_feedback_identical']=True
    finally:
        if opened:
            owner_request(url,{'op':'finish','task_id':source['task_id'],
                'request_id':uuid.uuid4().hex,'reason':'Excluded infrastructure replay probe'},60)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='tool_comparisons'}),flush=True)


if __name__=='__main__':main()
