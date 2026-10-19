"""Sealed, durable, identical task interface for agents and the 9B model."""
import argparse
import copy
from datetime import datetime,timezone
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen/protocol'))
os.environ['CUDA_VISIBLE_DEVICES']=''
from capsule import FrozenEnvironment,canonical
from transformers import AutoTokenizer
from transformers.utils import get_json_schema
import tool_mode
from secure_backend import NsjailSessionFactory


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temporary.replace(path)


def append(path,value):
    with path.open('a',encoding='utf-8') as out:
        out.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**value},ensure_ascii=False,default=str)+'\n')
        out.flush();os.fsync(out.fileno())


class Exam:
    def __init__(self,args):
        self.args=args;self.runtime=args.runtime.resolve();self.data=args.data.resolve()
        self.runtime.mkdir(parents=True,exist_ok=False)
        self.factory=NsjailSessionFactory(self.runtime/'numeric-sessions')
        self.env_type=tool_mode.make_numeric_environment(FrozenEnvironment,self.factory)
        self.tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        self.manifest=json.loads((self.data/'manifest.json').read_text())
        self.entries={x['task_id']:x for x in self.manifest['tasks']}
        for name,digest in self.manifest['owner']['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        evaluation=json.loads((self.data/'eval100.json').read_text())
        assert hashlib.sha256(canonical(evaluation)).hexdigest()==self.manifest['eval100_sha256']
        self.roles={};self.cases={};self.lock=threading.RLock()
        for family,ids in evaluation['task_ids_by_family'].items():
            self.roles[secrets.token_urlsafe(30)]={'kind':'agent','family':family,'task_ids':ids}
        self.roles[secrets.token_urlsafe(30)]={'kind':'qwen','family':'all','task_ids':evaluation['task_ids']}
        write(self.runtime/'private-control.json',{'roles':[{'token':token,**role} for token,role in self.roles.items()],
            'port':args.port,'dataset_sha256':self.manifest['tasks_merkle_input_sha256'],
            'protocol':tool_mode.PROTOCOL_VERSION,'context_limit':32768,'enable_thinking_for_public_accounting':False,
            'qwen_phase_gate':'All 100 agent tasks must be terminal before a qwen task is opened.'})
        self.progress()

    def tokens(self,messages,schemas):
        encoded=self.tokenizer.apply_chat_template(messages,tools=schemas,tokenize=True,return_dict=True,
            add_generation_prompt=True,enable_thinking=False)
        ids=encoded['input_ids']
        if ids and isinstance(ids[0],list):assert len(ids)==1;ids=ids[0]
        return len(ids)

    def progress(self):
        with self.lock:return self._progress_unlocked()

    def _progress_unlocked(self):
        result={}
        for kind in ('agent','qwen'):
            groups={}
            for role in self.roles.values():
                if role['kind']!=kind:continue
                for task_id in role['task_ids']:
                    family=self.entries[task_id]['family'];counter=groups.setdefault(family,{'total':0,'started':0,'completed':0})
                    counter['total']+=1
                    case=self.cases.get((kind,task_id))
                    counter['started']+=int(case is not None);counter['completed']+=int(bool(case and case['ended']))
            result[kind]={'by_family':groups,'completed':sum(g['completed'] for g in groups.values()),
                'started':sum(g['started'] for g in groups.values()),'total':sum(g['total'] for g in groups.values())}
        write(self.runtime/'progress.json',result)
        return result

    def get_case(self,role,task_id):
        if task_id not in role['task_ids']:raise ValueError('Task is not assigned to this role')
        key=(role['kind'],task_id)
        with self.lock:
            if key in self.cases:return self.cases[key]
            state=self.progress()['agent']
            if role['kind']=='qwen' and state['completed']!=state['total']:
                raise ValueError('Qwen phase begins after all assigned agent submissions')
            entry=self.entries[task_id];path=self.data/entry['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['compressed_sha256']
            with gzip.open(path,'rt',encoding='utf-8') as f:capsule=json.load(f)
            assert hashlib.sha256(canonical(capsule)).hexdigest()==entry['sha256']
            env=self.env_type();env.reset(capsule=capsule,**capsule['row'])
            schemas=[get_json_schema(env.act),get_json_schema(env.numerics)]
            folder=self.runtime/role['kind']/task_id;folder.mkdir(parents=True,exist_ok=False)
            case={'task_id':task_id,'role':role['kind'],'family':entry['family'],'env':env,
                'prompt':capsule['row']['prompt'],'schemas':schemas,'messages':copy.deepcopy(capsule['row']['prompt']),
                'folder':folder,'ended':False,'lock':threading.Lock(),'requests':{},'calls':0,
                'initial_tokens':self.tokens(capsule['row']['prompt'],schemas),'started':time.time()}
            write(folder/'prompt.json',{'messages':case['prompt'],'tools':schemas,'capsule_sha256':entry['sha256']})
            self.cases[key]=case;self.progress();return case

    def finish(self,case,reason):
        if case['ended']:return
        env=case['env'];ep=env.episode
        # A committed terminal act was already recorded before owner execution.
        result={'task_id':case['task_id'],'family':case['family'],'role':case['role'],
            'reason':reason,'submitted':bool(ep.completed),'reward':env.get_reward(),
            'terminal_score':ep.terminal_score,'terminal_reward_exact':str(ep.terminal_reward),
            'queries_used':ep.queries_used,'observations_used':ep.observations_used,
            'remaining_budget':ep.remaining_budget,'budget':ep.budget.max_observations,
            'normalized_public_qwen_tokens':self.tokens(case['messages'],case['schemas']),
            'initial_public_qwen_tokens':case['initial_tokens'],'tool_calls':case['calls'],
            'elapsed_seconds':time.time()-case['started']}
        write(case['folder']/'private-result.json',result)
        case['ended']=True
        # get_reward already closes this answer's isolated numerical session.
        self.progress()

    def dispatch(self,token,request):
        role=self.roles[token];op=request['op']
        if op=='status':return self.progress()[role['kind']]
        if op=='next':
            for task_id in role['task_ids']:
                case=self.cases.get((role['kind'],task_id))
                if not case or not case['ended']:
                    request={**request,'op':'prompt','task_id':task_id};op='prompt';break
            else:return {'done':True,'completed':len(role['task_ids'])}
        case=self.get_case(role,request['task_id'])
        if op=='prompt':return {'task_id':case['task_id'],'messages':case['prompt'],'tools':case['schemas'],
            'ended':case['ended'],'context_limit':32768}
        if op not in ('act','numerics','finish'):raise ValueError('Unsupported operation')
        request_id=request['request_id']
        if not isinstance(request_id,str) or not 8<=len(request_id)<=128:raise ValueError('Invalid request ID')
        fingerprint=hashlib.sha256(canonical(request)).hexdigest()
        with case['lock']:
            prior=case['requests'].get(request_id)
            if prior:
                if prior['fingerprint']!=fingerprint:raise ValueError('Request ID reused with different content')
                return prior['response']
            if case['ended']:return {'output':'Episode already complete. Make no more calls for this question.','ended':True}
            append(case['folder']/'events.jsonl',{'event':'request','request':request,'fingerprint':fingerprint})
            if op=='finish':
                self.finish(case,'unfinished:'+str(request.get('reason','no terminal act'))[:300])
                response={'output':'Episode closed without a terminal answer.','ended':True}
            else:
                arguments={'command':request['command']} if op=='act' else {'code':request['code']}
                call_id='call_'+str(case['calls'])
                assistant={'role':'assistant','content':'','tool_calls':[{'id':call_id,'type':'function',
                    'function':{'name':op,'arguments':arguments}}]}
                proposed=case['messages']+[assistant]
                if self.tokens(proposed,case['schemas'])>=32768:
                    self.finish(case,'public_context_limit_before_call')
                    response={'output':'Public context limit reached; question closed.','ended':True}
                else:
                    output=str(getattr(case['env'],op)(**arguments));case['calls']+=1
                    case['messages']=proposed+[{'role':'tool','tool_call_id':call_id,'name':op,'content':output}]
                    append(case['folder']/'events.jsonl',{'event':'tool_result','request_id':request_id,
                        'tool':op,'arguments':arguments,'output':output,'remaining_budget':case['env'].episode.remaining_budget})
                    if case['env'].episode.completed:self.finish(case,'terminal_answer')
                    elif self.tokens(case['messages'],case['schemas'])>=32768:self.finish(case,'public_context_limit_after_call')
                    response={'output':output,'ended':case['ended']}
            record={'fingerprint':fingerprint,'response':response};case['requests'][request_id]=record
            write(case['folder']/'request-cache'/f'{hashlib.sha256(request_id.encode()).hexdigest()}.json',record)
            append(case['folder']/'events.jsonl',{'event':'response','request_id':request_id,'response':response})
            return response


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=2*1024*1024:raise ValueError('Request size')
            request=json.loads(self.rfile.read(n))
            result=self.server.exam.dispatch(self.path.strip('/'),request)
            status=200
        except (KeyError,ValueError) as exc:result={'error':str(exc)};status=400
        except Exception as exc:
            append(self.server.exam.runtime/'service-errors.jsonl',{'event':'service_error','type':type(exc).__name__,'error':str(exc)})
            result={'error':'Exam service error; retain the same request ID for transport recovery.'};status=500
        blob=canonical(result);self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(blob)));self.end_headers()
        try:self.wfile.write(blob)
        except (BrokenPipeError,ConnectionResetError):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--port',type=int,default=18729)
    p.add_argument('--tokenizer',default='/home/chen/models/Qwen/Qwen3.5-9B');args=p.parse_args()
    exam=Exam(args);server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.exam=exam
    print(json.dumps({'ready':True,'port':args.port,'dataset_sha256':exam.manifest['tasks_merkle_input_sha256']}),flush=True)
    try:server.serve_forever()
    finally:exam.factory.close()


if __name__=='__main__':main()
