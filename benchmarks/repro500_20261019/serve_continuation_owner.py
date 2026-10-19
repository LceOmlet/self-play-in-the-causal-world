"""Fresh owner for timeout continuation; completed case bytes stay unchanged."""
import argparse
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil

import resume_timeouts
from serve_base_owner import BaselineExam
import server


class ContinuationExam(BaselineExam):
    def __init__(self,args):
        evaluation=json.loads((args.data/'eval100.json').read_text())
        model,records=resume_timeouts.source_records(args.source_transcripts,evaluation['task_ids'])
        assert model==args.model_label
        controls=json.loads((args.source_runtime/'private-control.json').read_text())
        progress=json.loads((args.source_runtime/'progress.json').read_text())
        assert progress['qwen']['completed']==progress['qwen']['total']==100
        manifest=json.loads((args.data/'manifest.json').read_text())
        assert controls['dataset_sha256']==manifest['tasks_merkle_input_sha256']
        assert controls['model_label']==args.model_label
        assigned=[t for role in controls['roles'] if role['kind']=='qwen' for t in role['task_ids']]
        assert len(assigned)==len(set(assigned))==100 and set(assigned)==set(evaluation['task_ids'])
        super().__init__(args)
        copied={}
        for task_id,(path,record,status) in records.items():
            if status!='complete':continue
            # Copy opaque owner records, without parsing a score or restarting a case.
            old=args.source_runtime/'qwen'/task_id
            target=self.runtime/'qwen'/task_id
            assert (old/'private-result.json').is_file()
            shutil.copytree(old,target)
            copied[task_id]={str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in old.rglob('*') if p.is_file()}
            self.cases[('qwen',task_id)]={'ended':True,'imported_completion':True}
        server.write(self.runtime/'continuation-proof.json',{'completed_cases_copied':len(copied),
            'copied_owner_files':copied,'source_transcripts_sha256':{
                t:hashlib.sha256(p.read_bytes()).hexdigest() for t,(p,_,_) in records.items()},
            'scores_parsed':False,'terminal_cases_resampled':0})
        self.progress()

    def get_case(self,role,task_id):
        prior=self.cases.get((role['kind'],task_id))
        if role['kind']=='qwen' and prior and prior.get('imported_completion'):
            raise ValueError('Terminal source case preserved; refusing to reopen')
        return super().get_case(role,task_id)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=server.ROOT/'data')
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--agent-runtime',type=Path,required=True)
    parser.add_argument('--source-runtime',type=Path,required=True)
    parser.add_argument('--source-transcripts',type=Path,required=True)
    parser.add_argument('--model-label',choices=('base','checkpoint2350'),required=True)
    parser.add_argument('--port',type=int,default=18733)
    parser.add_argument('--tokenizer',default='/home/chen/models/Qwen/Qwen3.5-9B')
    args=parser.parse_args()
    exam=ContinuationExam(args)
    http=ThreadingHTTPServer(('127.0.0.1',args.port),server.Handler);http.exam=exam
    print(json.dumps({'ready':True,'port':args.port,'model':args.model_label}),flush=True)
    try:http.serve_forever()
    finally:exam.factory.close()


if __name__=='__main__':main()
