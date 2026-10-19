"""Synthetic opaque terminal-record copying; no real capsule or grade reads."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import serve_continuation_owner as continuation


def main():
    original_init=continuation.BaselineExam.__init__
    original_progress=continuation.ContinuationExam.progress
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';data.mkdir()
            source=root/'source';source.mkdir();traces=root/'traces';traces.mkdir()
            ids=['synthetic-'+str(i) for i in range(100)]
            (data/'eval100.json').write_text(json.dumps({'task_ids':ids}))
            (data/'manifest.json').write_text(json.dumps({'tasks_merkle_input_sha256':'fixture'}))
            (source/'private-control.json').write_text(json.dumps({'dataset_sha256':'fixture',
                'model_label':'checkpoint2350','roles':[{'kind':'qwen','task_ids':ids}]}))
            (source/'progress.json').write_text(json.dumps({'qwen':{'completed':100,'total':100}}))
            (traces/'run_summary.json').write_text(json.dumps({'model':'checkpoint2350',
                'tasks':[{'task_id':t} for t in ids]}))
            for index,task_id in enumerate(ids):
                record={'task_id':task_id,'model':'checkpoint2350','ended_unix':1,
                    'termination_reason':'owner_ended','events':[]}
                folder=source/'qwen'/task_id;folder.mkdir(parents=True)
                # Invalid JSON proves this is copied, not parsed as a grade.
                (folder/'private-result.json').write_bytes(b'OPAQUE NON-JSON COMPLETION')
                if index==99:
                    record.update(termination_reason='model_api_failure',events=[
                        {'kind':'completion','error_type':'TimeoutError'}])
                (traces/(hashlib.sha256(task_id.encode()).hexdigest()+'.json')).write_text(json.dumps(record))
            def stub_init(self,args):
                self.runtime=args.runtime;self.runtime.mkdir();self.cases={}
            continuation.BaselineExam.__init__=stub_init
            continuation.ContinuationExam.progress=lambda self:None
            args=SimpleNamespace(data=data,source_runtime=source,source_transcripts=traces,
                runtime=root/'continued',model_label='checkpoint2350')
            exam=continuation.ContinuationExam(args)
            assert len(exam.cases)==99
            assert ('qwen',ids[-1]) not in exam.cases
            proof=json.loads((args.runtime/'continuation-proof.json').read_text())
            assert proof['completed_cases_copied']==99 and proof['scores_parsed'] is False
            assert all((args.runtime/'qwen'/t/'private-result.json').read_bytes()==
                b'OPAQUE NON-JSON COMPLETION' for t in ids[:-1])
            try:exam.get_case({'kind':'qwen'},ids[0])
            except ValueError:pass
            else:raise AssertionError('Completed cases must not be reopened')
            print('99 terminal cases copied byte-for-byte without parsing grades: PASS')
            print('Only timeout case gets a fresh episode; completed case reopening rejected: PASS')
    finally:
        continuation.BaselineExam.__init__=original_init
        continuation.ContinuationExam.progress=original_progress


if __name__=='__main__':main()
