"""Synthetic-only checks that tunnel/fallback/timeout preserve one request UUID."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import urllib.error
from unittest.mock import patch

CLIENT = Path(__file__).resolve().parents[1]/'public_client.py'
REQUEST = {'op':'act','task_id':'synthetic-only','command':{'sample':1},
    'request_id':'synthetic-same-uuid'}


class Reply:
    def __enter__(self):
        return self
    def __exit__(self,*args):
        pass
    def read(self):
        return b'{"output":"synthetic receipt","ended":false}'


def main():
    for use_tunnel in (True,False):
        payloads = []
        forwards = []
        output = io.StringIO()
        def forward(request, **kwargs):
            forwards.append(request.data.decode('utf-8'))
            if use_tunnel:
                return Reply()
            raise urllib.error.URLError('synthetic disconnected tunnel')
        def ssh(*args,**kwargs):
            payloads.append(kwargs['input'])
            if len(payloads)==1:
                raise subprocess.TimeoutExpired('synthetic-ssh',125)
            return SimpleNamespace(returncode=0,stdout=Reply().read().decode('utf-8'))
        with patch.object(sys,'argv',[str(CLIENT),'synthetic-token']), \
             patch.object(sys,'stdin',io.StringIO(json.dumps(REQUEST))), \
             patch('urllib.request.urlopen',side_effect=forward), \
             patch('subprocess.run',side_effect=ssh), contextlib.redirect_stdout(output):
            try:
                runpy.run_path(str(CLIENT),run_name='__main__')
            except SystemExit as exc:
                assert exc.code==0
        assert json.loads(output.getvalue())['output']=='synthetic receipt'
        assert len(forwards)==1 and json.loads(forwards[0])==REQUEST
        if use_tunnel:
            assert not payloads
        else:
            assert len(payloads)==2 and payloads[0]==payloads[1]==forwards[0]
        print(('persistent tunnel' if use_tunnel else 'failed tunnel + SSH timeout recovery')+': same UUID/payload PASS')


if __name__=='__main__':
    main()
