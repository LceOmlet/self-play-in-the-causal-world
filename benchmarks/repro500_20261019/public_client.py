"""Transport only; all task analysis must run in the question's numerics tool."""
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

REMOTE=r'''import sys,urllib.request
token=sys.argv[1]
request=urllib.request.Request('http://127.0.0.1:18729/'+token,data=sys.stdin.buffer.read(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=100) as response:sys.stdout.buffer.write(response.read())
'''

request=json.load(sys.stdin);token=sys.argv[1]
if not token.replace('-','').replace('_','').isalnum():raise ValueError('Invalid token')
request.setdefault('request_id',uuid.uuid4().hex)
payload=json.dumps(request,ensure_ascii=False)
# A persistent SSH localhost forward avoids a fresh SSH handshake per call.
# If unavailable, fall back with the EXACT same payload and request UUID.
try:
    forwarded=urllib.request.Request('http://127.0.0.1:28429/'+token,
        data=payload.encode('utf-8'),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(forwarded,timeout=100) as response:
        result=json.loads(response.read())
    print(json.dumps(result,ensure_ascii=False))
    sys.exit(0)
except (urllib.error.URLError,TimeoutError,ConnectionError):
    pass
command=shlex.join(['/home/chen/runs/official-grpo-rollout-repair-20260928/venv/bin/python','-c',REMOTE,token])
for attempt in range(3):
    try:
        result=subprocess.run(['ssh','-p','10179','-o','BatchMode=yes','-o','ConnectTimeout=20',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
            'chen@e1b4acb6c4c142d5a00d326cec07fb11.hn.takin.cc',command],
            input=payload,text=True,encoding='utf-8',capture_output=True,timeout=125)
    except subprocess.TimeoutExpired:
        if attempt==2:
            raise RuntimeError('Public transport timed out; retry only with the SAME explicit request_id') from None
        continue
    if result.returncode==0:
        response=json.loads(result.stdout)
        print(json.dumps(response,ensure_ascii=False));break
    if attempt==2:raise RuntimeError('Public transport failed; retry only with the SAME explicit request_id if needed')
    time.sleep(1)
