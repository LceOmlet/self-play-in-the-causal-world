"""Single-case transport only. Numerical analysis belongs in owner numerics."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


def prepare_request(capability, request):
    if request.get('op') not in ('prompt', 'act', 'numerics', 'finish'):
        raise ValueError('Only this case prompt, act, numerics or finish is allowed')
    assigned = capability['task_id']
    if request.get('task_id', assigned) != assigned:
        raise ValueError('Capability belongs to a different task')
    request_id = request.get('request_id', uuid.uuid4().hex)
    if not isinstance(request_id, str):
        raise ValueError('Request ID must be a string')
    # The native transport may name calls "a001". Namespace short/long labels
    # deterministically into the owner's UUID field; retries still use the
    # identical wire payload. Valid existing IDs are never changed.
    if not 8 <= len(request_id) <= 128:
        request_id = 'native-v2-' + hashlib.sha256(request_id.encode()).hexdigest()
    return {**request, 'task_id': assigned, 'request_id': request_id}


def main():
    capability = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    request = prepare_request(capability, json.load(sys.stdin))
    payload = json.dumps(request, ensure_ascii=False)
    remote = '''import sys,urllib.request
token=sys.argv[1]
req=urllib.request.Request('http://127.0.0.1:18735/'+token,data=sys.stdin.buffer.read(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=100) as res:sys.stdout.buffer.write(res.read())
'''
    command = shlex.join(['/home/chen/runs/official-grpo-rollout-repair-20260928/venv/bin/python',
        '-c', remote, capability['token']])
    for attempt in range(3):
        try:
            result = subprocess.run(['ssh', '-p', '10179', '-o', 'BatchMode=yes', '-o',
                'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
                'chen@e1b4acb6c4c142d5a00d326cec07fb11.hn.takin.cc', command],
                input=payload, capture_output=True, text=True, encoding='utf-8', timeout=125)
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise RuntimeError('Transport timed out; retain this exact request_id') from None
            continue
        if result.returncode == 0:
            print(json.dumps(json.loads(result.stdout), ensure_ascii=False))
            return
        if attempt == 2:
            raise RuntimeError('Transport failed; retain this exact request_id')


if __name__ == '__main__':
    main()
