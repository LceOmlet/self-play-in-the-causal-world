"""Persistent state across long idle periods and trusted-caller cancellation."""
import json
from pathlib import Path
import threading
import time
import _thread
from secure_backend import NsjailSessionFactory

base=Path(__file__).parent.parent
out=base/'idle-security'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
factory=NsjailSessionFactory(out/'sessions')
results=[]
try:
    s=factory()
    assert s.execute('x=41;print(x)',{}).strip()=='41'
    time.sleep(35)
    assert s.execute('print(x+1)',{}).strip()=='42'
    assert 'frozen 1' in (s.cgroup_paths[0]/'cgroup.events').read_text()
    results.append({'name':'state_after_35_second_idle','passed':True})
    # A trusted host interruption must also kill the answer, not bypass timeout.
    timer=threading.Timer(.5,_thread.interrupt_main);timer.start()
    try:s.execute('while True: pass',{})
    except KeyboardInterrupt:pass
    finally:timer.cancel();s.close()
    assert s.closed
    assert not any(t.is_alive() for t in s.worker_threads)
    results.append({'name':'caller_cancellation_cleanup','passed':True})
    s=factory()
    reply=s.execute("print('数'*3000)",{})
    assert s.closed and json.loads(reply)['error']=='output_limit',reply
    results.append({'name':'explicit_failure_reply_cleanup_race','passed':True})
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results),flush=True)
    print('EVIDENCE='+str(out),flush=True)
finally:factory.close()
