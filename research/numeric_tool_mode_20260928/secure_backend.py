"""Bounded host adapter; Jupyter/ZeroMQ parsing executes only inside NsJail.

Framing is Python's official Connection.send_bytes/recv_bytes(maxlength), never
pickle. NsJail/Linux own isolation, memory, CPU, PID, freeze and process cleanup.
"""
from __future__ import annotations
import atexit
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import tempfile
import threading
import time
import weakref

MAX_TEXT = 8192
MAX_REPLY = 16 + 1 + MAX_TEXT
MAX_REQUEST = 2 * 1024 * 1024
CALL_SECONDS = 30


def delegated_cgroup():
    path = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
    if path.name == 'controller':
        path = path.parent
    if not path.name.startswith('cpt-numerics-') or not path.name.endswith('.service'):
        raise RuntimeError('Dedicated systemd user service with Delegate=yes required')
    if not os.access(path / 'cgroup.procs', os.W_OK):
        raise RuntimeError('cgroup delegation is absent')
    child = path / 'controller'
    child.mkdir(exist_ok=True)
    (child / 'cgroup.procs').write_text(str(os.getpid()))
    (path / 'cgroup.subtree_control').write_text('+cpu +memory +pids')
    return path


def safe_display(text):
    """Keep model-controlled output literal in chat templates and terminals."""
    return ''.join(f'\\u{ord(c):04x}' if c in '<>' or (ord(c) < 32 and c not in '\n\t') or ord(c) == 127
                   else c for c in text)


def _destroy_resources(process, cg, sock, connection, log, worker_threads):
    # Kernel-owned group kill covers descendants regardless of their sessions.
    if cg is not None and cg.exists():
        try:
            (cg / 'cgroup.kill').write_text('1')
        except FileNotFoundError:
            pass
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    for thread in worker_threads:
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError('Numerical transport worker did not terminate')
    if connection is not None: connection.close()
    if sock is not None: sock.close()
    if log is not None: log.close()
    if cg is not None and cg.exists() and (cg / 'cgroup.procs').read_text().strip():
        raise RuntimeError('Numerical sandbox descendants survived cleanup')


class BoundedAnswerSession:
    def __init__(self, factory):
        self.closed = False
        self._lock = threading.Lock()
        self._process = None
        self._cg = None
        self._connection = None
        self._socket = None
        self._log = None
        self.worker_threads = []
        start = time.monotonic()
        self.evidence_dir = Path(tempfile.mkdtemp(prefix='answer-', dir=factory.evidence))
        self._socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        for sock in (self._socket, child_socket):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self._connection = Connection(os.dup(self._socket.fileno()))
        fd = child_socket.fileno()
        cfg = (Path(__file__).with_name('nsjail-secure.cfg.in')).read_text().replace(
            '@CGROUP@', str(factory.cgroup)).replace('@FD@', str(fd))
        (self.evidence_dir / 'nsjail.cfg').write_text(cfg)
        self._log = (self.evidence_dir / 'nsjail.log').open('wb')
        try:
            self._process = subprocess.Popen(['/opt/cpt-numerics/bin/nsjail', '--config',
                                             str(self.evidence_dir / 'nsjail.cfg')],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self._log,
                    start_new_session=True, close_fds=True, pass_fds=(fd,))
            child_socket.close()
            ready = self._exchange(None, timeout=20)
            if ready != b'READY-v2':
                (self.evidence_dir / 'startup-error.txt').write_bytes(ready)
                raise RuntimeError('Numerical sandbox startup failed')
            self.supervisor_pid = self._process.pid
            children = Path(f'/proc/{self.supervisor_pid}/task/{self.supervisor_pid}/children').read_text().split()
            if len(children) != 1:
                raise RuntimeError('Unexpected sandbox process ownership')
            self._cg = factory.cgroup / f'NSJAIL.{children[0]}'
            self.cgroup_paths = [self._cg]
            self.kernel_pids = [int(p) for p in (self._cg / 'cgroup.procs').read_text().split()]
            # Every parser and numerical process must be within the same limit.
            if (self._cg / 'memory.max').read_text().strip() != '1073741824':
                raise RuntimeError('Per-answer memory protection absent')
            (self._cg / 'memory.oom.group').write_text('1')
            self._freeze(True)
            self.startup_seconds = time.monotonic() - start
            self._finalizer = weakref.finalize(self, _destroy_resources, self._process,
                self._cg, self._socket, self._connection, self._log, self.worker_threads)
        except BaseException:
            child_socket.close()
            self.close()
            raise

    def _exchange(self, request, timeout):
        # recv_bytes(maxlength) rejects the length BEFORE allocating its body.
        # A bounded worker plus socket.shutdown handles partial-header/body stalls.
        done = threading.Event()
        result = []
        def transfer():
            try:
                if request is not None:
                    self._connection.send_bytes(request)
                result.append(self._connection.recv_bytes(MAX_REPLY))
            except BaseException as error:
                result.append(error)
            finally:
                done.set()
        thread = threading.Thread(target=transfer, name='bounded-numeric-bytes', daemon=True)
        self.worker_threads.append(thread)
        thread.start()
        if not done.wait(timeout):
            raise TimeoutError('Calculation deadline exceeded')
        thread.join(timeout=1)
        self.worker_threads[:] = [t for t in self.worker_threads if t.is_alive()]
        if isinstance(result[0], BaseException):
            raise RuntimeError('Numerical transport rejected')
        return result[0]

    def _freeze(self, frozen):
        (self._cg / 'cgroup.freeze').write_text('1' if frozen else '0')
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if f'frozen {int(frozen)}' in (self._cg / 'cgroup.events').read_text():
                return
            time.sleep(0.001)
        raise RuntimeError('cgroup freeze acknowledgement failed')

    def execute(self, code, public_context):
        with self._lock:
            if self.closed:
                raise RuntimeError('Numerical session is closed')
            if not isinstance(code, str):
                raise TypeError('code must be a string')
            nonce = secrets.token_bytes(16)
            request = nonce + json.dumps({'code': code, 'public': public_context}, ensure_ascii=False).encode('utf8')
            if len(request) > MAX_REQUEST:
                return '{"error":"request_limit","message":"Numerical request exceeds 2 MiB."}'
            try:
                if self._connection.poll(0):
                    raise RuntimeError('Unsolicited sandbox bytes')
                self._freeze(False)
                reply = self._exchange(request, timeout=CALL_SECONDS)
                if len(reply) < 17 or reply[:16] != nonce or reply[16:17] not in (b'O', b'E'):
                    raise RuntimeError('Malformed sandbox reply')
                if reply[16:17] == b'O':
                    self._freeze(True)
                output = safe_display(reply[17:].decode('utf8', errors='strict'))
                if len(output.encode('utf8')) > MAX_TEXT:
                    self.close()
                    return '{"error":"output_limit","state_lost":true}'
                if reply[16:17] == b'E':
                    self.close()
                return output
            except Exception as error:
                self.close()
                reason = 'time_limit' if isinstance(error, TimeoutError) else 'sandbox_or_protocol_failure'
                return json.dumps({'error': reason, 'state_lost': True})
            except BaseException:
                self.close()
                raise

    def close(self):
        if self.closed:
            return
        if hasattr(self, '_finalizer'):
            self._finalizer()
        else:
            _destroy_resources(self._process, self._cg, self._socket,
                               self._connection, self._log, self.worker_threads)
        self.closed = True


class NsjailSessionFactory:
    def __init__(self, evidence_dir):
        self.cgroup = delegated_cgroup()
        self.evidence = Path(evidence_dir)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.sessions = weakref.WeakSet()
        atexit.register(self.close)

    def __call__(self):
        session = BoundedAnswerSession(self)
        self.sessions.add(session)
        return session

    def close(self):
        errors = []
        for session in list(self.sessions):
            try:
                session.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError('Not all numerical sessions were cleaned up') from errors[0]
        self.sessions.clear()


if __name__ == '__main__':
    factory = NsjailSessionFactory(Path(__file__).parent.parent / 'secure-smoke')
    try:
        session = factory()
        print(json.dumps({'startup_seconds':session.startup_seconds,
                          'result':session.execute('print(np.linalg.solve([[2.,0.],[0.,4.]],[6.,8.]))', {})}), flush=True)
    finally:
        factory.close()
