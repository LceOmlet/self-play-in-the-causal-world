"""Jupyter's manager, client and parser all live INSIDE the answer sandbox.

Only bounded bytes cross to the trusted host. This is one numerical execution
adapter, not a tool router or interpreter. Never use Connection.recv()/pickle.
"""
import importlib.util
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

MAX_REQUEST = 2 * 1024 * 1024  # comfortably above the original 32K-token input
MAX_TEXT = 8192


def main():
    connection = Connection(int(sys.argv[1]))
    os.set_inheritable(connection.fileno(), False)
    spec = importlib.util.spec_from_file_location('numeric_adapter', '/opt/bridge/tool_mode.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    # Kernel discovery is pinned to a readonly, root-owned kernelspec.
    manager = KernelManager(kernel_name='numerics', transport='ipc', ip='/tmp/jupyter',
                            connection_file='/tmp/kernel.json', autorestart=False,
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=['/opt/kernels']))
    kernel_log = open('/tmp/kernel-stderr', 'w+b')
    manager.start_kernel(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=kernel_log, close_fds=True)
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=15)
    except Exception as error:
        kernel_log.seek(0, 2)
        kernel_log.seek(max(0, kernel_log.tell() - 3000))
        raise RuntimeError(kernel_log.read().decode('utf8', errors='replace')) from error

    def destroy():
        manager.shutdown_kernel(now=True)
        client.stop_channels()

    session = adapter.JupyterNumericSession(client, destroy)
    connection.send_bytes(b'READY-v2')
    try:
        while True:
            request = connection.recv_bytes(MAX_REQUEST)
            nonce, payload = request[:16], request[16:]
            row = json.loads(payload)
            result = session.execute(row['code'], row['public'])
            raw = result.encode('utf8')
            if len(raw) > MAX_TEXT:
                raw = b'{"error":"output_limit","state_lost":true}'
                session.close()
            connection.send_bytes(nonce + (b'E' if session.closed else b'O') + raw)
            if session.closed:
                break
    finally:
        session.close()
        connection.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        # Startup diagnostics are inside the sandbox; the host still bounds them.
        Connection(int(sys.argv[1])).send_bytes(b'INIT_ERROR:' + traceback.format_exc().encode()[-4096:])
        raise
