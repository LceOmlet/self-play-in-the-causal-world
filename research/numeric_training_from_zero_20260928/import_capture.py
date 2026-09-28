"""Verify a downloaded monitoring archive, safely extract, render, update latest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--archive',required=True,type=Path)
parser.add_argument('--sha256',required=True)
args = parser.parse_args()
actual = hashlib.sha256(args.archive.read_bytes()).hexdigest()
assert actual == args.sha256, (actual,args.sha256)
out = root/'checks'
out.mkdir(exist_ok=True)
with tarfile.open(args.archive) as archive:
    tops = set()
    for member in archive.getmembers():
        assert member.isfile() or member.isdir()
        target = (out/member.name).resolve()
        assert target.is_relative_to(out.resolve()) and target != out.resolve()
        assert member.size < 128*1024*1024
        tops.add(Path(member.name).parts[0])
    assert len(tops)==1
    capture = out/next(iter(tops))
    assert not capture.exists(), 'Keep prior evidence immutable; use a new capture'
    archive.extractall(out,filter='data')
(capture/'download-verification.json').write_text(json.dumps({'sha256':actual,'verified':True},indent=2))
subprocess.run([sys.executable,str(root/'render_report.py'),'--capture-dir',str(capture)],check=True)
latest = root/'latest'
latest.mkdir(exist_ok=True)
for name in ['summary.json','capture.json','status.json','rewards.png','task-errors.png',
             'resources.png','rollout-length.png','budget-usage.png']:
    shutil.copy2(capture/name,latest/name)
(root/'latest.json').write_text(json.dumps({'capture_dir':str(capture),'sha256':actual},indent=2))
