"""Read-only numerical runtime manifest; no task access."""
import hashlib
import json
from pathlib import Path

root = Path('/opt/cpt-numerics/rootfs')
packages = {}
for path in sorted(root.glob('opt/numerics/lib/python*/site-packages/*.dist-info/METADATA')):
    fields = {}
    for line in path.read_text().splitlines():
        if line.startswith(('Name: ', 'Version: ')):
            key, value = line.split(': ', 1)
            fields[key] = value
    packages[fields['Name']] = fields['Version']
files = {}
for path in sorted(root.rglob('*')):
    if path.is_symlink():
        files[str(path.relative_to(root))] = {'symlink':str(path.readlink())}
    elif path.is_file():
        files[str(path.relative_to(root))] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
result = {'rootfs_files':files, 'packages':packages,
    'nsjail_sha256':hashlib.sha256(Path('/opt/cpt-numerics/bin/nsjail').read_bytes()).hexdigest()}
print(json.dumps(result, sort_keys=True, separators=(',', ':')))
