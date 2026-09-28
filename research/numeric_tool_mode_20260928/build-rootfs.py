"""Stage a minimal numerical runtime; install it root-owned only after inspection."""
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=pathlib.Path, required=True, help='Trusted build directory containing runtime/')
base = parser.parse_args().base.resolve()
root = base / 'rootfs-staging'
if root.exists():
    raise SystemExit('Refusing to overwrite an existing staged rootfs')
root.mkdir()

def copy(path, destination=None):
    target = root / str(destination or path).lstrip('/')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target, follow_symlinks=True)

shutil.copytree(base / 'runtime', root / 'opt/numerics', symlinks=True)
shutil.copytree('/usr/lib/python3.12', root / 'usr/lib/python3.12', symlinks=True)
copy('/usr/bin/python3.12')
(root / 'usr/bin/python3').symlink_to('python3.12')
for directory in ('tmp', 'proc', 'channels', 'dev', 'etc'):
    (root / directory).mkdir(exist_ok=True)
for name in ('null', 'urandom'):
    (root / 'dev' / name).touch()
# No host account, resolv.conf, project files, models or credentials.
(root / 'etc/passwd').write_text('numerics:x:1000:1000:numerics:/tmp:/nonexistent\n')
(root / 'etc/group').write_text('numerics:x:1000:\n')

# Inspect dependencies of the trusted distro interpreter and pinned wheel ELFs.
# This is staging, never applied to code supplied by a model.
libraries = set()
for binary in [root / 'usr/bin/python3.12', *root.rglob('*.so'), *root.rglob('*.so.*')]:
    if not binary.is_file() or binary.is_symlink():
        continue
    output = subprocess.run(['ldd', str(binary)], capture_output=True, text=True).stdout
    for name in re.findall(r'(/[^\s()]+)', output):
        path = pathlib.Path(name)
        if path.exists() and not path.is_relative_to(root):
            libraries.add(path)
for path in sorted(libraries):
    copy(path)

# Remove launchers with an outside-venv shebang and installation-only packages.
# The numerical libraries themselves remain complete, without function whitelists.
for path in (root / 'opt/numerics/bin').iterdir():
    if path.name not in {'python', 'python3', 'python3.12'}:
        path.unlink()
packages = root / 'opt/numerics/lib/python3.12/site-packages'
for path in packages.glob('pip*'):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
manifest = {'python': sys.version, 'copied_system_libraries': sorted(map(str, libraries)),
            'rootfs': str(root), 'host_paths_mounted': False}
(base / 'rootfs-build.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'rootfs': str(root), 'system_libraries': len(libraries)}))
