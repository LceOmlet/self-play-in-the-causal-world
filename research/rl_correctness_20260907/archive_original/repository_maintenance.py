"""Save the existing project to its user-designated Git remote without rewriting history."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
AUDIT = Path('/home/chen/runs/rl-correctness-goal-20260907/repository')
BRANCH = 'codex/rl-correctness-official-dapo-20260907'
AUDIT.mkdir(parents=True, exist_ok=True)


def git(*args, check=True):
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT='0')
    return subprocess.run(['git', '-C', str(PROJECT), *args], env=env,
                          capture_output=True, check=check)


def paths(*args):
    return [path.decode() for path in git(*args, '-z').stdout.split(b'\0') if path]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    (AUDIT / name).write_text(json.dumps(value, indent=2) + '\n')
    display = {key: len(item) if key == 'files' and isinstance(item, dict) else item
               for key, item in value.items()}
    print(json.dumps(display, indent=2))


def backup():
    bundle = AUDIT / 'before-maintenance.bundle'
    archive = AUDIT / 'before-maintenance-worktree.tar.gz'
    if bundle.exists() or archive.exists():
        raise RuntimeError('Initial backup already exists; refusing to overwrite it.')
    assert git('diff', '--cached', '--quiet', check=False).returncode == 0
    git('bundle', 'create', str(bundle), '--all')
    selected = sorted(set(paths('ls-files') + paths('ls-files', '--others', '--exclude-standard')))
    manifest = {}
    with tarfile.open(archive, 'w:gz') as output:
        for name in selected:
            file = PROJECT / name
            if file.is_file():
                manifest[name] = digest(file.read_bytes())
                output.add(file, arcname=name)
    save('before-maintenance.json', {'head': git('rev-parse', 'HEAD').stdout.decode().strip(),
                                   'branch': git('branch', '--show-current').stdout.decode().strip(),
                                   'files': manifest, 'bundle_sha256': digest(bundle.read_bytes()),
                                   'archive_sha256': digest(archive.read_bytes())})


def integrate():
    assert (AUDIT / 'before-maintenance.json').is_file()
    archive = AUDIT / 'repository-docs-and-research.tar.gz'
    with tarfile.open(archive) as incoming:
        names = incoming.getnames()
        allowed = {'.gitattributes', '.gitignore', 'README.md', 'patches/README.md',
                   'docs/official-dapo-integration-20260906.md', 'docs/rl-correctness-goal-20260907.md'}
        for member in incoming.getmembers():
            assert member.isfile() and not member.issym()
            assert '..' not in Path(member.name).parts and not member.name.startswith('/')
            assert member.name in allowed or member.name.startswith('research/rl_correctness_20260907/')
        incoming.extractall(PROJECT, filter='data')
    save('integration.json', {'files': len(names), 'archive_sha256': digest(archive.read_bytes())})


def commit():
    assert git('diff', '--cached', '--quiet', check=False).returncode == 0
    research = PROJECT / 'research/rl_correctness_20260907'
    (research / 'archive_original/repository_maintenance.py').write_bytes(Path(__file__).read_bytes())
    fingerprints = {path.relative_to(research).as_posix(): digest(path.read_bytes())
                    for path in sorted(research.rglob('*')) if path.is_file()
                    and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pt'}
                    and path.name != 'archive-manifest.json'}
    (research / 'archive-manifest.json').write_text(json.dumps(fingerprints, indent=2) + '\n')
    candidates = sorted(set(paths('diff', '--name-only') + paths('ls-files', '--others', '--exclude-standard')))
    assert candidates
    allowed_roots = {'.gitattributes', '.gitignore', 'README.md', 'configs', 'docs',
                     'patches', 'scripts', 'src', 'tests', 'research'}
    secrets = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{32,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)')
    manifest = {}
    for name in candidates:
        assert Path(name).parts[0] in allowed_roots, name
        file = PROJECT / name
        assert file.is_file() and not file.is_symlink(), name
        data = file.read_bytes()
        assert len(data) < 3_000_000, (name, len(data))
        assert secrets.search(data) is None, 'Potential credential in ' + name
        manifest[name] = {'sha256': digest(data), 'bytes': len(data)}
    assert sum(v['bytes'] for v in manifest.values()) < 15_000_000
    checked = git('diff', '--check', check=False)
    assert checked.returncode == 0, checked.stdout.decode()
    branch = git('branch', '--show-current').stdout.decode().strip()
    if branch != BRANCH:
        assert git('show-ref', '--verify', '--quiet', 'refs/heads/' + BRANCH, check=False).returncode != 0
        git('switch', '-c', BRANCH)
    git('add', '--', *candidates)
    # Prove that Git's stored bytes preserve all patch and research fingerprints.
    for name in candidates:
        if name.endswith('.patch') or name.startswith('research/'):
            assert digest(git('show', ':' + name).stdout) == manifest[name]['sha256'], name
    git('diff', '--cached', '--check')
    result = git('commit', '-m', 'Preserve causal-kernel fixes and audited official DAPO integration',
                 '-m', 'Keep explicit upstream termination and action-mask compatibility patches, source fingerprints, independent proofs and failed-run evidence. Long training remains stopped; the fresh one-update audit is tracked separately.')
    head = git('rev-parse', 'HEAD').stdout.decode().strip()
    save('committed.json', {'branch': BRANCH, 'commit': head, 'files': manifest,
                            'summary': result.stdout.decode(), 'status': git('status', '--short').stdout.decode()})


def push():
    head = git('rev-parse', 'HEAD').stdout.decode().strip()
    assert git('branch', '--show-current').stdout.decode().strip() == BRANCH
    result = git('push', '--set-upstream', 'origin', BRANCH, check=False)
    text = (result.stdout + result.stderr).decode(errors='replace')
    text = re.sub(r'https://[^/@\s]+:[^/@\s]+@', 'https://[redacted]@', text)
    if result.returncode:
        save('push-failed.json', {'exit_code': result.returncode, 'output': text})
        raise SystemExit(result.returncode)
    remote = git('ls-remote', 'origin', 'refs/heads/' + BRANCH).stdout.decode().split()[0]
    assert remote == head
    git('bundle', 'create', str(AUDIT / 'saved-project.bundle'), '--all')
    git('bundle', 'verify', str(AUDIT / 'saved-project.bundle'))
    save('pushed.json', {'branch': BRANCH, 'commit': head, 'remote_commit': remote,
                        'verified': True, 'bundle_sha256': digest((AUDIT / 'saved-project.bundle').read_bytes()),
                        'output': text})


parser = argparse.ArgumentParser()
parser.add_argument('action', choices=('backup', 'integrate', 'commit', 'push'))
globals()[parser.parse_args().action]()
