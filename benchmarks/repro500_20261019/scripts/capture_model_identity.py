"""Fingerprint the evaluated local model/adapter; does not read any tasks."""
import hashlib
import json
from pathlib import Path
import sys

suite = Path(__file__).resolve().parents[1]
profile = json.loads((suite/'model_profile.json').read_text())
result = {'profile':profile, 'python':sys.version, 'files':{}}
for label, directory in [('base', profile['base_model_path']), ('adapter', profile['adapter_path'])]:
    files = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and (path.suffix in {'.safetensors', '.json', '.jinja', '.txt', '.model'}):
            hasher = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda:source.read(8*1024*1024), b''):
                    hasher.update(chunk)
            files[path.name] = {'bytes':path.stat().st_size, 'sha256':hasher.hexdigest()}
    result['files'][label] = files
print(json.dumps(result, sort_keys=True, separators=(',', ':')))
