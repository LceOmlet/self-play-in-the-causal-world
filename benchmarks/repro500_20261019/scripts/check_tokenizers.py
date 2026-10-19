"""Compare tokenizer semantics and public prompt encoding, without score access.

Prints metadata only; never opens evaluation grade files or emits task contents.
"""
import json
import gzip
from pathlib import Path
import sys
from transformers import AutoTokenizer

suite = Path(__file__).resolve().parents[1]
profile = json.loads((suite/'model_profile.json').read_text())
base_path,adapter_path = map(Path,(profile['base_model_path'],profile['adapter_path']))
details = {}
for filename in ('tokenizer.json','tokenizer_config.json'):
    base = json.loads((base_path/filename).read_text())
    adapter = json.loads((adapter_path/filename).read_text())
    details[filename] = {'semantic_json_equal':base==adapter,
        'different_top_level_keys':[k for k in set(base)|set(adapter) if base.get(k)!=adapter.get(k)]}
    if filename=='tokenizer.json':
        details[filename]['backend_components_equal'] = {k:base.get(k)==adapter.get(k)
            for k in ('model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens')}
base = AutoTokenizer.from_pretrained(base_path,local_files_only=True)
adapter = AutoTokenizer.from_pretrained(adapter_path,local_files_only=True)
details['runtime'] = {'vocab_equal':base.get_vocab()==adapter.get_vocab(),
    'special_tokens_equal':base.special_tokens_map==adapter.special_tokens_map,
    'chat_template_equal':base.chat_template==adapter.chat_template,
    'base_length':len(base),'adapter_length':len(adapter)}
samples = ['Synthetic test only. Hello 世界 ∣ →',
    '<tool_call>{"name":"act","arguments":{"command":{"type":"answer","value":"state_1"}}}</tool_call>',
    'import numpy as np\nprint(np.mean([1.0, 2.0]))\n']
details['runtime']['synthetic_encoding_equal'] = all(base.encode(text)==adapter.encode(text) for text in samples)
messages = [{'role':'user','content':samples[0]}]
details['runtime']['synthetic_chat_equal'] = base.apply_chat_template(messages,tokenize=True,
    add_generation_prompt=True,enable_thinking=False)==adapter.apply_chat_template(messages,tokenize=True,
    add_generation_prompt=True,enable_thinking=False)
normalized = [json.loads(t.backend_tokenizer.to_str()) for t in (base,adapter)]
details['runtime']['normalized_backend_differences'] = [k for k in set(normalized[0])|set(normalized[1])
    if normalized[0].get(k)!=normalized[1].get(k)]
sys.path.insert(0,str(suite/'frozen/protocol'))
sys.path.insert(0,str(suite))
import tool_mode
from capsule import FrozenEnvironment
from transformers.utils import get_json_schema
env = tool_mode.make_numeric_environment(FrozenEnvironment,lambda:None)()
schemas = [get_json_schema(env.act),get_json_schema(env.numerics)]
manifest = json.loads((suite/'data/manifest.json').read_text())
ids = set(json.loads((suite/'data/eval100.json').read_text())['task_ids'])
mismatches = []
checked = 0
for entry in manifest['tasks']:
    if entry['task_id'] not in ids:
        continue
    with gzip.open(suite/'data'/entry['file'],'rt',encoding='utf-8') as source:
        public_messages = json.load(source)['row']['prompt']
    encodings = [t.apply_chat_template(public_messages,tools=schemas,tokenize=True,
        add_generation_prompt=True,enable_thinking=False) for t in (base,adapter)]
    checked += 1
    if encodings[0]!=encodings[1]:
        mismatches.append(entry['task_id'])
details['public_eval_prompt_audit'] = {'checked':checked,'tools_included':True,
    'all_initial_token_ids_identical':not mismatches,'mismatched_task_ids':mismatches,
    'printed_task_contents':False,'read_grades':False}
print(json.dumps(details,sort_keys=True),flush=True)
