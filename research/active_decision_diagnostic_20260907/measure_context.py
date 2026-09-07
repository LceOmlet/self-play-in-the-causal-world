"""Count a command-only rendering of the diagnostic with the actual model tokenizer."""

import argparse
import gzip
import json
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained("/home/chen/models/Qwen/Qwen3.5-9B")


def token_count(encoded):
    # Installed Transformers returns a BatchEncoding here, not a token list.
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if ids and isinstance(ids[0], list):
        assert len(ids) == 1
        ids = ids[0]
    assert all(isinstance(token, int) for token in ids)
    assert len(ids) > 100
    return len(ids)


schema = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Execute one CPT-World experiment or terminal answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "object",
                    "description": "The exact command JSON object required by the task protocol.",
                }
            },
            "required": ["command"],
        },
    },
}
results = []
for path in sorted((args.root / "results").glob("*.public.json.gz")):
    public = json.loads(gzip.decompress(path.read_bytes()))
    messages = [
        {
            "role": "system",
            "content": "You are evaluated on active causal experimentation. "
            "Return exactly one legal JSON command and no prose.",
        },
        {"role": "user", "content": public["prompt"]},
    ]
    before = token_count(
        tokenizer.apply_chat_template(
            messages,
            tools=[schema],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    for event in public["transcript"]:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "act", "arguments": {"command": event["command"]}},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": event["feedback"]})
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "act",
                        "arguments": {"command": public["policy"]["answer"]},
                    },
                }
            ],
        }
    )
    total = token_count(
        tokenizer.apply_chat_template(
            messages,
            tools=[schema],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    results.append(
        {
            "public_file": path.name,
            "initial_tokens": before,
            "total_tokens": total,
            "tool_calls": len(public["transcript"]) + 1,
            "exceeds_32768": total > 32768,
        }
    )
report = {
    "scope": "Command-only reconstruction using the original task system/user text, act schema "
    "and model tokenizer. It omits model reasoning and is not an actual verl rollout, a lower "
    "bound for every possible strategy, or proof these tasks cannot fit the model context.",
    "episodes": results,
}
(args.root / "context-measurement.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "episodes": len(results),
            "exceeds_32768": sum(r["exceeds_32768"] for r in results),
            "maximum_tokens": max(r["total_tokens"] for r in results),
        }
    )
)
