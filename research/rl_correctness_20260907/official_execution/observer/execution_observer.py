"""Observe actual upstream calls using the Python profiler API.

No function replacement, loss implementation, optimizer, RNG draw, tensor hook,
or training decision. Copies are detached; only diagnostic files are written.
Profiling and CPU copies add overhead; this run is not a speed benchmark.
"""

import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ["CPT_DAPO_OBSERVATION_DIR"])
VERL = os.environ["VERL_ROOT"].rstrip("/") + "/verl/"
TARGETS = {
    "_update_actor": "trainer/ppo/ray_trainer.py",
    "extract_reward": "trainer/ppo/reward.py",
    "compute_grpo_outcome_advantage": "trainer/ppo/core_algos.py",
    "compute_policy_loss_vanilla": "trainer/ppo/core_algos.py",
    "compute_rollout_correction_and_add_to_batch": "trainer/ppo/rollout_corr_helper.py",
    "optimizer_step": "workers/engine/fsdp/transformer_impl.py",
    "run": "experimental/agent_loop/tool_agent_loop.py",
    "_handle_generating_state": "experimental/agent_loop/tool_agent_loop.py",
    "_handle_processing_tools_state": "experimental/agent_loop/tool_agent_loop.py",
}
COUNTERS = {}
OWNERS = {}
SOURCES = set()
LOCK = threading.RLock()


def plain(value):
    """Copy diagnostics without retaining autograd graphs or model owners."""
    import torch

    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if hasattr(detached, "to_local"):
            detached = detached.to_local()
        return detached.cpu().clone()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if dataclasses.is_dataclass(value):
        return {
            field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        return plain(OmegaConf.to_container(value, resolve=True))
    return {"unserialized_type": type(value).__qualname__}


def bundle_batch(batch):
    return {
        "tensors": {key: plain(value) for key, value in batch.batch.items()},
        "non_tensors": plain(batch.non_tensor_batch),
        "meta": plain(batch.meta_info),
    }


def record(kind, content, frame=None):
    import torch

    with LOCK:
        directory = ROOT / str(os.getpid())
        directory.mkdir(parents=True, exist_ok=True)
        serial = COUNTERS.get(kind, 0) + 1
        COUNTERS[kind] = serial
        name = f"{kind}-{serial:04d}.pt"
        torch.save(content, directory / name)
        event = {"event": kind, "pid": os.getpid(), "time": time.time(), "file": name}
        if frame is not None:
            path = frame.f_code.co_filename
            event.update(
                source=path, function=frame.f_code.co_name, line=frame.f_code.co_firstlineno
            )
            if path not in SOURCES:
                event["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                SOURCES.add(path)
        with (directory / "events.jsonl").open("a") as output:
            output.write(json.dumps(event) + "\n")


def parameters(module, *, include_values, include_gradients):
    trainable, frozen = {}, []
    for name, parameter in module.named_parameters():
        item = {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": parameter.requires_grad,
            "has_grad": parameter.grad is not None,
        }
        if parameter.requires_grad:
            if include_values:
                item["value"] = plain(parameter)
            if include_gradients:
                item["grad"] = plain(parameter.grad)
            trainable[name] = item
        else:
            frozen.append({"name": name, **item})
    return {"trainable": trainable, "frozen": frozen}


def optimizer_snapshot(optimizer, module):
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: [names.get(id(p), "UNKNOWN") for p in value]
                if key == "params"
                else plain(value)
                for key, value in group.items()
            }
        )
    return {
        "optimizer_type": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
        "param_groups": groups,
        "state": {
            names.get(id(p), "UNKNOWN"): plain(state) for p, state in optimizer.state.items()
        },
        **parameters(module, include_values=True, include_gradients=True),
    }


def observe(frame, event, result):
    if event != "call" and event != "return":
        return
    name = frame.f_code.co_name
    if name not in TARGETS and name != "step":
        return
    path = frame.f_code.co_filename
    local = frame.f_locals
    if name == "step":
        if not path.endswith(("/torch/optim/adam.py", "/torch/optim/adamw.py")):
            return
        optimizer = local.get("self")
        module = OWNERS.get(id(optimizer))
        if module is not None:
            record("adam_" + event, optimizer_snapshot(optimizer, module), frame)
        return
    if path != VERL + TARGETS[name]:
        return
    if name == "_update_actor" and event == "call":
        record(
            "actor_batch",
            {"batch": bundle_batch(local["batch"]), "config": plain(local["self"].config)},
            frame,
        )
    elif name == "extract_reward" and event == "return" and result is not None:
        record(
            "pre_filter_batch",
            {"batch": bundle_batch(local["batch"]), "result": plain(result)},
            frame,
        )
    elif name == "compute_grpo_outcome_advantage" and event == "return" and result is not None:
        record(
            "grpo",
            {
                key: plain(local.get(key))
                for key in (
                    "token_level_rewards",
                    "response_mask",
                    "index",
                    "epsilon",
                    "norm_adv_by_std_in_grpo",
                )
            }
            | {"result": plain(result)},
            frame,
        )
    elif (
        name == "compute_rollout_correction_and_add_to_batch"
        and event == "return"
        and result is not None
    ):
        record(
            "tis",
            {
                "batch": bundle_batch(result[0]),
                "metrics": plain(result[1]),
                "config": plain(local.get("rollout_corr_config")),
            },
            frame,
        )
    elif name == "compute_policy_loss_vanilla" and event == "return" and result is not None:
        record(
            "policy_loss",
            {
                key: plain(local.get(key))
                for key in (
                    "old_log_prob",
                    "log_prob",
                    "advantages",
                    "response_mask",
                    "loss_agg_mode",
                    "config",
                    "rollout_is_weights",
                )
            }
            | {"result": plain(result)},
            frame,
        )
    elif name == "optimizer_step":
        owner = local["self"]
        if event == "call":
            OWNERS[id(owner.optimizer)] = owner.module
            record(
                "pre_clip",
                parameters(owner.module, include_values=False, include_gradients=True)
                | {
                    "optimizer_config": plain(owner.optimizer_config),
                    "has_scaler": getattr(owner, "scaler", None) is not None,
                },
                frame,
            )
        else:
            record("optimizer_done", {"grad_norm": plain(result)}, frame)
            OWNERS.pop(id(owner.optimizer), None)
    elif event == "return" and name == "run" and type(result).__name__ == "AgentLoopOutput":
        data = local["agent_data"]
        record(
            "trajectory",
            {
                "request_id": data.request_id,
                "output": plain(result),
                "sampling_params": plain(local["sampling_params"]),
                "tools_kwargs": plain(data.tools_kwargs),
            },
            frame,
        )
    elif (
        event == "return" and name.startswith("_handle_") and type(result).__name__ == "AgentState"
    ):
        data = local["agent_data"]
        snapshot = {
            "request_id": data.request_id,
            "state": str(result),
            "prompt_ids": plain(data.prompt_ids),
            "response_mask": plain(data.response_mask),
            "response_logprobs": plain(data.response_logprobs),
            "extra_fields": plain(data.extra_fields),
            "tool_rewards": plain(data.tool_rewards),
            "assistant_turns": data.assistant_turns,
        }
        if name == "_handle_generating_state":
            snapshot.update(
                output=plain(local.get("output")),
                sampling_params=plain(local.get("sampling_params")),
            )
            record("generation", snapshot, frame)
        else:
            snapshot["responses"] = plain(local.get("responses"))
            record("tool_transition", snapshot, frame)


def profile(frame, event, result):
    try:
        observe(frame, event, result)
    except Exception:
        # Diagnostics must never select rewards or suppress upstream exceptions.
        # Any observer error fails audit acceptance even if training succeeds.
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / f"observer-errors-{os.getpid()}.log").open("a") as output:
            output.write(traceback.format_exc())


def install():
    ROOT.mkdir(parents=True, exist_ok=True)
    existing = sys.getprofile()
    if existing is not None:
        raise RuntimeError("Refuse to replace another profiler")
    marker = {
        "pid": os.getpid(),
        "time": time.time(),
        "python": sys.executable,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (ROOT / f"installed-{os.getpid()}.json").write_text(json.dumps(marker, indent=2))
    sys.setprofile(profile)
    threading.setprofile(profile)
