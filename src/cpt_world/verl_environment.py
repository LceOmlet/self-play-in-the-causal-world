"""CPT-World data/tool adapters for the unmodified official verl DAPO recipe.

The environment owns task semantics and raw reward. Official verl owns token
generation, tool dispatch, masks, overlong shaping, sampling and optimization.
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import numpy as np
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from cpt_world.identification import INTERACTION_SURFACE_VERSION
from cpt_world.trl_environment import CPTWorldEnvironment, task_advantage_utility


def _json_default(value):
    if isinstance(value, Fraction):
        return float(value)
    raise TypeError(f"Unsupported diagnostic type: {type(value).__name__}")


def _audit(event: dict) -> None:
    directory = os.environ.get("CPT_WORLD_VERL_AUDIT_DIR")
    if not directory:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    with (target / f"environment-{os.getpid()}.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, default=_json_default, ensure_ascii=False) + "\n")


def cpt_world_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "act",
                "description": "Execute one CPT-World experiment or terminal answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "object",
                            "description": "The exact command JSON object from the task protocol.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    )


class CPTWorldTool(BaseTool):
    """Bind verl's standard stateful-tool interface to the existing episode."""

    def __init__(self, config, tool_schema=None):
        if "terminate" not in ToolResponse.model_fields:
            raise RuntimeError("The reviewed official tool-termination patch is required")
        super().__init__(config, tool_schema or cpt_world_tool_schema())
        self._requests: dict[str, dict] = {}

    async def create(self, instance_id=None, **kwargs):
        instance_id = instance_id or str(uuid4())
        row = json.loads(kwargs["create_kwargs"]["row_json"])
        if not isinstance(row, dict):
            raise TypeError("CPT-World task row must be an object")
        if row.get("environment_version") != INTERACTION_SURFACE_VERSION:
            raise ValueError("Regenerate data for the current verified environment contract")
        self._requests[instance_id] = row
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        agent_data = kwargs["agent_data"]
        row = self._requests[instance_id]
        try:
            environment = getattr(agent_data, "_cpt_world_environment", None)
            if environment is None:
                environment = CPTWorldEnvironment()
                environment.reset(**row)
                # AgentData is scoped to this trajectory. Keeping the owner here
                # preserves state across BaseTool's per-call create/release cycle.
                agent_data._cpt_world_environment = environment
                agent_data._cpt_world_tape_key = row["tape_key"]
            if agent_data._cpt_world_tape_key != row["tape_key"]:
                raise RuntimeError("A trajectory changed its CPT-World task")
            command = parameters.get("command")
            feedback = environment.act(command)
            episode = environment.episode
            snapshot = {
                "query_type": row["query_type"],
                "sample_index": row["sample_index"],
                "tape_key": row["tape_key"],
                "raw_reward": environment.get_reward(),
                "completed": episode.completed,
                "terminal_score": episode.terminal_score,
                "observations_used": episode.observations_used,
                "queries_used": episode.queries_used,
            }
            # Only serializable owner diagnostics travel to the reward worker.
            snapshot = json.loads(json.dumps(snapshot, default=_json_default))
            agent_data.extra_fields["cpt_world"] = snapshot
            _audit(
                {
                    "event": "act",
                    "trajectory_id": agent_data.request_id,
                    "command": command,
                    "feedback": feedback,
                    **snapshot,
                }
            )
            # The official DAPO reward manager consumes terminal reward once.
            return ToolResponse(text=feedback, terminate=episode.completed), 0.0, {}
        except Exception as error:
            agent_data.extra_fields["cpt_world_adapter_error"] = repr(error)
            raise

    async def release(self, instance_id, **kwargs):
        self._requests.pop(instance_id, None)


class CPTWorldRewardMetadata(DAPORewardManager):
    """Map tool metadata to DAPO's score input; all reward shaping stays upstream."""

    async def run_single(self, data):
        data = data[-1:]
        item = data[0].non_tensor_batch
        extra = dict(item.get("extra_info", {}))
        tool_fields = item.get("tool_extra_fields", {}) or {}
        extra["cpt_world"] = tool_fields.get("cpt_world")
        extra["cpt_world_adapter_error"] = tool_fields.get("cpt_world_adapter_error")
        data.non_tensor_batch["extra_info"] = np.array([extra], dtype=object)
        return await super().run_single(data)


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """Return only the environment-owned quality; never trust text claiming a reward."""
    del solution_str, ground_truth, kwargs
    if extra_info.get("environment_version") != INTERACTION_SURFACE_VERSION:
        raise ValueError("Reward data belongs to an obsolete environment contract")
    if extra_info.get("cpt_world_adapter_error"):
        raise RuntimeError(extra_info["cpt_world_adapter_error"])
    family = extra_info["query_type"]
    if data_source != f"cpt_world/{family}":
        raise ValueError("Task family and data source disagree")
    state = extra_info.get("cpt_world")
    if state is None:  # No valid act call: an unfinished episode has quality zero.
        state = {"raw_reward": 0.0, "completed": False, "terminal_score": None}
    elif state["query_type"] != family or state["tape_key"] != extra_info["tape_key"]:
        raise ValueError("Reward state belongs to a different task")
    raw = task_advantage_utility(float(state["raw_reward"]), family)
    if not state["completed"] and raw != 0:
        raise ValueError("An unfinished episode cannot receive terminal quality")
    if not math.isfinite(raw):
        raise ValueError("Non-finite environment reward")
    result = {"score": raw, "acc": raw, "raw_reward": raw, "completed": float(state["completed"])}
    _audit({"event": "raw_score", "data_source": data_source, **state, **result})
    return result
