"""The visible task renderer and strict JSON command decoder."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, cast

from .episode import (
    Budget,
    EffectVector,
    HardIntervention,
    InterventionCommand,
    TerminalAnswer,
    Variable,
)
from .world import ASSIGNMENTS, SampleBatch, assignment_value

RENDERER_VERSION = "hidden-cpt-bidirectional-effects-v1"
DEFAULT_LABEL_SEED = 20260818
SYSTEM_MESSAGE = (
    "You are evaluated on active causal experimentation. "
    "Return exactly one legal JSON command and no prose."
)
_LABEL_DOMAIN = b"cpt-world-opaque-labels-v1\0"
_LABEL_POOL = "DEFGHIJKLMNOPQRSTUVW"


@dataclass(frozen=True, slots=True)
class VisibleLayout:
    """A truth-free mapping from canonical roles to opaque surface symbols."""

    layout_id: str
    first_label: str
    second_label: str
    isolated_label: str
    target_order: tuple[Variable, Variable, Variable]
    reverse_effect_order: bool

    def __post_init__(self) -> None:
        labels = self.labels
        if not self.layout_id:
            raise ValueError("layout_id must not be empty")
        if len(set(labels.values())) != 3:
            raise ValueError("visible labels must be distinct")
        if any(len(label) != 3 or set(label) - set(_LABEL_POOL) for label in labels.values()):
            raise ValueError("visible labels must be opaque three-letter tokens")
        if len(set("".join(labels.values()))) != 9:
            raise ValueError("visible labels must not reuse letters")
        if any(not isinstance(role, Variable) for role in self.target_order):
            raise TypeError("target_order must contain Variable values")
        if set(self.target_order) != set(Variable):
            raise ValueError("target_order must contain every canonical role exactly once")
        if not isinstance(self.reverse_effect_order, bool):
            raise TypeError("reverse_effect_order must be a bool")

    @property
    def labels(self) -> dict[Variable, str]:
        return {
            Variable.FIRST: self.first_label,
            Variable.SECOND: self.second_label,
            Variable.ISOLATED: self.isolated_label,
        }

    @property
    def visible_to_internal(self) -> dict[str, Variable]:
        return {label: role for role, label in self.labels.items()}


@dataclass(frozen=True, slots=True)
class VisibleTask:
    """Everything the renderer may see; notably absent are CPTs and truth."""

    layout: VisibleLayout
    budget: Budget = field(default_factory=Budget)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, VisibleLayout):
            raise TypeError("layout must be a VisibleLayout")
        if not isinstance(self.budget, Budget):
            raise TypeError("budget must be a Budget")


def opaque_labels(label_seed: int = DEFAULT_LABEL_SEED) -> tuple[str, str, str]:
    """Create stable, non-ordinal three-letter identifiers from a public seed."""

    if isinstance(label_seed, bool) or not isinstance(label_seed, int) or label_seed < 0:
        raise ValueError("label_seed must be a nonnegative integer")
    seed_bytes = str(label_seed).encode("ascii")
    ranked = sorted(
        _LABEL_POOL,
        key=lambda letter: hashlib.sha256(
            _LABEL_DOMAIN + seed_bytes + b"\0" + letter.encode("ascii")
        ).digest(),
    )
    selected = ranked[:9]
    labels = tuple("".join(selected[offset : offset + 3]) for offset in (0, 3, 6))
    return cast(tuple[str, str, str], labels)


def factorial_layouts(label_seed: int = DEFAULT_LABEL_SEED) -> tuple[VisibleLayout, ...]:
    """Fully cross role names, target order, and terminal effect-line order."""

    symbols = opaque_labels(label_seed)
    role_assignments = tuple(itertools.permutations(symbols))
    target_orders = tuple(itertools.permutations(tuple(Variable)))
    layouts: list[VisibleLayout] = []
    for role_index, role_labels in enumerate(role_assignments):
        for order_index, target_order in enumerate(target_orders):
            for reverse_effect_order in (False, True):
                layouts.append(
                    VisibleLayout(
                        layout_id=(
                            f"labels-{label_seed}-r{role_index}-t{order_index}-"
                            f"e{int(reverse_effect_order)}"
                        ),
                        first_label=role_labels[0],
                        second_label=role_labels[1],
                        isolated_label=role_labels[2],
                        target_order=target_order,
                        reverse_effect_order=reverse_effect_order,
                    )
                )
    return tuple(layouts)


def effect_field(source: str, target: str) -> str:
    return f"effect_{source.lower()}_to_{target.lower()}"


def _effect_fields(layout: VisibleLayout) -> tuple[str, str]:
    return (
        effect_field(layout.first_label, layout.second_label),
        effect_field(layout.second_label, layout.first_label),
    )


def render_task_prompt(task: VisibleTask) -> str:
    layout = task.layout
    labels = layout.labels
    first = labels[Variable.FIRST]
    second = labels[Variable.SECOND]
    isolated = labels[Variable.ISOLATED]
    forward_field, reverse_field = _effect_fields(layout)
    effect_lines = [
        f"{forward_field} = P({second}=1 | do({first}=1)) - P({second}=1 | do({first}=0))",
        f"{reverse_field} = P({first}=1 | do({second}=1)) - P({first}=1 | do({second}=0))",
    ]
    if layout.reverse_effect_order:
        effect_lines.reverse()
    visible_targets = [labels[role] for role in layout.target_order]
    visible_target_json = json.dumps(visible_targets, separators=(",", ":"))

    return f"""CPT-WORLD HIDDEN-MECHANISM CAUSAL EFFECT TASK

{", ".join(visible_targets)} are binary variables. The hidden acyclic world has
exactly one positive direct causal edge between the two focal variables {first}
and {second}: either {first} causes {second}, or {second} causes {first}. There
is no hidden confounding between them. {isolated} is isolated. The graph, CPT
probabilities, causal direction, and effect magnitudes are hidden.

Estimate both interventional effects:
{effect_lines[0]}
{effect_lines[1]}

The two terminal estimates receive equal weight. Allocate one shared
experimental budget; no variable name or display position is preferred.

You may perform at most {task.budget.max_rounds} batch interventions and use at
most {task.budget.max_samples} atomic samples in total. Legal targets, in a
presentation-only order, are {visible_target_json}. Legal values are 0 and 1. Legal
batch sizes are {list(task.budget.batch_sizes)}. Each batch returns IID full
joint counts from the same fixed hidden world. Intervening on {isolated} is
legal but cannot reveal the direction between the focal variables.

Return exactly one JSON object and no prose. An intervention has exactly the
keys `type`, `target`, `value`, and `batch_size`, with `type="intervene"`.

To finish, return `type="answer"` and numeric fields `{forward_field}` and
`{reverse_field}`. Both values must be JSON numbers in [-1, 1]. They are causal
effect point estimates, not confidence values. Both fields are mandatory.
"""


def render_initial_messages(task: VisibleTask) -> tuple[dict[str, str], dict[str, str]]:
    """Build the complete truth-free initial model input."""

    return (
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": render_task_prompt(task)},
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_command(
    raw: str,
    task: VisibleTask,
    *,
    remaining_rounds: int,
    remaining_samples: int,
) -> InterventionCommand | TerminalAnswer:
    """Decode one strict command and map visible labels to canonical roles."""

    if (
        isinstance(remaining_rounds, bool)
        or not isinstance(remaining_rounds, int)
        or remaining_rounds < 0
    ):
        raise ValueError("remaining_rounds must be a nonnegative integer")
    if (
        isinstance(remaining_samples, bool)
        or not isinstance(remaining_samples, int)
        or remaining_samples < 0
    ):
        raise ValueError("remaining_samples must be a nonnegative integer")

    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("response is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    command_type = value.get("type")
    if command_type not in {"intervene", "answer"}:
        raise ValueError("type must be intervene or answer")

    forward_field, reverse_field = _effect_fields(task.layout)
    if command_type == "answer":
        if set(value) != {"type", forward_field, reverse_field}:
            raise ValueError("terminal answer must contain exactly both effect fields")
        return TerminalAnswer(
            EffectVector(
                first_to_second=value[forward_field],
                second_to_first=value[reverse_field],
            )
        )

    if remaining_rounds <= 0 or remaining_samples < min(task.budget.batch_sizes):
        raise ValueError("no intervention is legal; a terminal answer is required")
    if set(value) != {"type", "target", "value", "batch_size"}:
        raise ValueError("intervention must contain exactly four protocol fields")
    target = value["target"]
    if not isinstance(target, str) or target not in task.layout.visible_to_internal:
        raise ValueError("unknown intervention target")
    intervention_value = value["value"]
    if isinstance(intervention_value, bool) or intervention_value not in (0, 1):
        raise ValueError("intervention value must be 0 or 1")
    batch_size = value["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer")
    if batch_size not in task.budget.batch_sizes or batch_size > remaining_samples:
        raise ValueError("batch_size is not legal with the remaining budget")
    return InterventionCommand(
        intervention=HardIntervention(
            target=task.layout.visible_to_internal[target],
            value=intervention_value,
        ),
        batch_size=batch_size,
    )


def render_batch(batch: SampleBatch, layout: VisibleLayout) -> dict[str, Any]:
    """Map canonical joint counts to the exact visible target order."""

    labels = layout.labels
    joint_counts: dict[str, int] = {}
    for assignment, count in zip(ASSIGNMENTS, batch.counts, strict=True):
        key = ",".join(
            f"{labels[role]}={assignment_value(assignment, role)}" for role in layout.target_order
        )
        joint_counts[key] = count
    return {"n": batch.sample_count, "joint_counts": joint_counts}
