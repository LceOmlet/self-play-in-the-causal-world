"""Reproducible old-versus-new benchmark for the WorldSpec probability owner.

The reference functions intentionally reproduce the removed full-joint paths.
They are benchmark-only oracles; production code must not import this module.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from fractions import Fraction

from cpt_world import (
    OutcomeTape,
    WorldIntervention,
    WorldInterventionCommand,
    WorldSpec,
    sample_worldspec_batch,
    worldspec_interventional_distribution,
    worldspec_projected_interventional_distribution,
)


def _binary_chain(node_count: int) -> WorldSpec:
    variables = tuple(f"V{node}" for node in range(node_count))
    return WorldSpec(
        family="benchmark_sparse_chain",
        topology="chain",
        variables=variables,
        domains=(2,) * node_count,
        state_names=(("state_0", "state_1"),) * node_count,
        edges=tuple((node - 1, node) for node in range(1, node_count)),
        parents={0: (), **{node: (node - 1,) for node in range(1, node_count)}},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            **{
                node: (
                    (Fraction(3, 4), Fraction(1, 4)),
                    (Fraction(1, 4), Fraction(3, 4)),
                )
                for node in range(1, node_count)
            },
        },
    )


def _reference_projected(
    world: WorldSpec,
    interventions: dict[int, int],
    measure: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    assignments = tuple((state,) for state in range(world.domains[measure[0]]))
    totals = dict.fromkeys(assignments, Fraction(0))
    for values, probability in worldspec_interventional_distribution(world, interventions):
        selected = tuple(values[node] for node in measure)
        totals[selected] += probability
    return tuple(totals.items())


def _reference_batch(
    world: WorldSpec,
    tape: OutcomeTape,
    command: WorldInterventionCommand,
) -> dict[tuple[int, ...], int]:
    target = command.intervention.target
    state = command.intervention.value
    law = worldspec_interventional_distribution(world, {target: state})
    cumulative: list[tuple[tuple[int, ...], Fraction]] = []
    total = Fraction(0)
    for assignment, probability in law:
        total += probability
        cumulative.append((assignment, total))
    counts: dict[tuple[int, ...], int] = {}
    for sample_index in range(command.batch_size):
        draw = tape.worldspec_uniform(target, state, sample_index)
        full = cumulative[-1][0]
        for assignment, threshold in cumulative:
            if draw < threshold:
                full = assignment
                break
        selected = tuple(full[node] for node in command.measure)
        counts[selected] = counts.get(selected, 0) + 1
    return counts


def _seconds(operation: Callable[[], object], *, repeats: int, loops: int) -> float:
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(loops):
            operation()
        durations.append((time.perf_counter() - started) / loops)
    return statistics.median(durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.nodes < 2:
        raise ValueError("--nodes must be at least 2")
    if args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("--batch-size and --repeats must be positive")

    world = _binary_chain(args.nodes)
    interventions = {0: 1}
    measure = (args.nodes - 1,)
    reference_law = _reference_projected(world, interventions, measure)
    accelerated_law = worldspec_projected_interventional_distribution(
        world,
        interventions,
        measure,
    )
    if accelerated_law != reference_law:
        raise RuntimeError("accelerated exact law disagrees with full-joint reference")

    tape = OutcomeTape("worldspec-acceleration-benchmark")
    command = WorldInterventionCommand(
        intervention=WorldIntervention(0, 1),
        measure=measure,
        batch_size=args.batch_size,
    )
    reference_exact_seconds = _seconds(
        lambda: _reference_projected(world, interventions, measure),
        repeats=args.repeats,
        loops=1,
    )
    accelerated_exact_seconds = _seconds(
        lambda: worldspec_projected_interventional_distribution(
            world,
            interventions,
            measure,
        ),
        repeats=args.repeats,
        loops=10,
    )
    reference_batch_seconds = _seconds(
        lambda: _reference_batch(world, tape, command),
        repeats=args.repeats,
        loops=1,
    )
    accelerated_batch_seconds = _seconds(
        lambda: sample_worldspec_batch(world, tape, command, start_index=0),
        repeats=args.repeats,
        loops=5,
    )
    result = {
        "nodes": args.nodes,
        "joint_states": 2**args.nodes,
        "batch_size": args.batch_size,
        "exact_marginal": {
            "reference_seconds": reference_exact_seconds,
            "accelerated_seconds": accelerated_exact_seconds,
            "speedup": reference_exact_seconds / accelerated_exact_seconds,
            "fraction_exact_match": True,
        },
        "batch_sampling": {
            "reference_seconds": reference_batch_seconds,
            "accelerated_seconds": accelerated_batch_seconds,
            "speedup": reference_batch_seconds / accelerated_batch_seconds,
            "law_version": "ancestral-node-tape-v2",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
