"""Resource isolation for training-time counterfactual certification.

The exact counterfactual owner remains :func:`compute_query_truth`. This
module only runs that owner in a disposable spawned process because native
SCIP model construction can exceed its cooperative time limit and otherwise
take the colocated trainer down with it.
"""

from __future__ import annotations

import faulthandler
import json
import multiprocessing
import os
import signal
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .world_space import WorldSpec

COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES = 8 * 1024**3
_COUNTERFACTUAL_WORKER_GRACE_SECONDS = 1.0
_COUNTERFACTUAL_WORKER_POLL_SECONDS = 0.1


class CounterfactualResourceLimitError(RuntimeError):
    """The exact owner exceeded a hard process resource boundary."""


class CounterfactualWorkerError(Exception):
    """The isolated owner failed for a reason other than its declared limits."""


def _set_address_space_limit(memory_limit_bytes: int) -> None:
    if os.name != "posix":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def _maximum_resident_set_kib() -> int | None:
    if os.name != "posix":
        return None
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value // 1024 if sys.platform == "darwin" else value


def _linux_process_memory_kib(pid: int) -> dict[str, int]:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return {}
    result: dict[str, int] = {}
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        name, separator, value = line.partition(":")
        if not separator or name not in {"VmRSS", "VmHWM", "VmSwap"}:
            continue
        fields = value.strip().split()
        if fields and fields[0].isdigit():
            result[name] = int(fields[0])
    return result


def _counterfactual_truth_worker(
    connection: Any,
    world: WorldSpec,
    seed: Mapping[str, Any],
    endpoint_time_limit_seconds: float,
    memory_limit_bytes: int,
    stack_path: str | None,
) -> None:
    stack_file = None
    started = time.perf_counter()
    try:
        _set_address_space_limit(memory_limit_bytes)
        if stack_path is not None:
            stack_file = open(stack_path, "w", encoding="utf-8")
            if hasattr(signal, "SIGUSR1"):
                faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True)
        from .query_truth import compute_query_truth

        truth = compute_query_truth(
            world,
            seed,
            counterfactual_endpoint_time_limit_seconds=endpoint_time_limit_seconds,
        )
        connection.send(
            {
                "status": "ok",
                "truth": dict(truth),
                "elapsed_seconds": time.perf_counter() - started,
                "max_rss_kib": _maximum_resident_set_kib(),
            }
        )
    except MemoryError as error:
        if stack_file is not None:
            faulthandler.dump_traceback(file=stack_file, all_threads=True)
            stack_file.flush()
        connection.send(
            {
                "status": "memory_limit",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
                "max_rss_kib": _maximum_resident_set_kib(),
            }
        )
    except RuntimeError as error:
        connection.send(
            {
                "status": "solver_rejected",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
                "max_rss_kib": _maximum_resident_set_kib(),
            }
        )
    except Exception as error:
        if stack_file is not None:
            faulthandler.dump_traceback(file=stack_file, all_threads=True)
            stack_file.flush()
        connection.send(
            {
                "status": "worker_error",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
                "max_rss_kib": _maximum_resident_set_kib(),
            }
        )
    finally:
        if stack_file is not None:
            stack_file.close()
        connection.close()


def _stop_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join()
        return
    if os.name == "posix" and hasattr(signal, "SIGUSR1"):
        try:
            os.kill(process.pid, signal.SIGUSR1)
            time.sleep(0.1)
        except ProcessLookupError:
            pass
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join()


def _write_diagnostic(
    diagnostic_dir: Path | None,
    candidate_id: str,
    payload: Mapping[str, Any],
) -> Path | None:
    if diagnostic_dir is None:
        return None
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostic_dir / f"{candidate_id}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _remove_empty_stack(stack_path: str | None) -> None:
    if stack_path is None:
        return
    stack = Path(stack_path)
    if stack.exists() and stack.stat().st_size == 0:
        stack.unlink()


def compute_counterfactual_truth_isolated(
    world: WorldSpec,
    seed: Mapping[str, Any],
    *,
    endpoint_time_limit_seconds: float,
    diagnostic_dir: Path | None = None,
) -> Mapping[str, Any]:
    """Run the exact truth owner in a bounded, disposable spawned process.

    The hard wall is the two endpoint allowances plus one second for process
    communication. Exceeding it or the 8-GiB address-space ceiling rejects
    only this candidate; no approximate truth is substituted.
    """

    if endpoint_time_limit_seconds <= 0:
        raise ValueError("counterfactual endpoint time limit must be positive")
    candidate_id = str(seed.get("seed_id", "counterfactual-candidate"))
    stack_path = (
        str(diagnostic_dir / f"{candidate_id}.stack.txt")
        if diagnostic_dir is not None
        else None
    )
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_counterfactual_truth_worker,
        args=(
            send_connection,
            world,
            dict(seed),
            endpoint_time_limit_seconds,
            COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES,
            stack_path,
        ),
        name=f"cpt-world-cf-{candidate_id}",
    )
    started = time.perf_counter()
    process.start()
    send_connection.close()
    hard_wall_seconds = 2.0 * endpoint_time_limit_seconds + _COUNTERFACTUAL_WORKER_GRACE_SECONDS
    deadline = started + hard_wall_seconds
    observed_memory: dict[str, int] = {}
    message: Mapping[str, Any] | None = None
    while time.perf_counter() < deadline:
        observed_memory.update(_linux_process_memory_kib(process.pid))
        if receive_connection.poll(_COUNTERFACTUAL_WORKER_POLL_SECONDS):
            try:
                message = receive_connection.recv()
            except EOFError:
                message = None
            break
        if not process.is_alive():
            break

    elapsed_seconds = time.perf_counter() - started
    if message is None and receive_connection.poll():
        try:
            message = receive_connection.recv()
        except EOFError:
            message = None
    if message is None and process.is_alive():
        observed_memory.update(_linux_process_memory_kib(process.pid))
        _stop_process(process)
        payload = {
            "status": "hard_timeout",
            "candidate_id": candidate_id,
            "elapsed_seconds": elapsed_seconds,
            "hard_wall_seconds": hard_wall_seconds,
            "memory_limit_bytes": COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES,
            "process_memory_kib": observed_memory,
            "exit_code": process.exitcode,
            "stack_path": stack_path,
        }
        path = _write_diagnostic(diagnostic_dir, candidate_id, payload)
        raise CounterfactualResourceLimitError(
            f"counterfactual candidate {candidate_id} exceeded the {hard_wall_seconds:g}s "
            f"hard wall; diagnostic={path}"
        )

    process.join(timeout=1.0)
    if process.is_alive():
        _stop_process(process)
    receive_connection.close()
    if message is None:
        payload = {
            "status": "worker_exit",
            "candidate_id": candidate_id,
            "elapsed_seconds": elapsed_seconds,
            "memory_limit_bytes": COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES,
            "process_memory_kib": observed_memory,
            "exit_code": process.exitcode,
            "stack_path": stack_path,
        }
        path = _write_diagnostic(diagnostic_dir, candidate_id, payload)
        raise CounterfactualResourceLimitError(
            f"counterfactual candidate {candidate_id} exited without a result; "
            f"diagnostic={path}"
        )

    status = str(message.get("status"))
    if status == "ok":
        _remove_empty_stack(stack_path)
        truth = message.get("truth")
        if not isinstance(truth, Mapping):
            raise RuntimeError("counterfactual worker returned a non-mapping truth")
        return dict(truth)
    if status == "solver_rejected":
        _remove_empty_stack(stack_path)
        raise RuntimeError(str(message.get("error", "counterfactual solver rejected candidate")))
    if status == "worker_error":
        path = _write_diagnostic(
            diagnostic_dir,
            candidate_id,
            {
                **dict(message),
                "candidate_id": candidate_id,
                "hard_wall_seconds": hard_wall_seconds,
                "memory_limit_bytes": COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES,
                "process_memory_kib": observed_memory,
                "exit_code": process.exitcode,
                "stack_path": stack_path,
            },
        )
        raise CounterfactualWorkerError(
            f"counterfactual candidate {candidate_id} failed unexpectedly; diagnostic={path}"
        )

    payload = {
        **dict(message),
        "candidate_id": candidate_id,
        "hard_wall_seconds": hard_wall_seconds,
        "memory_limit_bytes": COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES,
        "process_memory_kib": observed_memory,
        "exit_code": process.exitcode,
        "stack_path": stack_path,
    }
    path = _write_diagnostic(diagnostic_dir, candidate_id, payload)
    raise CounterfactualResourceLimitError(
        f"counterfactual candidate {candidate_id} failed in isolated worker "
        f"({status}); diagnostic={path}"
    )
