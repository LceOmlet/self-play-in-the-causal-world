"""Resource isolation for training-time counterfactual certification.

The exact counterfactual owner remains :func:`compute_query_truth`. This
module only runs that owner in a disposable Python process because native SCIP
model construction can exceed its cooperative time limit and otherwise take
the colocated trainer down with it.
"""

from __future__ import annotations

import faulthandler
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
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


def _worker_result(
    world: WorldSpec,
    seed: Mapping[str, Any],
    endpoint_time_limit_seconds: float,
    memory_limit_bytes: int,
    stack_path: str | None,
) -> dict[str, Any]:
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
        return {
            "status": "ok",
            "truth": dict(truth),
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kib": _maximum_resident_set_kib(),
        }
    except MemoryError as error:
        if stack_file is not None:
            faulthandler.dump_traceback(file=stack_file, all_threads=True)
            stack_file.flush()
        return {
            "status": "memory_limit",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kib": _maximum_resident_set_kib(),
        }
    except RuntimeError as error:
        return {
            "status": "solver_rejected",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kib": _maximum_resident_set_kib(),
        }
    except Exception as error:
        if stack_file is not None:
            faulthandler.dump_traceback(file=stack_file, all_threads=True)
            stack_file.flush()
        return {
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kib": _maximum_resident_set_kib(),
        }
    finally:
        if stack_file is not None:
            stack_file.close()


def _worker_main(arguments: list[str]) -> int:
    if len(arguments) != 4:
        return 2
    request_path = Path(arguments[0])
    result_path = Path(arguments[1])
    memory_limit_bytes = int(arguments[2])
    stack_path = arguments[3] or None
    with request_path.open("rb") as stream:
        world, seed, endpoint_time_limit_seconds = pickle.load(stream)
    result = _worker_result(
        world,
        seed,
        endpoint_time_limit_seconds,
        memory_limit_bytes,
        stack_path,
    )
    temporary = result_path.with_name(f"{result_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(result_path)
    return 0


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix" and hasattr(signal, "SIGUSR1"):
        try:
            os.kill(process.pid, signal.SIGUSR1)
            time.sleep(0.1)
        except ProcessLookupError:
            pass
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


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


def _temporary_pickle_path(prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".pickle")
    os.close(descriptor)
    return Path(name)


def compute_counterfactual_truth_isolated(
    world: WorldSpec,
    seed: Mapping[str, Any],
    *,
    endpoint_time_limit_seconds: float,
    diagnostic_dir: Path | None = None,
) -> Mapping[str, Any]:
    """Run the exact truth owner in a bounded, disposable interpreter.

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

    request_path = _temporary_pickle_path("cpt-world-cf-request-")
    result_path = _temporary_pickle_path("cpt-world-cf-result-")
    result_path.unlink()
    try:
        with request_path.open("wb") as stream:
            pickle.dump(
                (world, dict(seed), endpoint_time_limit_seconds),
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        command = (
            sys.executable,
            "-c",
            (
                "import sys; from cpt_world.counterfactual_isolation import _worker_main; "
                "raise SystemExit(_worker_main(sys.argv[1:]))"
            ),
            str(request_path),
            str(result_path),
            str(COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES),
            stack_path or "",
        )
        started = time.perf_counter()
        process = subprocess.Popen(command)
        hard_wall_seconds = (
            2.0 * endpoint_time_limit_seconds + _COUNTERFACTUAL_WORKER_GRACE_SECONDS
        )
        deadline = started + hard_wall_seconds
        observed_memory: dict[str, int] = {}
        while time.perf_counter() < deadline and process.poll() is None:
            observed_memory.update(_linux_process_memory_kib(process.pid))
            time.sleep(_COUNTERFACTUAL_WORKER_POLL_SECONDS)

        elapsed_seconds = time.perf_counter() - started
        if process.poll() is None:
            observed_memory.update(_linux_process_memory_kib(process.pid))
            _stop_process(process)
            payload = {
                "status": "hard_timeout",
                "candidate_id": candidate_id,
                "elapsed_seconds": elapsed_seconds,
                "hard_wall_seconds": hard_wall_seconds,
                "memory_limit_bytes": COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES,
                "process_memory_kib": observed_memory,
                "exit_code": process.returncode,
                "stack_path": stack_path,
            }
            path = _write_diagnostic(diagnostic_dir, candidate_id, payload)
            raise CounterfactualResourceLimitError(
                f"counterfactual candidate {candidate_id} exceeded the {hard_wall_seconds:g}s "
                f"hard wall; diagnostic={path}"
            )

        if not result_path.exists():
            payload = {
                "status": "worker_exit",
                "candidate_id": candidate_id,
                "elapsed_seconds": elapsed_seconds,
                "memory_limit_bytes": COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES,
                "process_memory_kib": observed_memory,
                "exit_code": process.returncode,
                "stack_path": stack_path,
            }
            path = _write_diagnostic(diagnostic_dir, candidate_id, payload)
            raise CounterfactualResourceLimitError(
                f"counterfactual candidate {candidate_id} exited without a result; "
                f"diagnostic={path}"
            )

        with result_path.open("rb") as stream:
            message = pickle.load(stream)
        if not isinstance(message, Mapping):
            raise CounterfactualWorkerError("counterfactual worker returned a non-mapping result")
        status = str(message.get("status"))
        if status == "ok":
            _remove_empty_stack(stack_path)
            truth = message.get("truth")
            if not isinstance(truth, Mapping):
                raise CounterfactualWorkerError(
                    "counterfactual worker returned a non-mapping truth"
                )
            return dict(truth)
        if status == "solver_rejected":
            _remove_empty_stack(stack_path)
            raise RuntimeError(
                str(message.get("error", "counterfactual solver rejected candidate"))
            )

        payload = {
            **dict(message),
            "candidate_id": candidate_id,
            "hard_wall_seconds": hard_wall_seconds,
            "memory_limit_bytes": COUNTERFACTUAL_WORKER_MEMORY_LIMIT_BYTES,
            "process_memory_kib": observed_memory,
            "exit_code": process.returncode,
            "stack_path": stack_path,
        }
        path = _write_diagnostic(diagnostic_dir, candidate_id, payload)
        if status == "worker_error":
            raise CounterfactualWorkerError(
                f"counterfactual candidate {candidate_id} failed unexpectedly; "
                f"diagnostic={path}"
            )
        raise CounterfactualResourceLimitError(
            f"counterfactual candidate {candidate_id} failed in isolated worker "
            f"({status}); diagnostic={path}"
        )
    finally:
        request_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
