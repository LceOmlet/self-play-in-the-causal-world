"""Project boundary adapter; execution and dispatch remain in upstream libraries.

This module deliberately has no unsandboxed session factory. Provisioning a
validated NsJail/Jupyter session is required before use with a model.
"""
from __future__ import annotations

import inspect
import json
import weakref
from collections.abc import Callable
from typing import Any


class OutputLimitError(RuntimeError):
    pass


class JupyterNumericSession:
    """Use an already isolated official BlockingKernelClient, never host exec."""

    def __init__(self, client: Any, destroy: Callable[[], None]) -> None:
        self._client = client
        self._destroy = destroy
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            # A failed destroy must not be silently accepted as isolation.
            self._destroy()
            self.closed = True

    def execute(self, code: str, public_context: dict) -> str:
        if self.closed:
            raise RuntimeError("Numerical session is closed")
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        chunks: list[str] = []
        size = 0

        def output(message: dict) -> None:
            nonlocal size
            kind = message["header"]["msg_type"]
            content = message["content"]
            if kind == "stream":
                text = content.get("text", "")
            elif kind in {"execute_result", "display_data"}:
                text = content.get("data", {}).get("text/plain", "")
            elif kind == "error":
                text = f"{content.get('ename', 'Error')}: {content.get('evalue', '')}"
            else:
                return
            size += len(text.encode("utf8"))
            if size > 8192:
                raise OutputLimitError("Calculation output exceeds 8192 UTF-8 bytes")
            chunks.append(text)

        # JSON contains only model-visible input, never the environment object,
        # task row, seed/tape, truth, grader, credentials or trainer state.
        # This string is submitted to the remote kernel, never evaluated here.
        context_literal = repr(json.dumps(public_context, ensure_ascii=False))
        combined = (
            "import numpy as np\nimport scipy as sp\n"
            f"public = __import__('json').loads({context_literal})\n"
            + code
        )
        try:
            reply = self._client.execute_interactive(
                code=combined, timeout=30, allow_stdin=False,
                store_history=False, stop_on_error=True, output_hook=output,
            )
        except Exception as error:
            self.close()
            # Transport exceptions can contain private connection filenames.
            reason = "output_limit" if isinstance(error, OutputLimitError) else "execution_or_transport_failure"
            return json.dumps({"error": reason, "state_lost": True,
                               "message": "Session destroyed; the next calculation starts empty."})
        status = reply.get("content", {}).get("status")
        if status not in {"ok", "error"}:
            self.close()
            return json.dumps({"error": "unexpected_kernel_reply", "state_lost": True})
        return "".join(chunks)


def make_numeric_environment(base_environment: type, session_factory: Callable):
    """Add exactly one tool to the existing causal owner via TRL's public hook."""
    assigned_sessions = weakref.WeakSet()

    class CPTWorldNumericEnvironment(base_environment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._numeric_session = None
            self._visible = None

        def _close_numeric_session(self):
            if self._numeric_session is not None:
                self._numeric_session.close()
                self._numeric_session = None

        def reset(self, *, prompt: object, **kwargs) -> str:
            """Discard previous computation state before resetting the causal task."""
            self._close_numeric_session()
            self._visible = None
            instruction = super().reset(prompt=prompt, **kwargs)
            visible_prompt = json.loads(json.dumps(prompt, ensure_ascii=False))
            numeric_instruction = (
                "\nThe only tools are act and numerics. Use numerics(code) for ordinary "
                "Python with NumPy as np and SciPy as sp. Variables persist within this "
                "answer only. public contains the visible prompt and your act feedback. "
                "Each calculation has 30 seconds and 8192 UTF-8 output bytes; the session "
                "has 1 CPU and 1 GiB memory. Errors that destroy the session report state_lost. "
                "Submit the terminal answer through act."
            )
            self._visible = {"prompt": visible_prompt, "observations": []}
            return (instruction or "") + numeric_instruction

        def act(self, command: dict[str, object]) -> str:
            """Perform the original causal experiment or submit the terminal answer.

            Args:
                command: The original task protocol's JSON command.

            Returns:
                Unchanged owner-rendered feedback or terminal acknowledgement.
            """
            result = super().act(command)
            if self._visible is not None:
                self._visible["observations"].append({
                    "command": json.loads(json.dumps(command, ensure_ascii=False)),
                    "feedback": result,
                })
            if self.episode is not None and self.episode.completed:
                self._close_numeric_session()
            return result

        def numerics(self, code: str) -> str:
            """Compute with full NumPy and SciPy in this answer's isolated Python session.

            Args:
                code: Ordinary Python numerical code. Print a compact result.

            Returns:
                Numerical output or an explicit error; never a causal experiment or grade.
            """
            if self._visible is None or self.episode is None:
                raise RuntimeError("Environment must be reset before numerics")
            if self.episode.completed:
                return "Episode already complete. Make no more tool calls."
            if self._numeric_session is None or self._numeric_session.closed:
                candidate = session_factory()
                if candidate.closed:
                    raise RuntimeError("Session factory returned a retired session")
                if candidate in assigned_sessions:
                    raise RuntimeError("Session factory attempted to share an answer session")
                assigned_sessions.add(candidate)
                self._numeric_session = candidate
            payload = json.loads(json.dumps(self._visible, ensure_ascii=False))
            return self._numeric_session.execute(code, payload)

        def get_reward(self) -> float:
            """Return the owner's unchanged reward and release even truncated answers."""
            try:
                return super().get_reward()
            finally:
                self._close_numeric_session()

    # Guard the official discovery surface, without replacing its dispatcher.
    exposed = {name for name, method in inspect.getmembers(CPTWorldNumericEnvironment,
               predicate=inspect.isfunction)
               if not name.startswith("_") and name not in {"reset", "get_reward"}}
    if exposed != {"act", "numerics"}:
        raise ValueError(f"Unexpected public environment methods: {sorted(exposed)}")
    return CPTWorldNumericEnvironment
