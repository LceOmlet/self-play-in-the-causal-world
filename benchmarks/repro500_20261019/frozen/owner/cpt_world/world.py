"""Action-keyed reproducible random tape for generic WorldSpec episodes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

_WORLDSPEC_TAPE_DOMAIN = b"cpt-world-worldspec-outcome-tape-v1\0"
_WORLDSPEC_OBSERVATION_TAPE_DOMAIN = b"cpt-world-worldspec-observation-tape-v1\0"
_WORLDSPEC_NODE_TAPE_DOMAIN = b"cpt-world-worldspec-ancestral-node-tape-v2\0"
_WORLDSPEC_OBSERVATION_NODE_TAPE_DOMAIN = (
    b"cpt-world-worldspec-observation-ancestral-node-tape-v2\0"
)


@dataclass(frozen=True, slots=True)
class OutcomeTape:
    """A deterministic potential-outcome stream keyed by intervention arm."""

    tape_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.tape_key, str) or not self.tape_key:
            raise ValueError("tape_key must be a nonempty string")
        if len(self.tape_key.encode("utf-8")) >= 2**32:
            raise ValueError("tape_key is too long")

    def worldspec_uniform(self, target: int, state: int, sample_index: int) -> Fraction:
        """Return one stable draw for a canonical WorldSpec intervention arm."""

        for value, field in ((target, "target"), (state, "state")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if not 0 <= value < 2**32:
                raise ValueError(f"{field} must lie in [0, 2^32)")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError("sample_index must be an integer")
        if not 0 <= sample_index < 2**64:
            raise ValueError("sample_index must lie in [0, 2^64)")
        key = self.tape_key.encode("utf-8")
        payload = b"".join(
            (
                _WORLDSPEC_TAPE_DOMAIN,
                len(key).to_bytes(4, "big"),
                key,
                target.to_bytes(4, "big"),
                state.to_bytes(4, "big"),
                sample_index.to_bytes(8, "big"),
            )
        )
        digest = hashlib.sha256(payload).digest()
        return Fraction(int.from_bytes(digest, "big"), 2**256)

    def worldspec_observation_uniform(self, sample_index: int) -> Fraction:
        """Return one stable draw from the natural, non-intervened world arm."""

        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError("sample_index must be an integer")
        if not 0 <= sample_index < 2**64:
            raise ValueError("sample_index must lie in [0, 2^64)")
        key = self.tape_key.encode("utf-8")
        payload = b"".join(
            (
                _WORLDSPEC_OBSERVATION_TAPE_DOMAIN,
                len(key).to_bytes(4, "big"),
                key,
                sample_index.to_bytes(8, "big"),
            )
        )
        digest = hashlib.sha256(payload).digest()
        return Fraction(int.from_bytes(digest, "big"), 2**256)

    def worldspec_node_uniform(
        self,
        target: int,
        state: int,
        sample_index: int,
        node: int,
    ) -> Fraction:
        """Return one v2 draw for a node in an intervention-arm sample."""

        for value, field, width in (
            (target, "target", 4),
            (state, "state", 4),
            (sample_index, "sample_index", 8),
            (node, "node", 4),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if not 0 <= value < 2 ** (8 * width):
                raise ValueError(f"{field} must lie in [0, 2^{8 * width})")
        key = self.tape_key.encode("utf-8")
        payload = b"".join(
            (
                _WORLDSPEC_NODE_TAPE_DOMAIN,
                len(key).to_bytes(4, "big"),
                key,
                target.to_bytes(4, "big"),
                state.to_bytes(4, "big"),
                sample_index.to_bytes(8, "big"),
                node.to_bytes(4, "big"),
            )
        )
        return Fraction(int.from_bytes(hashlib.sha256(payload).digest(), "big"), 2**256)

    def worldspec_observation_node_uniform(
        self,
        sample_index: int,
        node: int,
    ) -> Fraction:
        """Return one v2 draw for a node in a passive-observation sample."""

        for value, field, width in (
            (sample_index, "sample_index", 8),
            (node, "node", 4),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if not 0 <= value < 2 ** (8 * width):
                raise ValueError(f"{field} must lie in [0, 2^{8 * width})")
        key = self.tape_key.encode("utf-8")
        payload = b"".join(
            (
                _WORLDSPEC_OBSERVATION_NODE_TAPE_DOMAIN,
                len(key).to_bytes(4, "big"),
                key,
                sample_index.to_bytes(8, "big"),
                node.to_bytes(4, "big"),
            )
        )
        return Fraction(int.from_bytes(hashlib.sha256(payload).digest(), "big"), 2**256)
