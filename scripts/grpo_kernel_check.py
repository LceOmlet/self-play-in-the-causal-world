"""Fail closed when Qwen3.5 silently falls back from its Hub GDN kernels."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_GDN_KERNELS = frozenset(
    {
        "chunk_gated_delta_rule",
        "recurrent_gated_delta_rule",
    }
)


def enable_local_fla_kernels(model, repository: Path) -> None:
    """Kernelize only the GDN delta-rule functions from an official local repository."""
    from kernels import LocalLayerRepository, Mode, kernelize, use_kernel_mapping

    repository = repository.expanduser().resolve()
    metadata = repository / "build" / "torch-cuda" / "metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"FLA kernel repository is incomplete: {metadata}")

    mapping = {
        layer_name: {
            "cuda": {
                Mode.TRAINING: LocalLayerRepository(
                    repository,
                    layer_name=layer_name,
                )
            }
        }
        for layer_name in REQUIRED_GDN_KERNELS
    }
    with use_kernel_mapping(mapping, inherit_mapping=False):
        kernelize(
            model,
            mode=Mode.TRAINING,
            device="cuda",
            use_fallback=True,
        )


def require_gdn_kernels_active(model) -> None:
    """Verify that every required hidden GDN function has a replaced forward."""
    found: set[str] = set()
    active: dict[str, str] = {}
    for module in model.modules():
        for kernel_func in getattr(module, "_kernel_funcs", {}).values():
            layer_name = getattr(type(kernel_func), "kernel_layer_name", None)
            if layer_name not in REQUIRED_GDN_KERNELS:
                continue
            found.add(layer_name)
            fallback_forward = type(kernel_func).forward
            active_forward = getattr(kernel_func.forward, "__func__", kernel_func.forward)
            if active_forward is not fallback_forward:
                active[layer_name] = active_forward.__module__

    missing = REQUIRED_GDN_KERNELS - found
    fallback = REQUIRED_GDN_KERNELS - active.keys()
    if missing or fallback:
        raise RuntimeError(
            "Required Qwen3.5 GDN Hub kernels are not active: "
            f"missing={sorted(missing)}, fallback={sorted(fallback)}"
        )
    print("GDN_KERNELS=" + json.dumps(active, sort_keys=True), flush=True)
