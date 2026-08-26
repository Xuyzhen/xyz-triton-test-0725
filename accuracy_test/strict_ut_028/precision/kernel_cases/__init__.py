"""Kernel case registry.

Each case module must expose:
    KERNEL: str                       registry key
    CASES: list[CaseSpec]             parameterized cases
    build_inputs(params, seed) -> dict[str, torch.Tensor]   # CPU-only tensors
    run(side, tensors, params) -> dict[str, torch.Tensor]   # device in/out

`run` resolves imports lazily per side because the GPU side targets vanilla
vllm Triton kernels while the NPU side targets vllm-ascend kernels, and the
two APIs have already diverged (e.g. gumbel logits-cache).
"""

from __future__ import annotations

import importlib

# kernel name -> module path under this package
REGISTRY = {
    "expand_idx_mapping": "kernel_cases.expand_idx_mapping_cases",
    "penalties": "kernel_cases.penalties_cases",
    "gumbel_sample": "kernel_cases.gumbel_cases",
}


def load_module(kernel: str):
    return importlib.import_module(REGISTRY[kernel])


def all_specs(kernels: list[str] | None = None):
    for kernel, path in REGISTRY.items():
        if kernels and kernel not in kernels:
            continue
        mod = importlib.import_module(path)
        for spec in mod.CASES:
            yield kernel, spec
