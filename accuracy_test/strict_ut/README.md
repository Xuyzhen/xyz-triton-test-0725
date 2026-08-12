# Strict GPU/NPU accuracy UT

This suite is generated only from the reviewed tests under
`accuracy_test/codex/` plus new tests for the two vLLM-main block-verification
kernels.

Run from this directory:

```bash
python -m pytest -c pytest.ini gpu -m gpu -v
python -m pytest -c pytest.ini npu -m npu -v
```

On an Ascend host, run the lightweight import smoke test first:

```bash
python check_npu_imports.py
```

The strict NPU runtime avoids importing
`vllm_ascend.ops.triton.triton_utils` through the normal package path. That
path executes `vllm_ascend.ops.__init__` and imports fused MoE, which can
break unrelated Triton test collection when vLLM and vLLM-Ascend expose
different `FusedMoE` APIs. A narrow test-only shim provides only the device
property helpers; target kernels still come from their real
`vllm_ascend.worker...` modules.

GPU tests always target the original vLLM implementation. NPU tests target a
vLLM-Ascend implementation when one exists, otherwise the upstream kernel
reused by the NPU backend. `npu_upstream_unwired` means the kernel itself is
tested on NPU but the Ascend production wrapper has not wired that feature in.

The ordinary suite is the PR-sized strict subset. Standard 2.1 case-count and
repeat requirements belong to nightly/release jobs and are selected with
markers rather than multiplying pytest nodes in PR CI.
