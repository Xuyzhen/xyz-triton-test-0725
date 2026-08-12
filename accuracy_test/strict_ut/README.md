# Strict GPU/NPU accuracy UT

This suite is generated only from the reviewed tests under
`accuracy_test/codex/` plus new tests for the two vLLM-main block-verification
kernels.

Run from this directory:

```bash
python -m pytest -c pytest.ini gpu -m gpu -v
python -m pytest -c pytest.ini npu -m npu -v
```

GPU tests always target the original vLLM implementation. NPU tests target a
vLLM-Ascend implementation when one exists, otherwise the upstream kernel
reused by the NPU backend. `npu_upstream_unwired` means the kernel itself is
tested on NPU but the Ascend production wrapper has not wired that feature in.

The ordinary suite is the PR-sized strict subset. Standard 2.1 case-count and
repeat requirements belong to nightly/release jobs and are selected with
markers rather than multiplying pytest nodes in PR CI.
