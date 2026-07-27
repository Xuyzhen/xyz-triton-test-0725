# Upstream vLLM Triton kernel smoke test

## Scope

These tests answer one narrow question:

> Can an upstream vLLM Triton kernel be imported, compiled by Triton Ascend,
> and launched once on an NPU?

The tests intentionally set `VLLM_PLUGINS=""` and import upstream vLLM kernel
modules directly. They do not validate normal vLLM Ascend plugin dispatch or
end-to-end model execution. A `PASS` is a compile-and-launch result, not a
numerical-correctness result.

## Environment check

Run inside the vLLM Ascend development container with compatible vLLM, vLLM
Ascend, PyTorch, torch-npu, Triton Ascend, and CANN versions.

```bash
python3 -c "import torch, torch_npu, triton, vllm, vllm_ascend; print(torch.__version__); print(torch_npu.__version__); print(triton.__version__); print(vllm.__version__); print(vllm_ascend.__version__)"
npu-smi info
```

The JSON report records installed Python package versions automatically. Record
the vLLM and vLLM Ascend Git revisions alongside the report.

## Run one kernel

```bash
cd /home/lingmutian/code/vllm-ascend/benchmarks/ops
python3 run_upstream_smoke.py \
  --device npu:0 \
  --match eagle_prepare_next_token_padded
```

The runner forces `--warmup 0 --repeat 1`, so an in-place kernel sees the
original input state exactly once.

A benchmark can also be invoked directly:

```bash
python3 x_benchmark_mrv2_upstream__eagle_prepare_next_token_padded_kernel.py \
  --device npu:0 --warmup 0 --repeat 1
```

## Run all kernels

```bash
cd /home/lingmutian/code/vllm-ascend/benchmarks/ops
python3 run_upstream_smoke.py --device npu:0 --timeout 300
```

The runner continues after individual failures and writes a JSON report under
`benchmark/results/`. Its exit code is zero only when all selected scripts
pass.

Use repeated `--match` arguments for a subset:

```bash
python3 run_upstream_smoke.py --device npu:0 \
  --match eagle \
  --match rejection
```

Use a stable report path in CI:

```bash
python3 run_upstream_smoke.py \
  --device npu:0 \
  --output ../results/upstream_smoke.json
```

## Statuses

| Status | Meaning |
| --- | --- |
| `PASS` | Every case in the script imported and launched once. |
| `IMPORT_ERROR` | The installed vLLM lacks the imported module or symbol. |
| `UNSUPPORTED_DTYPE` | Input construction or execution used an unsupported NPU dtype. |
| `COMPILE_ERROR` | Triton Ascend failed while compiling the kernel. |
| `RUNTIME_ERROR` | The process failed for another execution reason. |
| `PROCESS_SIGNAL` | The child process was terminated by a signal. |
| `TIMEOUT` | The script exceeded `--timeout`. |

Inspect the `output` field for every non-`PASS` result. Missing `vllm._C`
warnings do not fail a script that exits successfully.

## Performance and correctness follow-up

After a smoke-test pass, performance can be sampled directly:

```bash
python3 x_benchmark_mrv2_upstream__topk_topp_kernel.py \
  --device npu:0 --warmup 20 --repeat 100
```

Do not interpret repeated execution of an in-place kernel as a correctness
test. Numerical validation requires fresh inputs and a CPU, PyTorch, or trusted
upstream reference implementation.
