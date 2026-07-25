# SPDX-License-Identifier: Apache-2.0
"""Benchmark sample_recovered_tokens_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.sample.rejection_sampler import sample_recovered_tokens_kernel


def case_sample_recovered_tokens(device):
    for batch, max_spec_len, vocab in ((4, 5, 1024), (16, 5, 1024)):
        num_tokens = batch * max_spec_len
        cu_num_draft_tokens = torch.arange(1, batch + 1, dtype=torch.int32, device=device) * max_spec_len
        draft_token_ids = torch.randint(0, vocab, (num_tokens,), dtype=torch.int32, device=device)
        draft_probs = torch.rand((num_tokens, vocab), dtype=torch.float32, device=device)
        target_probs = torch.rand((num_tokens, vocab), dtype=torch.float32, device=device)
        inv_q = torch.rand((batch, vocab), dtype=torch.float32, device=device).reciprocal()
        output_token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        BLOCK_SIZE = 1024
        holder = {}

        def fn(_oti=output_token_ids, _cndt=cu_num_draft_tokens, _dti=draft_token_ids, _dp=draft_probs, _tp=target_probs, _iq=inv_q, _v=vocab, _b=batch, _msl=max_spec_len):
            sample_recovered_tokens_kernel[(_b, _msl)](_oti, _cndt, _dti, _dp, _tp, _iq, _v, BLOCK_SIZE, NO_DRAFT_PROBS=False, USE_FP64_GUMBEL=False)
            holder['out'] = _oti
            return holder['out']

        yield (f"batch={batch} max_spec_len={max_spec_len} vocab={vocab}", fn, lambda: int(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_sample_recovered_tokens(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=sample_recovered_tokens_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
