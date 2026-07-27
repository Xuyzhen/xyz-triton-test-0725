# SPDX-License-Identifier: Apache-2.0
"""Benchmark rejection_greedy_sample_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel, PLACEHOLDER_TOKEN_ID


def case_rejection_greedy_sample(device):
    for batch, max_spec_len, vocab in ((4, 5, 32000), (16, 5, 32000)):
        num_tokens = batch * max_spec_len
        output_token_ids = torch.full((batch, max_spec_len + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device)
        cu_num_draft_tokens = torch.arange(1, batch + 1, dtype=torch.int32, device=device) * max_spec_len
        draft_token_ids = torch.randint(0, vocab, (num_tokens,), dtype=torch.int32, device=device)
        target_argmax = torch.randint(0, vocab, (num_tokens,), dtype=torch.int32, device=device)
        bonus_token_ids = torch.randint(0, vocab, (batch,), dtype=torch.int32, device=device)
        holder = {}

        def fn(_oti=output_token_ids, _cndt=cu_num_draft_tokens, _dti=draft_token_ids, _ta=target_argmax, _bti=bonus_token_ids, _msl=max_spec_len, _b=batch):
            rejection_greedy_sample_kernel[(_b,)](_oti, _cndt, _dti, _ta, _bti, None, _msl, None, None, SYNTHETIC_MODE=False)
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

    for spec, fn, checksum in case_rejection_greedy_sample(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=rejection_greedy_sample_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
