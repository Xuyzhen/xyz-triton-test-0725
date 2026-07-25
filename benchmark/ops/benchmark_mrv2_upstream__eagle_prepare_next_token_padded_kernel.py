# SPDX-License-Identifier: Apache-2.0
"""Benchmark eagle_prepare_next_token_padded_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.spec_decode.utils import eagle_prepare_next_token_padded_kernel


def case_eagle_prepare_next_token_padded(device):
    for num_reqs_val, num_spec_tokens in ((4, 5), (16, 5)):
        num_sampled_tokens_per_req = num_spec_tokens + 1
        vocab = 32000
        BLOCK_SIZE_TOKENS = 1
        while BLOCK_SIZE_TOKENS < num_sampled_tokens_per_req:
            BLOCK_SIZE_TOKENS *= 2

        sampled_token_ids = torch.randint(0, vocab, (num_reqs_val, num_sampled_tokens_per_req), dtype=torch.int32, device=device)
        sampled_token_ids[:, -1] = -1
        discard_request_mask = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        backup_next_token_ids = torch.randint(0, vocab, (num_reqs_val,), dtype=torch.int32, device=device)
        next_token_ids = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = torch.zeros(num_reqs_val, dtype=torch.uint32, device=device)
        holder = {}

        def fn(_sti=sampled_token_ids, _drm=discard_request_mask, _bnti=backup_next_token_ids, _nti=next_token_ids, _vstc=valid_sampled_tokens_count, _v=vocab, _nstpr=num_sampled_tokens_per_req, _nr=num_reqs_val, _str=sampled_token_ids.stride(0), _bst=BLOCK_SIZE_TOKENS):
            eagle_prepare_next_token_padded_kernel[(_nr,)](_sti, _drm, _bnti, _nti, _vstc, _v, _nstpr, _nr, _str, BLOCK_SIZE_TOKENS=_bst)
            holder['out'] = _nti
            return holder['out']

        yield (f"num_reqs={num_reqs_val} num_spec={num_spec_tokens}", fn, lambda: int(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_eagle_prepare_next_token_padded(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=eagle_prepare_next_token_padded_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
