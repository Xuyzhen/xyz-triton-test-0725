import pytest
import torch

from vllm.v1.worker.gpu.model_states.mamba_hybrid import _scatter_num_accepted_kernel


def _scatter_num_accepted_cpu(
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    max_num_reqs: int,
) -> torch.Tensor:
    num_accepted = torch.zeros(max_num_reqs, dtype=torch.int32)
    for row in range(idx_mapping.shape[0]):
        req_state_idx = int(idx_mapping[row])
        if req_state_idx < 0:
            continue
        sampled = int(num_sampled[row])
        num_accepted[req_state_idx] = max(sampled, 1)
    return num_accepted


@pytest.mark.parametrize(
    "num_reqs,max_num_reqs",
    [(4, 6), (3, 5)],
)
def test_scatter_num_accepted_kernel(num_reqs, max_num_reqs):
    torch.manual_seed(42)
    idx_mapping = torch.tensor([2, 0, -1, 3], dtype=torch.int32)[:num_reqs]
    num_sampled = torch.tensor([5, 0, 3, 1], dtype=torch.int32)[:num_reqs]

    expected = _scatter_num_accepted_cpu(idx_mapping, num_sampled, max_num_reqs)

    device = torch.device("npu")
    idx_mapping_npu = idx_mapping.to(device)
    num_sampled_npu = num_sampled.to(device)
    num_accepted = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    _scatter_num_accepted_kernel[(num_reqs,)](
        idx_mapping_npu,
        num_sampled_npu,
        num_accepted,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_accepted.cpu(), expected, rtol=0, atol=0)


def test_scatter_num_accepted_kernel_clamps_to_one():
    max_num_reqs = 4
    idx_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    num_sampled = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

    expected = _scatter_num_accepted_cpu(idx_mapping, num_sampled, max_num_reqs)

    device = torch.device("npu")
    num_accepted = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    _scatter_num_accepted_kernel[(4,)](
        idx_mapping.to(device),
        num_sampled.to(device),
        num_accepted,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_accepted.cpu(), expected, rtol=0, atol=0)


def test_scatter_num_accepted_kernel_skip_negative():
    max_num_reqs = 3
    idx_mapping = torch.tensor([0, -1, 2], dtype=torch.int32)
    num_sampled = torch.tensor([3, 7, 1], dtype=torch.int32)

    expected = _scatter_num_accepted_cpu(idx_mapping, num_sampled, max_num_reqs)

    device = torch.device("npu")
    num_accepted = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    _scatter_num_accepted_kernel[(3,)](
        idx_mapping.to(device),
        num_sampled.to(device),
        num_accepted,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_accepted.cpu(), expected, rtol=0, atol=0)
