"""NPU coverage status for an operator without an Ascend adaptation."""

import pytest


OPERATOR = "{operator}"


@pytest.mark.npu_upstream_unwired
def test_npu_adaptation_status():
    pytest.skip(
        f"{{OPERATOR}} has no vLLM-Ascend adaptation in the installed stack; "
        "the original kernel is covered by the GPU strict UT and is not "
        "launched on NPU"
    )
