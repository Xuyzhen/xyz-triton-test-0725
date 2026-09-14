"""Fast import smoke test for the NPU strict suite."""

import importlib


MODULES = [
    "accuracy_test.strict_ut.runtime_npu",
    "vllm_ascend.worker.v2.structured_outputs",
    "vllm_ascend.worker.v2.sample.bad_words",
    "vllm_ascend.worker.v2.sample.gumbel",
    "vllm_ascend.worker.v2.sample.logprob",
    "vllm_ascend.worker.v2.sample.min_p",
    "vllm_ascend.worker.v2.sample.penalties",
    "vllm_ascend.worker.v2.block_table",
    "vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils",
]


def main() -> None:
    for module_name in MODULES:
        importlib.import_module(module_name)
        print(f"OK {module_name}")


if __name__ == "__main__":
    main()
