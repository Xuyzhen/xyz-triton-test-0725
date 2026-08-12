"""Static validation for the generated strict accuracy project."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = 37
ADAPTED = {
    "test_bad_words.py",
    "test_temperature.py",
    "test_gumbel_sample.py",
    "test_topk_log_softmax.py",
    "test_ranks.py",
    "test_min_p.py",
    "test_penalties.py",
    "test_bincount.py",
    "test_prepare_dflash_inputs_kernel.py",
    "test_rejection_kernel.py",
    "test_resample_kernel.py",
    "test_post_update_kernel.py",
    "test_apply_grammar_bitmask_kernel.py",
}
VERSION_ADAPTIVE = {"test_compute_slot_mappings_kernel.py"}
MAIN_ONLY = {
    "test_compute_cumulative_log_p_kernel.py",
    "test_compute_local_residual_mass_kernel.py",
}


def _has_valid_source(path: Path, text: str) -> bool:
    return (
        "accuracy_test/codex/" in text
        or path.name in MAIN_ONLY
        or path.name in VERSION_ADAPTIVE
    )


def main() -> None:
    gpu = sorted((ROOT / "gpu").glob("test_*.py"))
    npu = sorted((ROOT / "npu").glob("test_*.py"))
    assert len(gpu) == EXPECTED, f"expected {EXPECTED} GPU tests, got {len(gpu)}"
    assert len(npu) == EXPECTED, f"expected {EXPECTED} NPU tests, got {len(npu)}"
    assert {p.name for p in gpu} == {p.name for p in npu}

    for path in gpu:
        text = path.read_text(encoding="utf-8")
        assert _has_valid_source(path, text)
        assert "torch.npu" not in text
        assert ".npu()" not in text
        assert "vllm_ascend.worker" not in text

    for path in npu:
        text = path.read_text(encoding="utf-8")
        assert _has_valid_source(path, text)
        if path.name in ADAPTED:
            assert "vllm_ascend.worker" in text, (
                "adapted NPU test does not resolve an Ascend implementation: "
                f"{path.name}"
            )
        if path.name in VERSION_ADAPTIVE:
            assert "ascend_adapted" in text and "upstream_reuse" in text

    for name in MAIN_ONLY:
        for backend in ("gpu", "npu"):
            text = (ROOT / backend / name).read_text(encoding="utf-8")
            assert "_compute_" in text

    print("strict UT static validation passed: 37 GPU + 37 NPU tests")


if __name__ == "__main__":
    main()
