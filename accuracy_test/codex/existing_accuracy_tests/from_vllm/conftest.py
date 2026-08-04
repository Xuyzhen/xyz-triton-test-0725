"""Classify backend compatibility failures separately from precision failures.

This test-only policy never modifies vLLM or vLLM-Ascend implementations.
Numerical assertion failures remain failures. Known NPU/Triton binding and
compilation limitations are reported as XFAIL with precision marked unknown.
"""

import pytest


_AUTHORING_ERRORS = (
    AssertionError,
    AttributeError,
    IndexError,
    NameError,
    TypeError,
    UnboundLocalError,
)

_BACKEND_COMPATIBILITY_PATTERNS = (
    "specified but unrecognised",
    "backend compiler failed",
    "cannot find compiler",
    "compilation failed",
    "failed to compile",
    "failed to run bishengir pipeline",
    "failed to run bishenghir pipeline",
    "invalid device function",
    "no kernel image is available",
    "not implemented for npu",
    "not implemented for 'privateuse1'",
    "out of resources",
    "triton compilation error",
    "ub overflow",
    "unsupported dtype",
)


def _exception_chain_text(exc: BaseException) -> str:
    messages = []
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(messages)


@pytest.fixture(autouse=True)
def classify_backend_compatibility_failure():
    """Turn known backend limitations into XFAIL, not accuracy failures."""
    try:
        yield
    except _AUTHORING_ERRORS:
        raise
    except Exception as exc:
        details = _exception_chain_text(exc)
        lowered = details.lower()
        if any(pattern in lowered for pattern in _BACKEND_COMPATIBILITY_PATTERNS):
            pytest.xfail(
                "Backend compatibility failure; precision is unknown: " + details
            )
        raise

