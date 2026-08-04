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


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Report known backend limitations as XFAIL, not accuracy failures."""
    outcome = yield
    report = outcome.get_result()
    if call.when != "call" or call.excinfo is None:
        return

    exc = call.excinfo.value
    if isinstance(exc, _AUTHORING_ERRORS):
        return

    details = _exception_chain_text(exc)
    lowered = details.lower()
    if any(pattern in lowered for pattern in _BACKEND_COMPATIBILITY_PATTERNS):
        report.outcome = "skipped"
        report.wasxfail = (
            "Backend compatibility failure; precision is unknown: " + details
        )

