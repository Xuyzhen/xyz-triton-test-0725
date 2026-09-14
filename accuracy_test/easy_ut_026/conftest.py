# SPDX-License-Identifier: Apache-2.0
"""easy_ut_026 conftest: register deterministic marker used by tests.

Backend compilation/import failures are surfaced as pytest.fail (not xfail),
matching strict_ut policy: a collected strict test must either pass or fail
visibly.
"""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.deterministic)
