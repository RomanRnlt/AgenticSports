"""Shared test fixtures for ReAgt test suite.

Provides AGENT_V3 toggling for testing both v2.0 and v3.0 code paths.
"""

import os
import pytest


@pytest.fixture
def agent_v3():
    """Enable AGENT_V3 for a test."""
    os.environ["AGENT_V3"] = "true"
    yield
    os.environ.pop("AGENT_V3", None)


@pytest.fixture
def agent_v2():
    """Ensure AGENT_V3 is disabled for a test."""
    os.environ.pop("AGENT_V3", None)
    yield
