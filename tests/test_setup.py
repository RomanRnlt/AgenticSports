"""Step 0: Verify project setup, imports, and Gemini API connectivity."""

import os
from pathlib import Path


ROOT = Path(__file__).parent.parent


def test_gemini_api_key_is_set():
    """GEMINI_API_KEY must be present in the environment (loaded from .env)."""
    import src  # noqa: F401 – triggers dotenv loading
    assert os.environ.get("GEMINI_API_KEY"), "GEMINI_API_KEY not set"


def test_package_directories_have_init():
    """All source packages must have __init__.py."""
    for pkg in ["src", "src/agent", "src/tools", "src/memory", "src/interface"]:
        init_file = ROOT / pkg / "__init__.py"
        assert init_file.exists(), f"Missing {init_file}"


def test_data_directories_exist():
    """All data directories must exist."""
    for d in ["data/fit_files", "data/athlete", "data/plans", "data/episodes"]:
        assert (ROOT / d).is_dir(), f"Missing directory {d}"


def test_llm_import():
    """The LLM module must be importable."""
    from src.agent import llm
    assert hasattr(llm, "get_client")
    assert hasattr(llm, "test_connection")


def test_gemini_connection():
    """Integration test: Gemini API returns a non-empty response."""
    from src.agent.llm import test_connection

    result = test_connection()
    assert result, "Gemini returned an empty response"
    assert len(result) > 0
