"""Smoke tests for the package command-line entry point."""

import subprocess
import sys


def test_app_module_runs() -> None:
    """Running ``python -m app`` should succeed and print the project name."""
    # sys.executable guarantees the subprocess uses the same uv environment as
    # the test runner rather than an unrelated system Python installation.
    result = subprocess.run(
        [sys.executable, "-m", "app"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "SmartGroceryAI"
