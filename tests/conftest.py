"""
conftest.py — Pytest configuration.

Marks the entire test suite as requiring a running server at localhost:8000.
Run tests with: pytest tests/ -v  (with uvicorn running in another terminal)
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring a live server at localhost:8000",
    )
