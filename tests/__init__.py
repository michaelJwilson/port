"""Test suite for port."""

from pathlib import Path

TESTS = Path(__file__).resolve().parent
"""This directory: the one place the tests locate themselves (#749 WP11)."""

ROOT = TESTS.parent
"""The repository root, as the tests see it."""
