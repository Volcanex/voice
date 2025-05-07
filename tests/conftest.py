"""
Pytest configuration for Tkinter tests.
"""
import pytest
import tkinter as tk
from unittest.mock import patch

# Mock Tkinter before all tests to avoid creating actual windows
@pytest.fixture(scope="session", autouse=True)
def mock_tkinter():
    """
    Globally mock Tkinter to avoid creating actual windows during tests.
    """
    with patch('tkinter.Tk', autospec=True) as mock_tk:
        with patch('tkinter.Toplevel', autospec=True) as mock_toplevel:
            with patch('tkinter._default_root', None):
                yield