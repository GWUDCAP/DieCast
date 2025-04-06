import pytest
import sys
import os
from typing import Set

# Add src dir to path to allow importing diecast
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from diecast import config

def test_color_constants_exist_and_are_strings():
    """Verify color constants exist and are strings."""
    assert isinstance(config.COLOR_RED, str)
    assert isinstance(config.COLOR_YELLOW_ORANGE, str)
    assert isinstance(config.COLOR_BLUE, str)
    assert isinstance(config.COLOR_CYAN, str)
    assert isinstance(config.COLOR_BOLD, str)
    assert isinstance(config.COLOR_RESET, str)

def test_display_constants_exist_and_are_ints():
    """Verify display setting constants exist and are integers."""
    assert isinstance(config.DEFAULT_TERMINAL_WIDTH, int)
    assert isinstance(config.MAX_VALUE_REPR_LENGTH, int)
    assert isinstance(config.MAX_FRAMES_TO_ANALYZE, int)

def test_type_checking_constants_exist_and_correct_type():
    """Verify type checking constants exist and have correct types."""
    assert isinstance(config._SELF_NAMES, Set)
    assert all(isinstance(name, str) for name in config._SELF_NAMES)
    assert isinstance(config._RETURN_ANNOTATION, str)
    assert isinstance(config._DIECAST_MARKER, str)

# Check if color constants exist and are strings
# assert isinstance(config.COLOR_RED, str)
# assert isinstance(config.COLOR_RESET, str)

# TODO: Add tests for diecast.config settings and functions 