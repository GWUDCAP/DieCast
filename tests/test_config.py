# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from typing import Set
import sys
import os

## ===== LOCAL IMPORTS ===== ##
from src.diecast import config


# ===== SETUP (Add src to path) ===== #
# TODO: Consider moving sys.path modification to pytest configuration (e.g., pytest.ini or conftest.py)
# This ensures 'diecast' can be imported correctly during test runs.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


# ===== TEST FUNCTIONS ===== #
def test_color_constants_exist_and_are_strings():
    """Verify color constants exist and are strings."""
    assert isinstance(config.COLOR_RED, str)
    assert isinstance(config.COLOR_YELLOW_ORANGE, str)
    assert isinstance(config.COLOR_BLUE, str)
    assert isinstance(config.COLOR_CYAN, str)
    assert isinstance(config.COLOR_BOLD, str)
    assert isinstance(config.COLOR_RESET, str)

def test_display_constants_exist_and_are_ints():
    """Verify display setting constants exist, are integers, and have correct values."""
    # Check types
    assert isinstance(config.DEFAULT_TERMINAL_WIDTH, int)
    assert isinstance(config.MAX_VALUE_REPR_LENGTH, int)
    assert isinstance(config.MAX_FRAMES_TO_ANALYZE, int)
    # Check values against spec
    assert config.DEFAULT_TERMINAL_WIDTH == 80
    assert config.MAX_VALUE_REPR_LENGTH == 100
    assert config.MAX_FRAMES_TO_ANALYZE == 30

def test_type_checking_constants_exist_and_correct_type():
    """Verify type checking constants exist, have correct types, and correct values."""
    # Check types
    assert isinstance(config._SELF_NAMES, Set)
    assert all(isinstance(name, str) for name in config._SELF_NAMES)
    assert isinstance(config._RETURN_ANNOTATION, str)
    assert isinstance(config._DIECAST_MARKER, str)
    # Check values against spec
    assert config._SELF_NAMES == {'self', 'cls'}
    assert config._RETURN_ANNOTATION == 'return'
    assert config._DIECAST_MARKER == '_diecast_marker'