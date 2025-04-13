# ===== MODULE DOCSTRING ===== #
"""
DieCast Configuration Module

This module contains configuration constants and settings used throughout
the DieCast package. It provides centralized configuration for:

1. Terminal Output:
   - Color codes for different types of output
   - Display formatting and styling
   - Terminal width and text wrapping

2. Type Checking:
   - Internal markers for tracking processed functions
   - Special parameter names (self/cls)
   - Return value annotation handling

3. Error Reporting:
   - Stack trace analysis limits
   - Value representation limits
   - Error message formatting

Note:
    The color codes use True Color ANSI escape sequences, which may not
    work on all terminals. The package will gracefully handle terminals
    that don't support these codes.
"""
#-#

# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from typing import Final, Set
import logging
##-##
#-#

# ===== GLOBALS ===== #

## ===== LOGGER ===== ##
_log = logging.getLogger('diecast')
##-##

## ===== TERMINAL OUTPUT ===== ##
# True Color ANSI escape sequences for terminal output
# These colors are used for different parts of error messages and output formatting
COLOR_RED: Final[str] = "\033[38;2;255;51;51m"          # Error messages and type violations
COLOR_YELLOW_ORANGE: Final[str] = "\033[38;2;255;204;102m"  # Titles and section headers
COLOR_BLUE: Final[str] = "\033[38;2;83;154;252m"         # File locations and function names
COLOR_CYAN: Final[str] = "\033[38;2;89;194;255m"         # Type information and annotations
COLOR_BOLD: Final[str] = "\033[1m"                       # Emphasis and important text
COLOR_RESET: Final[str] = "\033[0m"                      # Reset all formatting
##-##

## ===== DISPLAY SETTINGS ===== ##
# Terminal display and formatting configuration
DEFAULT_TERMINAL_WIDTH: Final[int] = 80      # Default width for text wrapping
MAX_VALUE_REPR_LENGTH: Final[int] = 100      # Maximum length for value representation in error messages
MAX_FRAMES_TO_ANALYZE: Final[int] = 30       # Maximum number of stack frames to analyze for error context
##-##

## ===== TYPE CHECKING ===== ##
# Special parameter names and annotations for type checking
_SELF_NAMES: Final[Set[str]] = {'self', 'cls'}     # Parameter names that indicate instance/class methods
_RETURN_ANNOTATION: Final[str] = 'return'          # Special key for return type annotations

# Internal markers for tracking processed/ignored functions
_DIECAST_MARKER: Final[str] = '_diecast_marker'    # Indicates a function has been processed or should be ignored
##-##
#-#