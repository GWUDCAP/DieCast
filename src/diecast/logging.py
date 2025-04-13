# ===== MODULE DOCSTRING ===== #
"""
DieCast Logging Configuration

This module provides a singleton logger for the DieCast package,
obtained via `logging.getLogger(__name__)`.

It follows standard Python library practice by *not* configuring
handlers or levels directly. The application using DieCast is
responsible for configuring the logging system (e.g., setting levels,
adding handlers, specifying formats) as needed.

Usage:
    import logging
    from diecast.logging import logger # diecast package logger

    # Application configures logging (example)
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

    # Use logger in diecast code or application code accessing diecast internals
    logger.debug("Detailed information from DieCast")
"""

# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from typing import Final, List
import logging

# ===== GLOBALS ===== #

## ===== LOGGER INSTANCE ===== ##
# Create the package-specific logger instance
# Configuration (handlers, level, propagation) is left to the application.
_log: Final[logging.Logger] = logging.getLogger('diecast')

## ===== PUBLIC API ALIAS ===== ##
# Provide 'logger' alias for public API compatibility
logger = _log

## ===== EXPORTS ===== ##
__all__: Final[List[str]] = [
    'logger',      # Package logger instance (points to _log)
]

# ===== FUNCTIONS ===== #
# No functions defined in this simplified module. 