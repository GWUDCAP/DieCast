# ===== MODULE DOCSTRING ===== #
"""
DieCast Logging Configuration

This module configures and manages logging for the DieCast package.
It provides a centralized logger with configurable verbosity levels
and consistent formatting across the package.

The logger is configured with the following defaults:
- Output: Standard error stream (sys.stderr)
- Format: "%(levelname)s:%(name)s: %(message)s"
- Default Level: WARNING
- Propagation: Disabled (messages don't reach root logger)

Features:
- Singleton logger instance for consistent logging across the package
- Configurable verbosity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Prevents duplicate handler registration
- Thread-safe logging

Usage:
    from diecast.logging import logger, set_verbosity
    import logging

    # Set logging level (e.g., to see debug messages)
    set_verbosity(logging.DEBUG)
    
    # Use logger in your code
    logger.debug("Detailed information")
    logger.info("General information")
    logger.warning("Warning messages")
    logger.error("Error messages")
"""

# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from typing import Final, List
import logging
import sys

# ===== GLOBALS ===== #

## ===== CONSTANTS ===== ##
# Logging format string for consistent message formatting
LOG_FORMAT: Final[str] = '%(levelname)s:%(name)s: %(message)s'

# Valid logging levels for verbosity configuration
VALID_LEVELS: Final[List[int]] = [
    logging.DEBUG,
    logging.INFO,
    logging.WARNING,
    logging.ERROR,
    logging.CRITICAL
]

## ===== LOGGER SETUP ===== ##
# Create the package-specific logger instance
_log: Final[logging.Logger] = logging.getLogger('diecast')

# Configure stream handler for stderr output
handler: Final[logging.StreamHandler] = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Initialize logger configuration if not already done
if not _log.handlers:
    _log.addHandler(handler)
    # Note: Propagation is currently ENABLED. If the intent was to prevent
    # propagation (as a previous comment suggested), this should be False.
    # This potential functional issue is logged in phase_A_potential_bugs.md.
    _log.propagate = True
    _log.setLevel(logging.WARNING)  # Set default level to WARNING
    # Removed debug print statement during Phase A cleanup.
    # print(f"\n*** diecast.logging SETUP: Logger ID={id(_log)}, Handlers={_log.handlers}, Level={_log.level} ***\n", file=sys.stderr)
    # Consider using a log message instead if setup confirmation is needed:
    if _log.isEnabledFor(logging.INFO):
        _log.info(f"DieCast logger initialized. Level: {logging.getLevelName(_log.level)}")

## ===== PUBLIC API ALIAS ===== ##
# Provide 'logger' alias for public API compatibility
logger = _log

## ===== EXPORTS ===== ##
__all__: Final[List[str]] = [
    'logger',      # Package logger instance (points to _log)
    'set_verbosity',  # Function to configure logging verbosity
]

# ===== FUNCTIONS ===== #

def set_verbosity(level: int) -> None:
    """Set the logging verbosity level for the DieCast logger.
    
    This function allows users to control the verbosity of DieCast's logging output.
    It validates the provided level and updates the logger's level accordingly.
    
    Args:
        level: A logging level constant from the logging module
              (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    
    Raises:
        ValueError: If an invalid logging level is provided
        
    Example:
        >>> import logging
        >>> from diecast.logging import set_verbosity
        >>> set_verbosity(logging.DEBUG)  # Enable debug messages
    """
    _log.debug(f"TRACE logging.set_verbosity: Entering with level={level!r}")
    if level not in VALID_LEVELS:
        _log.debug(f"TRACE logging.set_verbosity: Invalid level provided: {level!r}")
        raise ValueError(
            f"Invalid logging level: {level}. "
            f"Use logging module constants (e.g., logging.DEBUG). "
            f"Valid levels: {[logging.getLevelName(l) for l in VALID_LEVELS]}"
        )
    
    _log.setLevel(level)
    # Performance guard for function call in log message
    if _log.isEnabledFor(logging.DEBUG):
        level_name = logging.getLevelName(level)
        _log.debug(f"TRACE logging.set_verbosity: DieCast verbosity set to {level_name}") 