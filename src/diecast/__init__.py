# ===== MODULE DOCSTRING ===== #
"""
DieCast - Runtime Type Checking for Python

This package provides runtime type checking for Python functions and methods.
It enforces type hints at runtime, providing detailed error messages when type
violations occur.

Features:
- Runtime type checking for function arguments and return values
- Support for complex types (Union, Optional, List, Dict, etc.)
- Detailed error messages with variable names and stack traces
- Automatic type checking for entire modules using mold()
- Standard Python logging integration (logger name 'diecast', requires user configuration)
- Thread-safe operation

Basic Usage:
    import diecast

    @diecast.diecast
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    # Type checking is enforced
    greet("World")  # Returns "Hello, World!"
    greet(123)      # Raises YouDiedError (subclass of TypeError) with detailed error message

Using the mold function:
    # In your module
    import diecast
    
    def process_data(items: list[int]) -> dict:
        return {"processed": sum(items)}
        
    def another_func(flag: bool) -> None:
        print(flag)

    # Apply type checking to all annotated functions/methods 
    # defined in this module above this line
    diecast.mold()

    # Now calls to process_data and another_func will be type-checked
    process_data([1, 2]) # OK
    # process_data(["a"]) # Raises YouDiedError
    another_func(True) # OK
    # another_func(1) # Raises YouDiedError
Version Management:
    The package version is managed through the __version__ variable.
    For development, this is a static version number. For releases,
    this should be updated to match the release version.
"""
#-#

# ===== IMPORTS ===== #

## ===== LOCAL ===== ##
from .logging import _log
from .decorator import diecast, ignore
from .mold import mold
##-##
#-#

# ===== GLOBALS ===== #

## ===== VERSION ===== ##
__version__ = "0.1.0"
##-##

## ===== API SETUP ===== ##
# Attach ignore as an attribute to the main decorator
diecast.ignore = ignore
##-##

## ===== PUBLIC API ALIAS ===== ##
# Provide 'logger' alias for public API compatibility
logger = _log
##-##

## ===== EXPORTS ===== ##
# Core functionality
__all__ = [
    'diecast',      # Main decorator for type checking
    'ignore',       # Decorator to ignore type checking (accessible via diecast.ignore)
    'mold',         # Function to apply type checking to a module
]
##-##
#-#

# Configuration functions (set_verbosity removed from public API)

# Debugging utilities
__all__.append('logger')  # Logger instance for debugging (points to _log)
