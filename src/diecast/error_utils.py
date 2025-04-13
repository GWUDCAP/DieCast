# ===== MODULE DOCSTRING ===== #
"""Error utilities for the DieCast package, including custom exceptions and message formatting."""
#-#

# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from typing import (
    Dict, List, Any, TypeVar,
    Optional, Union, Final
)
import dataclasses
import inspect
import logging
##-##

## ===== LOCAL ===== ##
from .config import (
    MAX_VALUE_REPR_LENGTH, COLOR_YELLOW_ORANGE, COLOR_BLUE,
    COLOR_CYAN, COLOR_BOLD, COLOR_RESET, COLOR_RED
)
from .logging import _log
##-##
#-#

# ===== GLOBALS ===== #

## ===== LOGGER ===== ##
_log: Final[logging.Logger] = logging.getLogger('diecast')
##-##

## ===== EXPORTS ===== ##
__all__: Final[List[str]] = [
    'Obituary',
    '_create_obituary',
    '_construct_type_error',
    '_format_path',
    '_get_caller_info',
    'generate_arg_error_message',
    'generate_return_error_message',
]
##-##
#-#

# ===== CLASSES ===== #
@dataclasses.dataclass(frozen=True)
class Obituary:
    """Holds structured details about a type check failure.

    Used as the return value alongside False from check_type when a mismatch occurs.

    Attributes:
        expected_repr (str): String representation of the expected type.
        received_repr (str): String representation of the received type.
        value (Any): The actual value that failed the check.
        path (List[Union[str, int]]): Path to the failure within nested structures.
        message (Optional[str]): Specific failure reason message.
    """
    expected_repr: str
    received_repr: str
    value: Any
    path: List[Union[str, int]]
    message: Optional[str] = None
#-#

# ===== FUNCTIONS ===== #
def _create_obituary(expected_repr: str, received_repr: str, value: Any, path: List[Union[str, int]], message: Optional[str] = None) -> Obituary:
    """Creates a structured object containing details about a type check failure.

    Args:
        expected_repr: String representation of the expected type.
        received_repr: String representation of the received value's type.
        value: The actual value that failed the check.
        path: The path (list of indices/keys) to the location of the failure.
        message: An optional specific message about the failure reason.

    Returns:
        An Obituary object with failure details.
    """
    # This function is simple and called only on failure, low logging priority.
    return Obituary(
        expected_repr=expected_repr,
        received_repr=received_repr,
        value=value,
        path=path,
        message=message
    )

def _construct_type_error(obituary: Optional[Obituary]) -> str:
    """Constructs a basic error message string from Obituary details."""
    if obituary is None:
        return "Type check failed with unspecified details."
    
    base_msg = f"Expected {obituary.expected_repr}, got {obituary.received_repr}"
    if obituary.message:
        base_msg += f". Reason: {obituary.message}"
    if obituary.path:
        base_msg += f" at path '{_format_path(obituary.path)}'"
    return base_msg

def _format_path(path: List[Union[str, int]]) -> str:
    """Format the failure path list into a readable string (e.g., "['key'][0]['nested']")."""
    # Simple function, logging likely not needed unless debugging path formatting itself.
    if not path:
        return ""
    result = ""
    for item in path:
        if isinstance(item, int):
            result += f"[{item}]"
        elif isinstance(item, str):
            # Always use bracket notation with repr for strings
            result += f"[{item!r}]"
        else:
            # Fallback for other segment types (should be rare)
            result += f"[{item!r}]"
    return result

def _get_caller_info(depth: int = 1) -> Dict[str, Any]:
    """Get information about the caller's stack frame.

    Dynamically searches up the stack to find the first frame outside
    the diecast module.

    Args:
        depth: Initial search depth (ignored, kept for compatibility).

    Returns:
        A dictionary containing filename, lineno, function name, and
        code context of the relevant caller frame, or empty strings/None
        if unavailable.
    """
    _log.debug(f"TRACE type_utils._get_caller_info: Entering (initial depth={depth}). Searching for non-diecast frame.")
    try:
        frame = inspect.currentframe()
        if frame is None:
            _log.warning("_get_caller_info: Could not get current frame.")
            return {'filename': '', 'lineno': 0, 'function': '', 'code_context': None}

        # Start searching from the caller of _get_caller_info
        search_frame = frame.f_back
        while search_frame:
            module_name = search_frame.f_globals.get('__name__', '')
            if not 'diecast' in module_name:
                # Found the first frame outside the diecast module
                caller_frame_info = inspect.getframeinfo(search_frame)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._get_caller_info: Found non-diecast caller frame: {caller_frame_info.filename}:{caller_frame_info.lineno} in {caller_frame_info.function}")
                # Make sure context is a list or None
                code_context = caller_frame_info.code_context
                if code_context and not isinstance(code_context, list):
                    code_context = list(code_context)
                    
                return {
                    'filename': caller_frame_info.filename or '',
                    'lineno': caller_frame_info.lineno or 0,
                    'function': caller_frame_info.function or '',
                    'code_context': code_context
                }
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._get_caller_info: Skipping internal frame in module: {module_name}")
            search_frame = search_frame.f_back

        # If loop finishes without finding a non-diecast frame (unlikely)
        _log.warning("_get_caller_info: Could not find a caller frame outside the diecast module.")
        return {'filename': '', 'lineno': 0, 'function': '', 'code_context': None}

    except Exception as e:
        _log.warning(f"Error getting caller info: {e}", exc_info=True)
        return {'filename': '', 'lineno': 0, 'function': '', 'code_context': None}
    finally:
        # Explicitly delete frame objects to help garbage collection
        # Note: inspect.currentframe() does not guarantee frames are kept alive,
        # but this is good practice based on documentation.
        if 'frame' in locals(): del frame
        if 'search_frame' in locals(): del search_frame
        if 'caller_frame_info' in locals(): del caller_frame_info

def _generate_error_message_core(
    func_name: str, func_module: str, func_lineno: int,
    check_type: str, # 'Argument' or 'Return Value'
    original_annotation: Any, # <<< Added: The original type hint object
    obituary: Obituary, # <<< CHANGED from details Dict to Obituary object
    caller_info: Dict[str, Any], # Info from _get_caller_info
    # --- ADDED func_class_name ---
    func_class_name: Optional[str] = None,
    # --- END ADDITION ---
    param_name: Optional[str] = None, # Only for arguments
    arg_index: Optional[int] = None, # Only for arguments
    is_kwarg: Optional[bool] = None, # Only for arguments
    is_yield_value: bool = False # <<< Added for yield errors
) -> str:
    """Core logic to construct the detailed error message using Obituary details."""
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._generate_error_message_core: Entering. CheckType='{check_type}', Func='{func_name}'")
    # Determine check type string based on flag
    check_type_str = "Yield value" if is_yield_value else check_type
    
    # --- ADDED: Prepend class name if present --- #
    full_func_name = f"{func_class_name}.{func_name}" if func_class_name else func_name
    # --- END ADDITION ---
    
    # Extract details FROM OBITUARY
    expected_type_repr = obituary.expected_repr
    received_type = obituary.received_repr
    value_repr = repr(obituary.value) # Re-generate repr here? Or store it?
    # Truncate long value representations
    if len(value_repr) > MAX_VALUE_REPR_LENGTH:
        value_repr = value_repr[:MAX_VALUE_REPR_LENGTH] + f"... (truncated at {MAX_VALUE_REPR_LENGTH} chars)"
    path = obituary.path
    message = obituary.message

    # --- Refactored Message Construction ---
    from .type_utils import format_type_for_display # Local import

    # 1. Simple Header
    simple_header = f"{COLOR_RED}{COLOR_BOLD}YouDiedError:{COLOR_RESET} "
    if check_type_str == 'Argument':
        simple_header += f"Argument '{COLOR_BOLD}{param_name}{COLOR_RESET}' FAILED type check"
    else: # Return or Yield
        simple_header += f"{check_type_str} FAILED type check"

    # 2. Function Info - Match test format like "Function: _err_basic_func(...)"
    func_loc_info = f"Function: {COLOR_BOLD}{full_func_name}{COLOR_RESET}(module: {func_module}, line: {func_lineno})"

    # 3. Caller Location - Match test format
    caller_file = caller_info.get('filename', 'unknown')
    caller_line = caller_info.get('lineno', 'unknown')
    # Example from a log: Location: /home/somekidpunk/gwudcap/diecast/src/diecast/decorator.py:108
    caller_loc = f"Location: {COLOR_BLUE}{caller_file}:{caller_line}{COLOR_RESET}" # Seems correct based on logs

    # 4. Expected Type (Show original TypeVar if applicable)
    expected_info = f"Expected: {COLOR_CYAN}{obituary.expected_repr}{COLOR_RESET}" # Default
    if original_annotation is not None and isinstance(original_annotation, TypeVar):
        original_repr = format_type_for_display(original_annotation)
        # Only add original if it's different from the resolved one shown in obituary
        if original_repr != obituary.expected_repr:
            expected_info += f" (original: {COLOR_CYAN}{original_repr}{COLOR_RESET})(Resolved: {COLOR_YELLOW_ORANGE}{obituary.expected_repr}{COLOR_RESET})"

    # 5. Received Value/Type
    received_info = f"Received: {COLOR_YELLOW_ORANGE}{value_repr}{COLOR_RESET} ({COLOR_CYAN}{received_type}{COLOR_RESET})"

    # 6. Path (Optional)
    path_info = ""
    if path:
        formatted_path = _format_path(path)
        path_info = f"Path: {COLOR_BLUE}{formatted_path}{COLOR_RESET}"

    # 7. Reason (Optional)
    reason_info = ""
    if message:
        reason_info = f"Reason: {COLOR_RED}{message}{COLOR_RESET}"

    # Combine Parts (New Order)
    # Assemble message parts in the order observed in failing test logs/assertions
    message_parts = [
        simple_header,      # YouDiedError: Argument 'x' FAILED type check
        func_loc_info,      # Function: func_name(module: mod, line: 123)
        caller_loc,         # Location: /path/to/caller.py:456
        expected_info,      # Expected: str
        received_info,      # Received: 1 (int)
    ]
    if path_info:           # Path: ['key'][0]
        message_parts.append(path_info)
    if reason_info:         # Reason: Value is not an instance of expected type
        message_parts.append(reason_info)

    # Join with newlines and ensure a trailing newline
    final_message = "\n".join(message_parts) + "\n"

    # Future: Add optional code snippet from caller_info['code_context']
    # Future: Add optional traceback details
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._generate_error_message_core: Generated message (length={len(final_message)}).")
    return final_message

def generate_arg_error_message(
    func_name: str, func_module: str, func_lineno: int,
    param: inspect.Parameter, annotation: Any, value: Any,
    arg_index: int, is_kwarg: bool,
    caller_info: Dict[str, Any],
    obituary: Obituary, # <<< CHANGED from failure_details Dict to Obituary
    original_annotation: Optional[Any] = None,
    func_class_name: Optional[str] = None
) -> str:
    """Generate a detailed TypeError message for an argument mismatch using Obituary details."""
    # Simple wrapper, logging maybe less critical here.
    # Pass the original annotation to the core function
    # Value is now directly available in Obituary
    # if 'received_value' not in failure_details:
    #      failure_details['received_value'] = value

    return _generate_error_message_core(
        func_name=func_name, func_module=func_module, func_lineno=func_lineno,
        check_type='Argument',
        original_annotation=original_annotation, # <<< Pass original annotation
        obituary=obituary, # Pass Obituary object
        caller_info=caller_info,
        func_class_name=func_class_name, 
        param_name=param.name,
        arg_index=arg_index,
        is_kwarg=is_kwarg
    )

def generate_return_error_message(
    func_name: str, func_module: str, func_lineno: int,
    annotation: Any, value: Any,
    caller_info: Dict[str, Any],
    obituary: Obituary, # <<< CHANGED from failure_details Dict to Obituary
    original_annotation: Optional[Any] = None,
    func_class_name: Optional[str] = None,
    is_yield_value: bool = False
) -> str:
    """Generate a detailed TypeError message for a return value mismatch using Obituary details."""
    # Simple wrapper, logging maybe less critical here.
    # Pass the original annotation to the core function
    # Value is now directly available in Obituary
    # if 'received_value' not in failure_details:
    #      failure_details['received_value'] = value

    return _generate_error_message_core(
        func_name=func_name, func_module=func_module, func_lineno=func_lineno,
        check_type='Return value', # Base check type
        original_annotation=annotation, # Pass original annotation
        obituary=obituary, # Pass Obituary object
        caller_info=caller_info,
        # --- ADDED func_class_name --- #
        func_class_name=func_class_name, 
        # --- END ADDITION ---
        # Pass is_yield_value through if provided (defaults to False)
        # Maybe get this from obituary.cause if added?
        is_yield_value=is_yield_value
    )
#-#