# ===== MODULE DOCSTRING ===== #
"""
DieCast Mold Module

This module provides the entry point for applying type checking to the module that imports it.
It automatically applies the @diecast decorator to eligible functions and methods.

Features:
1. Automatic Application:
   - Applies @diecast to all annotated functions in the calling module
   - Skips already processed functions
   - Handles class methods and static methods
   - Preserves function metadata and docstrings

2. Module Analysis:
   - Identifies the calling module
   - Finds all eligible functions and methods
   - Handles nested classes and functions
   - Respects existing decorators

3. Error Handling:
   - Graceful handling of missing annotations
   - Preserves original function behavior
   - Provides detailed error messages
   - Logs processing status

Usage:
    from diecast import mold

    def process_data(items: list[int]) -> dict:
        return {"sum": sum(items)}

    # Apply type checking to all annotated functions
    mold()
"""

# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from typing import Optional, Final, List, Any
import logging
import inspect
import types
import sys

## ===== LOCAL IMPORTS ===== ##
from .config import _DIECAST_MARKER
from .decorator import diecast
from .logging import _log

# ===== GLOBALS ===== #

## ===== EXPORTS ===== ##
__all__: Final[List[str]] = ['mold']  # Public API of the module

# ===== FUNCTIONS ===== #

## ===== MODULE PROCESSING UTILITIES ===== ##
def _safe_get_file(obj: Any) -> Optional[str]:
    """Safely get the file path of an object, handling TypeError."""
    try:
        return inspect.getfile(obj)
    except TypeError:
        # For objects like built-ins or C extensions where getfile fails
        if hasattr(obj, '__module__'):
            # Try getting module file
            try:
                module = sys.modules.get(obj.__module__)
                if module:
                    return getattr(module, '__file__', None)
            except KeyError:
                pass # Module not in sys.modules
        # Fallback for classes that might have __file__ directly
        return getattr(obj, '__file__', None)
    except Exception as e:
        # Catch other potential unexpected errors during inspection
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE mold._safe_get_file: Unexpected error getting file for {obj!r}: {e}")
        return None

def _get_module_info(frame: types.FrameType) -> tuple[Optional[str], Optional[Any]]:
    """Get module name and object from a frame.
    
    Args:
        frame: The frame to get module info from
        
    Returns:
        Tuple of (module_name, module_object)
    """
    _log.debug(f"TRACE mold._get_module_info: Entering with frame={frame!r}")
    module_name = frame.f_globals.get('__name__')
    if module_name is None:
        _log.debug("TRACE mold._get_module_info: No '__name__' found in frame globals.")
        _log.debug("TRACE mold._get_module_info: Exiting with (None, None)")
        return None, None
    module = sys.modules.get(module_name)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE mold._get_module_info: Found module_name={module_name!r}, module={module!r}")
        _log.debug(f"TRACE mold._get_module_info: Exiting with ({module_name!r}, {module!r})")
    return module_name, module

def _get_module_file(module: Any) -> Optional[str]:
    """Get the file path of a module using the safe helper."""
    _log.debug(f"TRACE mold._get_module_file: Entering with module={module!r}")
    file_path = _safe_get_file(module)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE mold._get_module_file: Found file_path={file_path!r} using _safe_get_file")
        _log.debug(f"TRACE mold._get_module_file: Exiting with {file_path!r}")
    return file_path

def _apply_diecast_safely(target_obj: Any, func_name: str, container_name: str, container_type: str = "module") -> Any:
    """Applies the diecast decorator, logging errors.
    
    Returns the decorated object or the original on error.
    """
    try:
        _log.debug(f"TRACE mold._apply_diecast_safely: Applying diecast to {func_name} in {container_type} '{container_name}'")
        decorated = diecast(target_obj)
        _log.debug(f"TRACE mold._apply_diecast_safely: Successfully applied diecast to {func_name}")
        return decorated
    except Exception as err:
        _log.error(f"Error applying diecast to {func_name} in {container_type} '{container_name}': {err}", exc_info=True)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE mold._apply_diecast_safely: Exception applying diecast to {func_name}: {err!r}")
        return target_obj # Return original on error to avoid breaking the module

def _process_module_function(module: Any, name: str, obj: types.FunctionType, module_file: Optional[str]):
    """Processes a standalone function found in the module."""
    _log.debug(f"TRACE mold._process_module_function: Processing function '{name}'")
    if hasattr(obj, _DIECAST_MARKER):
        _log.debug(f"TRACE mold._process_module_function: Skipping '{name}', already has {_DIECAST_MARKER}")
        return

    obj_file = _safe_get_file(obj)
    has_annotations = getattr(obj, '__annotations__', {})

    if obj_file == module_file and has_annotations:
        _log.debug(f"TRACE mold._process_module_function: Function '{name}' is in module file and has annotations.")
        decorated_obj = _apply_diecast_safely(obj, name, getattr(module, '__name__', 'unknown'))
        if decorated_obj is not obj:
            module.__dict__[name] = decorated_obj
    else:
        if _log.isEnabledFor(logging.DEBUG):
            reason = []
            if obj_file != module_file:
                reason.append(f"file mismatch (expected {module_file!r}, got {obj_file!r})")
            if not has_annotations:
                reason.append("no annotations")
            if not reason:
                reason.append("unknown reason") # Should not happen but good for debugging
            _log.debug(f"TRACE mold._process_module_function: Skipping function '{name}'. Reason: {', '.join(reason)}")

def _process_class_method(cls: type, meth_name: str, meth_obj: Any):
    """Processes a single method within a class."""
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE mold._process_class_method:   Processing method: name={meth_name!r}, obj type={type(meth_obj).__name__}")

    if getattr(meth_obj, _DIECAST_MARKER, False):
        _log.debug(f"TRACE mold._process_class_method:   Skipping method '{meth_name}', already has {_DIECAST_MARKER}")
        return

    # Determine the target function and method type
    target_func = None
    is_classmethod = isinstance(meth_obj, classmethod)
    is_staticmethod = isinstance(meth_obj, staticmethod)
    _log.debug(f"TRACE mold._process_class_method:   Method '{meth_name}': classmethod={is_classmethod}, staticmethod={is_staticmethod}")

    if inspect.isfunction(meth_obj):
        target_func = meth_obj
        _log.debug(f"TRACE mold._process_class_method:   Method '{meth_name}' is a regular function.")
    elif is_classmethod or is_staticmethod:
        try:
            target_func = meth_obj.__func__
            _log.debug(f"TRACE mold._process_class_method:   Accessed underlying function for '{meth_name}': {target_func!r}")
            if getattr(target_func, _DIECAST_MARKER, False):
                _log.debug(f"TRACE mold._process_class_method:   Skipping method '{meth_name}', underlying function has {_DIECAST_MARKER}")
                return  # Skip if underlying func is marked
        except AttributeError:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE mold._process_class_method:   Could not get __func__ for {meth_name} in class {cls.__name__}. Skipping.")
            return
    else:
        _log.debug(f"TRACE mold._process_class_method:   Skipping '{meth_name}', not a function/classmethod/staticmethod.")
        return # Skip non-callable attributes etc.

    # Apply diecast to valid methods with annotations
    if target_func:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE mold._process_class_method:   Target function for '{meth_name}' is {target_func!r}")
        if getattr(target_func, '__isabstractmethod__', False):
            _log.debug(f"TRACE mold._process_class_method:   Skipping abstract method '{meth_name}'")
            return

        annotations = getattr(target_func, '__annotations__', {})
        if annotations:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE mold._process_class_method:   Method '{meth_name}' has annotations: {annotations!r}")
            decorated_func = _apply_diecast_safely(target_func, meth_name, cls.__name__, container_type="class")
            if decorated_func is not target_func:
                # Re-wrap class/static methods correctly
                if is_classmethod:
                    setattr(cls, meth_name, classmethod(decorated_func))
                    _log.debug(f"TRACE mold._process_class_method:   Re-wrapped '{meth_name}' as classmethod.")
                elif is_staticmethod:
                    setattr(cls, meth_name, staticmethod(decorated_func))
                    _log.debug(f"TRACE mold._process_class_method:   Re-wrapped '{meth_name}' as staticmethod.")
                else:
                    setattr(cls, meth_name, decorated_func)  # Regular method
                    _log.debug(f"TRACE mold._process_class_method:   Set '{meth_name}' as regular method.")
        else:
            _log.debug(f"TRACE mold._process_class_method:   Skipping method '{meth_name}', no annotations.")

def _process_module_class(module: Any, name: str, cls: type, module_file: Optional[str]):
    """Process a class found in the module, applying diecast if appropriate."""
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE mold._process_module_class: Processing class '{name}' in module '{module.__name__}'")

    # Check if the class itself was defined in the target module
    try:
        cls_file = _safe_get_file(cls)
        if module_file and cls_file != module_file:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE mold._process_module_class: Skipping class '{name}' (imported from {cls_file}).")
            return
    except TypeError:
        if _log.isEnabledFor(logging.DEBUG):
            _log.warning(f"Could not determine source file for class '{name}'. Applying decorator cautiously.")
        # Proceed cautiously if file check fails

    # --- UPDATED LOGIC: Apply diecast directly to the class --- 
    # Let the @diecast decorator handle class-specific logic (generics, etc.)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE mold._process_module_class: Applying diecast directly to class '{name}'.")
    decorated_cls = _apply_diecast_safely(cls, name, module.__name__, container_type="module class")
    # Decorator modifies the class in-place within the module's namespace.
    # No further update needed here.

## ===== PUBLIC API ===== ##
def mold() -> None:
    """Apply type checking to all annotated functions and methods in the calling module.
    
    This function identifies the calling module, finds eligible functions and classes
    defined within that module, and applies the @diecast decorator.
    
    Skips:
    - Functions/classes without type annotations
    - Functions/classes/methods already processed or marked with @ignore
    - Functions/classes not defined directly within the calling module's file
    - Built-in functions and methods
    - Abstract methods
    
    Note:
        Call at the module level after all definitions.
    """
    _log.debug("TRACE mold.mold: Entering mold()")
    frame = None
    try:
        # 1. Get Calling Module Info
        _log.debug("TRACE mold.mold: Attempting to get frame 1")
        frame = sys._getframe(1)
        if frame is None:
            _log.error("Could not get calling frame")
            _log.debug("TRACE mold.mold: Exiting early - frame is None.")
            return
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE mold.mold: Got frame: {frame!r}")

        module_name, module = _get_module_info(frame)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE mold.mold: Got module info: name={module_name!r}, module={module!r}")
        if module is None:
            _log.error(f"Could not find module '{module_name or 'unknown'}' from frame.")
            _log.debug("TRACE mold.mold: Exiting early - module is None.")
            return

        module_file = _get_module_file(module)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE mold.mold: Got module file: {module_file!r}")
        if module_file is None:
            _log.warning(f"Could not find file for module '{module_name}'. Cannot reliably check function/class origins.")
            # Proceeding without file checking is risky, might wrap imports
            _log.debug("TRACE mold.mold: Proceeding without module_file check - potential risk.")

        # 2. Process Items in Module Namespace
        _log.debug(f"TRACE mold.mold: Starting to process items in module '{module_name}'")
        # Iterate over a copy of items in case the dictionary changes during processing
        items_to_process = list(module.__dict__.items())
        for name, obj in items_to_process:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE mold.mold: Processing item: name={name!r}, obj type={type(obj).__name__}")

            if inspect.isfunction(obj):
                _process_module_function(module, name, obj, module_file)
            elif inspect.isclass(obj):
                _process_module_class(module, name, obj, module_file)
            else:
                 if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE mold.mold: Skipping item '{name}', not a function or class.")

        _log.debug(f"TRACE mold.mold: Finished processing items in module '{module_name}'")

    except Exception as e:
        _log.error(f"Failed to apply type checking via mold(): {e}", exc_info=True)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE mold.mold: Caught top-level exception: {e!r}")
    finally:
        # Clean up frame reference to avoid potential reference cycles
        if frame:
            _log.debug("TRACE mold.mold: Deleting frame reference in finally block.")
            del frame
        _log.debug("TRACE mold.mold: Exiting mold()")
