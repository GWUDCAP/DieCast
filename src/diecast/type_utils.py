# ===== MODULE DOCSTRING ===== #
"""
Type checking utilities for DieCast.

This module provides utility functions for working with type annotations
and performing runtime type checking. It handles various type scenarios
including:
- Basic type checking
- Generic type checking
- Union types
- Optional types
- Forward references
- TypeVar resolution
- Method resolution order (MRO) caching

The module is designed to work seamlessly with Python's typing system
and provides detailed error messages when type violations occur.
"""

# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from functools import lru_cache
import collections.abc
from typing import (
    get_type_hints, get_origin, 
    get_args, _GenericAlias, 
    Dict, List, Set, Any,    
    Sequence, Optional, 
    Callable, Literal, 
    Final, ForwardRef, 
    Mapping, Tuple, 
    Union, Type, 
    TypeVar, 
)
import dataclasses
import threading
import inspect
import logging
import typing
import types
import sys

try:
    from typing import (
        get_origin as typing_get_origin, 
        get_args as typing_get_args
    )
    from typing import Annotated # >= 3.10 / 3.12
except ImportError: # Python 3.7/3.8 compatibility
    typing_get_origin = lambda tp: getattr(tp, "__origin__", None)
    typing_get_args = lambda tp: getattr(tp, "__args__", ())
    try:
        # Try typing_extensions for Annotated
        from typing_extensions import Annotated
    except ImportError:
        Annotated = None # Define as None if not available

## ===== LOCAL ===== ##
from .error_utils import (
    _construct_type_error, 
    _get_caller_info, 
    _format_path, 
    YouDiedError, 
    Obituary
)

# ===== GLOBALS ===== #

## ===== TYPE ALIASES ===== ##
NoneType: Final[Type[None]] = type(None)

## ===== LOGGER ===== ##
_log: Final[logging.Logger] = logging.getLogger('diecast')

## ===== MRO CACHE ===== ##
# Global cache for MRO sets (maps type -> set of MRO types)
_mro_cache: Dict[type, Set[type]] = {}
_mro_cache_lock = threading.Lock() # Thread safety for MRO cache

## ===== CHECK_TYPE CACHE ===== ##
# Global cache for TypeVar bindings within a specific function call context
# Maps (func_id, TypeVar) -> concrete_type
_TYPEVAR_BINDINGS: Dict[Tuple[int, TypeVar], type] = {}
_typevar_bindings_lock = threading.Lock() # Thread safety for bindings

## ===== OBITUARY CACHE ===== ##
# Global cache for check_type results (used for safe-to-cache types)
# Maps (value_type, expected_type, func_id, path_tuple) -> (bool_match, Optional[Dict]_fail_details)
_check_type_cache_obituary: Dict[Tuple[type, Any, Optional[str], Optional[Tuple[Union[str, int], ...]]], Tuple[bool, Optional[Obituary]]] = {}
_check_type_cache_lock = threading.Lock()

## ===== TYPE HANDLERS ===== ##
TypeCheckHandler3Args = Callable[[Any, Any, List[Union[str, int]]], Optional[Tuple[bool, Optional[Obituary]]]]
TypeCheckHandler5Args = Callable[[Any, Any, Dict[str, Any], Optional[Dict[str, Any]], List[Union[str, int]]], Optional[Tuple[bool, Optional[Obituary]]]]

# ===== FUNCTIONS ===== #

## ===== MRO CACHE ===== ##
def get_cached_mro_set(value_type: Type) -> Set[Type]:
    """Calculates and caches the Method Resolution Order (MRO) set for a given type.

    Uses a lock for thread safety during cache writes and double-checking
    to minimize lock contention. Falls back gracefully if MRO calculation fails.

    Args:
        value_type: The type for which to get the MRO set.

    Returns:
        A set containing the types in the MRO of value_type.
    """
    # Logging inside the lock for cache misses
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils.get_cached_mro_set: Cache miss for type: {value_type!r}. Acquiring lock.")
    with _mro_cache_lock: # Acquire lock only if type is not in cache
        # Double check cache after acquiring lock
        cached_result = _mro_cache.get(value_type)
        if cached_result is not None:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.get_cached_mro_set: Found in cache after acquiring lock for: {value_type!r}")
                _log.debug(f"TRACE type_utils.get_cached_mro_set: Exiting (cache hit after lock)")
            return cached_result
        
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.get_cached_mro_set: Confirmed cache miss for {value_type!r}. Calculating MRO.")
        try:
            # Calculate MRO set (potentially expensive)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.get_cached_mro_set: Calling inspect.getmro({value_type!r})")
            mro = inspect.getmro(value_type)
            mro_set = set(mro)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.get_cached_mro_set: Calculated MRO for {value_type!r}: {mro_set!r}")
            _mro_cache[value_type] = mro_set
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.get_cached_mro_set: Exiting (calculated)")
            return mro_set
        except Exception as e:
            if _log.isEnabledFor(logging.DEBUG):
                _log.warning(f"Failed to calculate MRO for {value_type!r}: {e}. Performance may be affected.", exc_info=True)
            mro_set = {value_type}
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.get_cached_mro_set: Exiting (fallback after exception)")
            _mro_cache[value_type] = mro_set
            return mro_set

def is_instance_optimized(value: Any, expected_type: Type) -> bool:
    """Checks isinstance using the MRO cache for potential speedup.

    Performs a direct type check first, then uses the cached MRO set.
    Falls back to standard `isinstance` if MRO caching fails or raises an error.

    Args:
        value: The value to check.
        expected_type: The type to check against.

    Returns:
        True if the value is an instance of the expected type, False otherwise.
    """
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils.is_instance_optimized: Entering with value={value!r} (type={type(value).__name__}), expected_type={expected_type!r}")
    value_type = type(value)
    if value_type is expected_type:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils.is_instance_optimized: Direct type match. Result: True")
            _log.debug("TRACE type_utils.is_instance_optimized: Exiting")
        return True
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug("TRACE type_utils.is_instance_optimized: Direct type mismatch.")
    try:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils.is_instance_optimized: Checking MRO cache...")
        mro_set = get_cached_mro_set(value_type)
        if expected_type in mro_set:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.is_instance_optimized: Expected type {expected_type!r} found in MRO set {mro_set!r}. Result: True")
                _log.debug("TRACE type_utils.is_instance_optimized: Exiting")
            return True
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.is_instance_optimized: Expected type {expected_type!r} NOT in MRO set {mro_set!r}. Falling back to isinstance.")
        result = isinstance(value, expected_type)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.is_instance_optimized: Fallback isinstance({value!r}, {expected_type!r}) result: {result}")
            _log.debug("TRACE type_utils.is_instance_optimized: Exiting")
        return result
    except Exception as e: # Catch potential errors during MRO lookup or isinstance
        _log.warning(f"Instance check failed for {value_type!r} against {expected_type!r}. Falling back. Error: {e!r}", exc_info=True)
        try:
            result = isinstance(value, expected_type)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.is_instance_optimized: Final fallback isinstance({value!r}, {expected_type!r}) result: {result}")
                _log.debug("TRACE type_utils.is_instance_optimized: Exiting (after exception fallback)")
            return result
        except TypeError as te:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.is_instance_optimized: Final fallback isinstance raised TypeError: {te!r}. Result: False")
                _log.debug("TRACE type_utils.is_instance_optimized: Exiting (after exception fallback TypeError)")
            return False

## ===== TYPEVAR HANDLING ===== ##
def bind_typevar(func_id: int, typevar: TypeVar, concrete_type: Any) -> None:
    """Bind a TypeVar to a concrete type for the duration of a function call.

    Used to enforce consistency within a single function execution context.

    Args:
        func_id: The unique ID of the function execution context (e.g., id(wrapper)).
        typevar: The TypeVar instance to bind.
        concrete_type: The concrete type to bind the TypeVar to.
    """
    with _typevar_bindings_lock:
        if func_id not in _TYPEVAR_BINDINGS:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.bind_typevar: Creating new binding dict for func_id={func_id}")
            _TYPEVAR_BINDINGS[func_id] = {}

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.bind_typevar [func_id={func_id}]: Binding {typevar!r} -> {concrete_type!r}")
        _TYPEVAR_BINDINGS[func_id][typevar] = concrete_type

def get_typevar_binding(func_id: int, typevar: TypeVar) -> Optional[Any]:
    """Get the concrete type that a TypeVar is bound to for a function call.

    Args:
        func_id: The unique ID of the function execution context.
        typevar: The TypeVar instance to look up.

    Returns:
        The concrete type the TypeVar is bound to, or None if not bound in this context.
    """
    with _typevar_bindings_lock:
        bindings = _TYPEVAR_BINDINGS.get(func_id, {})
        bound_type = bindings.get(typevar)
        if _log.isEnabledFor(logging.DEBUG):
            if bound_type is not None:
                _log.debug(f"TRACE type_utils.get_typevar_binding [func_id={func_id}]: Found binding for {typevar!r}: {bound_type!r}")
            else:
                _log.debug(f"TRACE type_utils.get_typevar_binding [func_id={func_id}]: No binding found for {typevar!r}")
        return bound_type

def clear_typevar_bindings(func_id: int) -> None:
    """Clear all TypeVar bindings for a function context when it completes.

    Args:
        func_id: The unique ID of the function execution context to clear bindings for.
    """
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils.clear_typevar_bindings: Entering for func_id={func_id}")
    with _typevar_bindings_lock:
        if func_id in _TYPEVAR_BINDINGS:
            if _log.isEnabledFor(logging.DEBUG):
                bindings = _TYPEVAR_BINDINGS[func_id]
                _log.debug(f"TRACE type_utils.clear_typevar_bindings [func_id={func_id}]: Clearing bindings: {bindings!r}")
            del _TYPEVAR_BINDINGS[func_id]
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.clear_typevar_bindings [func_id={func_id}]: Bindings cleared.")
        else:
            _log.debug(f"TRACE type_utils.clear_typevar_bindings [func_id={func_id}]: No bindings found to clear.")
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils.clear_typevar_bindings: Exiting for func_id={func_id}")

## ===== TYPE INTROSPECTION ===== ##
def get_origin(tp: Any) -> Optional[Any]:
    """Get the unsubscripted origin of a generic type (e.g., list for List[int]).

    Provides compatibility across Python versions.

    Args:
        tp: The type annotation.

    Returns:
        The origin type, or None if not a generic type.
    """
    # Simple function, no logging needed unless debugging typing internals
    return typing_get_origin(tp)

def get_args(tp: Any) -> Tuple[Any, ...]:
    """Get the arguments of a generic type (e.g., (int,) for List[int]).

    Provides compatibility across Python versions.

    Args:
        tp: The type annotation.

    Returns:
        A tuple of type arguments, or an empty tuple if not applicable.
    """
    # Simple function, no logging needed unless debugging typing internals
    return typing_get_args(tp)

def is_optional_type(tp: Any) -> Tuple[bool, Any]:
    """Check if a type is Optional[X] (i.e., Union[X, NoneType]).

    Args:
        tp: The type annotation.

    Returns:
        Tuple[bool, Any]: (True, X) if it's Optional[X], otherwise (False, tp).
    """
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        if len(args) == 2 and NoneType in args:
            inner_type = args[0] if args[1] is NoneType else args[1]
            return True, inner_type
    return False, tp

def is_union_type(tp: Any) -> Tuple[bool, Tuple[Any, ...]]:
    """Check if a type is Union[X, Y, ...] (including Optional[X]).

    Args:
        tp: The type annotation.

    Returns:
        Tuple[bool, Tuple[Any, ...]]: (True, (X, Y, ...)) if it's a Union/Optional,
        otherwise (False, (tp,)). Returns the args *including* NoneType if present.
    """
    origin = get_origin(tp)
    if origin is Union:
        return True, get_args(tp)
    if hasattr(types, 'UnionType') and isinstance(tp, types.UnionType):
        return True, get_args(tp)
    return False, (tp,)

def is_generic_alias(tp: Any) -> bool:
    """Check if a type is a generic alias (e.g., List[int], Dict[str, Any]).

    Args:
        tp: The type annotation.

    Returns:
        True if it's a generic alias, False otherwise.
    """
    return isinstance(tp, (_GenericAlias, getattr(types, 'GenericAlias', type(None))))

def resolve_forward_ref(ref: Union[str, ForwardRef], globalns: Dict[str, Any], localns: Optional[Dict[str, Any]] = None) -> Type:
    """Resolve a forward reference string or ForwardRef object to an actual type.

    Uses the appropriate evaluation mechanism based on Python version.

    Args:
        ref: The forward reference string or ForwardRef object.
        globalns: The global namespace for resolution.
        localns: The local namespace for resolution (optional).

    Returns:
        The resolved type object.

    Raises:
        NameError: If the forward reference cannot be resolved.
        Exception: Other errors during evaluation.
    """
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils.resolve_forward_ref: Entering with ref={ref!r}")
        _log.debug(f"TRACE type_utils.resolve_forward_ref: Namespaces: globalns keys={list(globalns.keys())!r}, localns keys={(list(localns.keys()) if localns else None)!r}")
    
    if localns is None:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils.resolve_forward_ref: localns is None, falling back to globalns.")
        localns = globalns # Fallback for older resolve mechanisms

    if isinstance(ref, ForwardRef):
        ref_str = ref.__forward_arg__
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.resolve_forward_ref: ref is ForwardRef, using __forward_arg__: '{ref_str}'")
    elif isinstance(ref, str):
        ref_str = ref
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.resolve_forward_ref: ref is str: '{ref_str}'")
    else:
        if _log.isEnabledFor(logging.DEBUG):
            _log.error(f"TRACE type_utils.resolve_forward_ref: Invalid type for ref: {type(ref)}. Raising TypeError.")
        raise TypeError(f"Expected str or ForwardRef, got {type(ref)}")

    # Special handling for 'self' if it appears in localns (usually in methods)
    if ref_str == 'self' and localns is not None and 'self' in localns:
        _log.debug("TRACE type_utils.resolve_forward_ref: ref_str is 'self' and found in localns.")
        self_obj = localns['self']
        if self_obj is not None:
            resolved_type = self_obj.__class__
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.resolve_forward_ref: Resolved 'self' to {resolved_type!r} via localns['self'].__class__")
            _log.debug("TRACE type_utils.resolve_forward_ref: Exiting")
            return resolved_type
        else:
            _log.debug("TRACE type_utils.resolve_forward_ref: localns['self'] is None. Falling through to standard resolution.")
            # This case should be rare (self=None), but fall through to normal resolution
            pass

    try:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.resolve_forward_ref: Attempting resolution for '{ref_str}'")
        
        # Use ForwardRef._evaluate if available (Python 3.9+)
        if hasattr(ForwardRef, '_evaluate'):
            _log.debug("TRACE type_utils.resolve_forward_ref: Using ForwardRef._evaluate")
            # The _evaluate method itself is potentially slow
            resolved_type = ForwardRef(ref_str)._evaluate(globalns, localns or {}, frozenset()) # Pass empty dict if localns is None
        else:
            # Fallback for older versions (eval is necessary here and slow)
            _log.debug("TRACE type_utils.resolve_forward_ref: Using eval() fallback")
            # pylint: disable=eval-used
            resolved_type = eval(ref_str, globalns, localns or {}) # Pass empty dict if localns is None
        
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.resolve_forward_ref: Successfully resolved '{ref_str}' to {resolved_type!r}")
        _log.debug("TRACE type_utils.resolve_forward_ref: Exiting")
        return resolved_type
    except NameError as e:
        if _log.isEnabledFor(logging.DEBUG):
            _log.error(f"TRACE type_utils.resolve_forward_ref: NameError resolving '{ref_str}': {e}", exc_info=False) # Keep log clean
        raise NameError(f"Could not resolve forward reference '{ref_str}': {e}") from e
    except Exception as e:
        if _log.isEnabledFor(logging.DEBUG):
            _log.error(f"TRACE type_utils.resolve_forward_ref: Exception resolving '{ref_str}': {e!r}", exc_info=True)
        raise Exception(f"Error resolving forward reference '{ref_str}': {e}") from e

def get_resolved_type_hints(obj: Callable, globalns: Optional[Dict[str, Any]] = None, localns: Optional[Dict[str, Any]] = None, include_extras: bool = False) -> Dict[str, Any]:
    """Safely get type hints, resolving forward references.
    # ... (rest of docstring) ...
    """
    if _log.isEnabledFor(logging.DEBUG):
        obj_name = getattr(obj, '__name__', str(obj))
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.get_resolved_type_hints: Entering for obj='{obj_name}'")
            _log.debug(f"TRACE type_utils.get_resolved_type_hints: include_extras={include_extras}")

    resolved_hints = {}
    try:
        # --- Primary Strategy: Attempt full resolution --- 
        # Determine appropriate namespaces if not provided
        obj_globalns = getattr(obj, '__globals__', {}) if globalns is None else globalns
        # Determine localns more carefully
        effective_localns = localns
        if localns is None:
            # Try to get class namespace for methods
            if inspect.ismethod(obj):
                bound_instance = getattr(obj, '__self__', None)
                if bound_instance is not None:
                    cls = bound_instance if isinstance(bound_instance, type) else type(bound_instance)
                    try:
                        effective_localns = dict(vars(cls)) # May fail for some builtins
                        if _log.isEnabledFor(logging.DEBUG):
                            _log.debug(f"TRACE type_utils.get_resolved_type_hints: Using class dict for localns: {list(effective_localns.keys())!r}")
                    except TypeError:
                        if _log.isEnabledFor(logging.DEBUG):
                            _log.debug(f"TRACE type_utils.get_resolved_type_hints: Could not get vars(cls) for method localns.")
                        effective_localns = None # Fallback if vars() fails
            # Keep effective_localns as None if not a method or class dict failed
            if effective_localns is None:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug("TRACE type_utils.get_resolved_type_hints: effective_localns remains None.")
                pass # Keep it None
        else:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("TRACE type_utils.get_resolved_type_hints: Using provided localns.")
            pass

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.get_resolved_type_hints: Calling get_type_hints with globalns keys={list(obj_globalns.keys())!r}, localns keys={(list(effective_localns.keys()) if effective_localns else None)!r}")
        
        # Performance: get_type_hints can be slow, involves resolution
        if sys.version_info >= (3, 9):
            resolved_hints = get_type_hints(obj, globalns=obj_globalns, localns=effective_localns, include_extras=include_extras)
        else:
        # include_extras not supported before 3.9
            if include_extras:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.warning("'include_extras=True' requires Python 3.9+. Ignoring.")
            resolved_hints = get_type_hints(obj, globalns=obj_globalns, localns=effective_localns)
        
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.get_resolved_type_hints: Successfully got hints via get_type_hints: {resolved_hints!r}")

    except NameError as e:
        # --- Fallback Strategy: Handle NameError (likely ForwardRef) --- 
        obj_name = getattr(obj, '__name__', str(obj))
        if _log.isEnabledFor(logging.DEBUG):
            _log.warning(f"NameError during initial hint resolution for '{obj_name}': {e}. Falling back to manual ForwardRef creation.", exc_info=False)
        fallback_hints = {}
        raw_annotations = getattr(obj, '__annotations__', {})
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.get_resolved_type_hints: Processing raw annotations: {raw_annotations!r}")
        for name, annotation in raw_annotations.items():
            if isinstance(annotation, str):
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils.get_resolved_type_hints: Creating ForwardRef for '{name}: {annotation}'")
                fallback_hints[name] = ForwardRef(annotation)
            else:
                # Keep non-string annotations (types, complex structures like List[ForwardRef(...)]) as is
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils.get_resolved_type_hints: Keeping original annotation for '{name}: {annotation!r}'")
                fallback_hints[name] = annotation
        resolved_hints = fallback_hints # Use the manually constructed dictionary
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.get_resolved_type_hints: Fallback hints created: {resolved_hints!r}")

    except (TypeError, Exception) as e: 
        # --- Error Handling: Catch other critical errors --- 
        obj_name = getattr(obj, '__name__', str(obj))
        log_func = _log.error if isinstance(e, Exception) else _log.warning
        log_func(f"{type(e).__name__} getting type hints for '{obj_name}': {e!r}. Returning empty hints.", exc_info=isinstance(e, Exception))
        resolved_hints = {} # Return empty dict on these critical errors

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.get_resolved_type_hints: Exiting")
    return resolved_hints

## ===== TYPE FORMATTING ===== ##
@lru_cache(maxsize=512)
def format_type_for_display(tp: Any) -> str:
    """Format a type annotation into a user-friendly string representation.

    Handles special cases like Optional, Union, and ForwardRef, aiming for
    readability in error messages.

    Args:
        tp: The type annotation.

    Returns:
        A string representation of the type.
    """
    # Performance: This function can be recursive and potentially complex.
    # Most operations are simple lookups/checks, but string formatting
    # and recursion add overhead. Using @lru_cache helps a lot.
    if tp is Any: return "Any"
    if tp is NoneType: return "None"
    if isinstance(tp, TypeVar): return str(tp) # e.g., "~T"
    if isinstance(tp, ForwardRef): 
        result = tp.__forward_arg__
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.format_type_for_display: Formatted ForwardRef({tp!r}) -> '{result}'")
        return result # Show the string itself

    # Handle Optional[X]
    is_opt, inner_type = is_optional_type(tp)
    if is_opt:
        # Recursive call needs guard if format_type_for_display itself becomes very complex
        # Assuming it's fast enough for now.
        formatted_inner = format_type_for_display(inner_type)
        result = f"Optional[{formatted_inner}]"
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils.format_type_for_display: Formatted Optional({tp!r}) -> '{result}'")
        return result

    # Handle Union[X, Y, ...] (excluding Optional which was handled above)
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        # Ensure it's not just Optional disguised as Union
        if not (len(args) == 2 and NoneType in args):
            # Performance guard for list comprehension + recursive calls
            if _log.isEnabledFor(logging.DEBUG):
                formatted_args_list = [format_type_for_display(t) for t in args]
                formatted_args_str = ", ".join(formatted_args_list)
                result = f"Union[{formatted_args_str}]"
                _log.debug(f"TRACE type_utils.format_type_for_display: Formatted Union({tp!r}) -> '{result}'")
                return result
            else:
                 # Avoid intermediate list if not logging
                 formatted_args_str = ", ".join(format_type_for_display(t) for t in args)
                 return f"Union[{formatted_args_str}]"

    # Handle Literal[...]
    # Check if Literal exists before checking origin (avoid AttributeError)
    if hasattr(typing, 'Literal') and origin is Literal:
        literal_args = get_args(tp)
        # Performance guard for list comprehension with repr
        if _log.isEnabledFor(logging.DEBUG):
            formatted_args_list = [repr(a) for a in literal_args]
            formatted_args_str = ", ".join(formatted_args_list)
            result = f"Literal[{formatted_args_str}]"
            _log.debug(f"TRACE type_utils.format_type_for_display: Formatted Literal({tp!r}) -> '{result}'")
            return result
        else:
            formatted_args_str = ", ".join(repr(a) for a in literal_args)
            return f"Literal[{formatted_args_str}]"

    # Handle Callable[[...], ...] specifically
    elif origin is collections.abc.Callable:
        args = get_args(tp)
        # Check for the expected structure: ([arg_types], return_type)
        if args and len(args) == 2 and isinstance(args[0], list):
            arg_types = args[0]
            return_type = args[1]
            # Recursively format argument types and return type
            formatted_arg_types = ", ".join(format_type_for_display(at) for at in arg_types)
            formatted_return_type = format_type_for_display(return_type)
            result = f"Callable[[{formatted_arg_types}], {formatted_return_type}]"
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.format_type_for_display: Formatted Callable({tp!r}) -> '{result}'")
            return result
        else: # Handle bare Callable or unexpected args format
            result = "Callable" # Default representation
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils.format_type_for_display: Formatted bare Callable or unexpected args ({tp!r}) -> '{result}'")
            return result

    # Handle Generics (List[int], Dict[str, float], etc.) - Changed from if to elif
    elif origin:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"!!! FORMATTER DEBUG: Processing tp={tp!r}, origin={origin!r}")
        origin_name = getattr(origin, '__name__', str(origin))
        args = get_args(tp)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"!!! FORMATTER DEBUG: origin_name={origin_name!r}, args={args!r}")
        if args:
            # Performance guard for list comprehension + recursive calls
            if _log.isEnabledFor(logging.DEBUG):
                formatted_args_list = [format_type_for_display(a) for a in args]
                formatted_args_str = ", ".join(formatted_args_list)
                result = f"{origin_name}[{formatted_args_str}]"
                _log.debug(f"TRACE type_utils.format_type_for_display: Formatted Generic/Origin({tp!r}) -> '{result}'")
                return result
            else:
                formatted_args_str = ", ".join(format_type_for_display(a) for a in args)
                return f"{origin_name}[{formatted_args_str}]"
        else:
            _log.debug(f"TRACE type_utils.format_type_for_display: Formatted Origin without args ({tp!r}) -> '{origin_name}'")
            return origin_name # e.g., "list" if annotation was just `list`

    # Handle regular types (int, str, custom classes)
    if isinstance(tp, type):
        result = tp.__name__
        _log.debug(f"TRACE type_utils.format_type_for_display: Formatted simple type ({tp!r}) -> '{result}'")
        return result

    # Fallback for other cases (should be rare)
    result = str(tp)
    _log.debug(f"TRACE type_utils.format_type_for_display: Fallback format ({tp!r}) -> '{result}'")
    return result

## ===== OBITUARY UTILITIES ===== ##
def _create_obituary(expected_repr: str, received_repr: str, value: Any, path: List[Union[str, int]], message: Optional[str] = None) -> Obituary:
    """Create an Obituary object containing details about a type check failure.

    Args:
        expected_repr: String representation of the expected type.
        received_repr: String representation of the received type.
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

## ===== BASE TYPE CHECKERS ===== ##
def _check_any(value: Any, expected_type: Any, path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle the `Any` type. Always matches."""
    # No entry log, too simple/frequent
    if expected_type is Any:
        _log.debug(f"TRACE type_utils._check_any: Match (Any).") # Too verbose
        return True, None
    # Return None (not a tuple!) if not handling this type
    return None

def _check_none(value: Any, expected_type: Any, path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `NoneType`."""
    if expected_type is NoneType:
        if value is None:
            return True, None
        else:
            obituary = _create_obituary("NoneType", value, path, "Expected None")
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_none: Fail (Expected None, got {type(value).__name__}). Details: {obituary!r}")
            return False, obituary
    else:
        return None

def _check_optional(
    value: Any,
    expected_type: Any,
    opt_inner: Any,
    globalns: Dict,
    localns: Optional[Dict],
    path: List[Union[str, int]]
) -> Tuple[bool, Optional[Obituary]]:
    """Handle Optional[X] type."""
    # Called directly from check_type if is_optional_type is true.
    # If value is None, it's automatically a match for Optional[X].
    # If value is not None, we recursively check against X.

    # Performance guard: Simple comparison
    if value is None:
        _log.debug(f"TRACE type_utils._check_optional: Value is None, match. Path='{_format_path(path)}'")
        return True, None

    # Value is not None, check against the inner type
    path_repr = _format_path(path) # Cache for logging
    inner_repr = format_type_for_display(opt_inner) # Cache for logging
    _log.debug(f"TRACE type_utils._check_optional: Value is not None. Checking against inner type {inner_repr}. Path='{path_repr}'")

    # Recursive call
    # Performance guard: Recursive call
    match, details_obj = check_type(value, opt_inner, globalns, localns, path) # Renamed details -> details_obj

    if match:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_optional: Inner type check passed. Match. Path='{path_repr}'")
        return True, None
    else:
        # Failure: Report the failure against the *inner* type, but with a specific Optional message.
        # The details_obj from the inner check already contains the relevant info (expected/received repr, value, path).
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_optional: Fail (Inner type check failed). Using inner details: {details_obj!r}. Path='{path_repr}'")
        
        # Create a *new* Obituary with a more specific message, but reuse inner details
        if details_obj:
            fail_msg = f"Value does not match inner type {details_obj.expected_repr} of Optional"
            # Append original reason if present and different
            if details_obj.message and details_obj.message != "Value is not an instance of expected type":
                fail_msg += f" ({details_obj.message})"
            
            # Reuse details from inner failure, but override message
            final_obituary = dataclasses.replace(details_obj, message=fail_msg)
            return False, final_obituary
        else:
            # Should not happen if check_type guarantees an Obituary on failure, but handle defensively
            expected_repr_opt = format_type_for_display(expected_type)
            received_repr = format_type_for_display(type(value))
            fail_msg = f"Value does not match inner type {inner_repr} of Optional"
            return False, _create_obituary(expected_repr_opt, received_repr, value, path, fail_msg)

def _check_callable(value: Any, expected_type: Any, path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `typing.Callable` checks.

    Checks if the value is callable. Warns if the hint includes a nested
    signature (e.g., Callable[[int], str]), as signature validation is not performed.
    """
    # No entry log, simple check
    callable_origin = get_origin(expected_type)
    if callable_origin is collections.abc.Callable or expected_type is collections.abc.Callable:
        if not callable(value):
            # FIX: Use type representation for received value
            received_repr = format_type_for_display(type(value))
            obituary = _create_obituary(
                format_type_for_display(expected_type),
                received_repr, # Use type representation
                value,
                path,
                "Value is not callable"
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_callable: Fail (Not callable). Details: {obituary!r}")
            return False, obituary

        # Check for nested signature and warn if present
        callable_args = get_args(expected_type)
        if callable_args:
            # Only log the warning once per unique callable hint encountered
            @lru_cache(maxsize=32)
            def _log_callable_warning_once(hint_repr):
                warning_msg = (
                    f'DieCast encountered Callable hint with nested signature "{hint_repr}". '
                    f'Runtime validation of nested Callable signatures is not currently implemented. '
                    f'Only checking if the value itself is callable.'
                )
                _log.warning(warning_msg)
            # Performance guard: repr can be complex
            _log_callable_warning_once(repr(expected_type))

            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_callable: Match (Is callable).") # Too verbose
            return True, None # Value is callable, passes basic check

    # Return None (not a tuple!) when not handling this type
    return None

def _check_literal(value: Any, expected_type: Any, path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `typing.Literal[X, Y, ...]` checks."""
    # No entry log, simple check
    origin = get_origin(expected_type)
    if hasattr(typing, 'Literal') and origin is typing.Literal:
        literal_values = get_args(expected_type)
        if value in literal_values:
            return True, None
        else:
            # Value not found in the allowed literals
            # Performance guard: formatting potentially long list of literals
            allowed_repr = "Unknown"
            # FIX: Need to ensure log check doesn't cause errors if repr fails
            try:
                # Ensure allowed_repr is calculated for the message
                allowed_repr = ', '.join(map(repr, literal_values))
            except Exception:
                pass # Ignore potential repr errors in formatting path
            # Construct failure message INCLUDING allowed literals
            fail_msg = f"Value {value!r} of type {type(value).__name__} not in allowed literals: {allowed_repr}"
            # Corrected based on findings: Use type representation for received value
            received_repr = format_type_for_display(type(value))
            expected_repr = format_type_for_display(expected_type)
            obituary = _create_obituary(
                expected_repr,
                received_repr, # Use type representation
                value,
                path,
                fail_msg
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_literal: Fail. Details: {obituary!r}")
            return False, obituary
    # Return None (not a tuple!) when not handling this type
    return None

def _check_final(value: Any, expected_type: Any, globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `typing.Final[X]` checks. Checks against the inner type X."""
    # No entry log, simple check
    if hasattr(typing, 'Final') and get_origin(expected_type) is Final:
        final_args = get_args(expected_type)
        inner_type = final_args[0] if final_args else Any
        if inner_type is Any:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_final: Match (Final[Any]).") # Too verbose
            return True, None # Final[Any] matches anything

        # Recursively check the inner type
        if _log.isEnabledFor(logging.DEBUG):
             _log.debug(f"TRACE type_utils._check_final: Checking inner type {inner_type!r}")
        match, details_obj = check_type(value, inner_type, globalns, localns, path) # Rename

        if not match:
            # Always create a new Obituary for the Final failure
            expected_repr = format_type_for_display(expected_type) # Final[X]
            inner_repr = format_type_for_display(inner_type) # X
            fail_msg = f"Value does not match inner type {inner_repr} of Final"
            # Append inner reason if available
            if details_obj and details_obj.message:
                fail_msg += f": {details_obj.message}"

            # Use the correct _create_obituary signature
            received_repr = format_type_for_display(type(value))
            final_obituary = _create_obituary(
                expected_repr,    # Correct Positional: expected_repr
                received_repr,    # Correct Positional: received_repr
                value,            # Correct Positional: value
                path,             # Correct Positional: path
                fail_msg          # Correct Positional: message
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_final: Fail (Inner check failed). Details: {final_obituary!r}")
            return False, final_obituary

        # Inner type matched
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_final: Match (Inner check passed).") # Too verbose
        return True, None

    # Not a Final type, return None.
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug("TRACE type_utils._check_final: Not a Final type. Returning None as unhandled.")
    return None # Reverted: Indicate not handled

def _check_newtype(value: Any, expected_type: Any, globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `NewType('Name', BaseType)` checks. Checks against the BaseType."""
    # No entry log, simple check
    if hasattr(expected_type, '__supertype__'):  # NewType instances have __supertype__
        supertype = expected_type.__supertype__
        if _log.isEnabledFor(logging.DEBUG):
             _log.debug(f"TRACE type_utils._check_newtype: Found NewType. Checking supertype {supertype!r}")
        # Recursively check the supertype
        match, details_obj = check_type(value, supertype, globalns, localns, path) # Rename
        if not match:
            # Always create a new Obituary for the NewType failure
            newtype_name = getattr(expected_type, '__name__', 'NewType')
            supertype_repr = format_type_for_display(supertype)
            expected_repr = format_type_for_display(expected_type) # NewType representation
            fail_msg = f"Value does not match supertype {supertype_repr} of {newtype_name}"
            # Append inner reason if available
            if details_obj and details_obj.message:
                fail_msg += f": {details_obj.message}"

            # Use the correct _create_obituary signature
            received_repr = format_type_for_display(type(value))
            final_obituary = _create_obituary(
                expected_repr,
                received_repr,
                value,
                path,
                fail_msg
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_newtype: Fail (Supertype check failed). Details: {final_obituary!r}")
            return False, final_obituary

        # Supertype matched
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_newtype: Match (Supertype check passed).") # Too verbose
        return True, None
    else:
        # Not a NewType, return None.
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_newtype: Not a NewType. Returning None as unhandled.")
        return None # Reverted: Indicate not handled

def _check_protocol(value: Any, expected_type: Any, path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `typing.Protocol` checks using `isinstance` if possible, or structural checks."""
    # Check requires Python 3.8+ where Protocol is properly defined
    if not hasattr(typing, 'Protocol') or not isinstance(expected_type, type) or not issubclass(expected_type, typing.Protocol):
         return None # Return None if not a Protocol or not supported version

    # Performance guard: format_type_for_display
    expected_repr = "Protocol" # Default
    if _log.isEnabledFor(logging.DEBUG):
        expected_repr = format_type_for_display(expected_type)
        _log.debug(f"TRACE type_utils._check_protocol: Checking value {value!r} against protocol {expected_repr}")

    # Lazily create fail details
    fail_obituary = lambda msg: _create_obituary(expected_repr, value, path, msg)

    # If the protocol is runtime_checkable, isinstance should work
    is_runtime = getattr(expected_type, '_is_runtime_protocol', False)
    if is_runtime:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_protocol: Protocol {expected_repr} is runtime_checkable. Using isinstance.")
        try:
            # Performance guard: isinstance can be slow for protocols
            if isinstance(value, expected_type):
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_protocol: Match (isinstance passed for runtime protocol).")
                return True, None
            else:
                # Optional: Add detailed structural check here even for runtime_checkable if isinstance fails?
                # For now, trust isinstance for runtime_checkable.
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_protocol: Fail (isinstance failed for runtime protocol).")
                return False, fail_obituary("Value does not match runtime checkable protocol structure")
        except TypeError as e:
            # isinstance might fail in weird edge cases, fallback to structural check
            _log.warning(f"TRACE type_utils._check_protocol: TypeError during isinstance check for runtime protocol {expected_repr}: {e}. Falling back to structural check.")
            pass
    else:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_protocol: Protocol {expected_repr} is not runtime_checkable. Using structural check.")

    # Fallback: Structural check for non-runtime_checkable protocols
    # This is a basic check and might miss complex method signature compatibility
    try:
        # Performance guard: get_resolved_type_hints can be slow
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_protocol: Getting protocol members for {expected_repr}")
        protocol_members = get_resolved_type_hints(expected_type, include_extras=True)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_protocol: Protocol members: {protocol_members!r}")
        if not protocol_members: # If no members defined, any object technically conforms
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_protocol: Protocol {expected_repr} has no members defined. Match.")
                # However, require Protocol definition itself as minimal check
            # This was already checked at the start, so we can return True
            return True, None

        missing_members = []
        for member_name in protocol_members:
            if not hasattr(value, member_name):
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_protocol: Structural check fail: Missing member '{member_name}'")
                missing_members.append(member_name)
            # Future enhancement: Could add callable checks or signature comparisons here

        if not missing_members:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_protocol: Match (All required members found).")
            return True, None # All required members found
        else:
            fail_msg = f"Missing required attributes/methods: {', '.join(map(repr, missing_members))}"
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_protocol: Fail ({fail_msg}).")
            return False, fail_obituary(fail_msg)

    except Exception as e:
        if _log.isEnabledFor(logging.DEBUG):
            _log.warning(f"Error during structural protocol check for {expected_type!r}: {e}", exc_info=True)
        return False, fail_obituary("Protocol check failed due to internal error")

def _check_union(value: Any, expected_type: Any, globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `typing.Union` checks (including Optionals handled by caller)."""
    func_id = localns.get("_func_id", "unknown") if localns else "unknown"
    is_union, union_args = is_union_type(expected_type)
    if not is_union:
        return None # Not handling this type

    # Always calculate the full representation for use in failure messages/obituary
    expected_repr = format_type_for_display(expected_type)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Entering. Value={value!r}, Expected={expected_repr}, Args={union_args!r}, Path='{_format_path(path)}'")

    # Check value against each member type in the union
    member_match_found = False
    for member_type in union_args:
        member_repr = "Unknown" # Default
        if _log.isEnabledFor(logging.DEBUG):
            member_repr = format_type_for_display(member_type)
            _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Checking against member: {member_repr}")

        if member_type is NoneType:
            if value is None:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Match found for member {member_repr} (None).")
                member_match_found = True
                break # Found a match
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Skipping NoneType member, value is not None.")
                continue # Value isn't None, try next member type

        # Check non-None member types recursively
        # Performance guard: Recursive call can be expensive
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Recursive call: check_type({value!r}, {member_repr}, ...)")
        match, details_obj = check_type(value, member_type, globalns, localns, path) # Ignore details from inner check for now

        # --- ADDED: Detailed log after recursive call --- #
        _log.info(f"!!! _check_union LOOP: Result for Member={member_repr}, Value={value!r} -> Match={match}, Details={details_obj!r}")
        # --- END ADDITION --- #

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Recursive result for {member_repr}: match={match}")

        if match:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Match found for member {member_repr}.")
            member_match_found = True
            break # Found a match
        else:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: No match for member {member_repr}.")

    # Final result determination
    if member_match_found:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Exiting. Result: (True, None)")
        return True, None # Correct: Return tuple
    else:
        # No match found after checking all members
        # Use the expected_repr calculated at the start of the function
        fail_msg = f"Value does not match any type in {expected_repr}" 
        # Calculate received_repr for the Obituary
        received_repr = format_type_for_display(type(value))
        final_obituary = _create_obituary(
            expected_repr, # Use the fully formatted Union repr
            received_repr, # Pass the calculated received repr
            value, 
            path, 
            fail_msg,
        ) 
        # --- ADDED: Explicit failure return log --- #
        _log.info(f"!!! _check_union FAILURE: Returning (False, {final_obituary!r}) for expected={expected_repr}, value={value!r}, path='{_format_path(path)}'")
        # --- END ADDITION --- #
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_union [ID:{func_id}]: Exiting. Result: (False, ...details...). Details: {final_obituary!r}")
        return False, final_obituary # Correct: Return tuple

def _check_typevar(value: Any, expected_type: Any, globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `typing.TypeVar` checks, including constraints, bounds, and consistency."""
    # Requires localns to contain _func_id for consistency tracking
    if not isinstance(expected_type, TypeVar):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_typevar: Not a TypeVar. Returning None as unhandled.")
        return None # Reverted: Indicate not handled

    func_id = localns.get('_func_id') if localns else None
    if func_id is None:
        _log.warning(f"DieCast: TypeVar check for {expected_type!r} skipped: Missing function context ('_func_id') in local namespace. Path='{_format_path(path)}'")
        # Treat as match if context is missing? Or fail? For now, maybe match to avoid breaking things.
        return True, None # Original behavior: Warn and allow

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_typevar [ID:{func_id}]: Entering. Value={value!r}, Expected={expected_type!r} ('{expected_type.__name__}'), Path='{_format_path(path)}'")

    constraints = expected_type.__constraints__
    bound = expected_type.__bound__

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_typevar [ID:{func_id}]: Constraints={constraints!r}, Bound={bound!r}")

    # 1. Check against constraints if they exist
    if constraints:
        match_found_in_constraints = False
        constraint_details_list = []
        for constraint in constraints:
            constraint_match, constraint_details = check_type(value, constraint, globalns, localns, path)
            if constraint_match:
                match_found_in_constraints = True
                break
            constraint_details_list.append(constraint_details) # Collect details even on match failure

        if not match_found_in_constraints:
            fail_msg = "Value not in allowed types for constrained TypeVar"
            # --- FIX: Ensure expected_repr uses the TypeVar itself --- 
            final_expected_repr = format_type_for_display(expected_type) # <-- Correctly use the TypeVar
            # Try to provide details from the *first* constraint failure if available
            first_details = constraint_details_list[0] if constraint_details_list else None
            # Use the *value's* actual type for received_repr if no inner details
            final_received_repr = first_details.received_repr if first_details else format_type_for_display(type(value))
            final_obituary = _create_obituary(
                expected_repr=final_expected_repr, 
                received_repr=final_received_repr, 
                value=(first_details.value if first_details else value),
                path=(first_details.path if first_details else path),
                message=fail_msg
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_typevar: Fail (Constraint mismatch). Details: {final_obituary!r}. Exiting.")
            return False, final_obituary

    # 2. Check against bound if it exists (and no constraints, or constraints passed)
    if bound:
        bound_repr = format_type_for_display(bound)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_typevar [ID:{func_id}]: Checking against bound: {bound_repr}")
        # Recursively check if the value is compatible with the bound type
        bound_match, bound_details = check_type(value, bound, globalns, localns, path) # Check against the bound
        if not bound_match:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_typevar [ID:{func_id}]: Fail (Bound check failed). Details: {bound_details!r}")
            # FIX: Use the inner message directly if available for better detail.
            # --- MODIFIED: Prioritize specific bound message --- 
            bound_repr = format_type_for_display(bound) # Ensure bound_repr is calculated
            fail_msg = f"Value does not meet bound type {bound_repr}" # Specific bound message
            if bound_details and bound_details.message and bound_details.message != "Value is not an instance of expected type":
                # Append inner reason if it's more specific than the generic isinstance failure
                fail_msg += f": {bound_details.message}"
            # --- END MODIFICATION ---

            # Create the final obituary, preserving details if possible
            received_repr = format_type_for_display(type(value))
            final_obituary = _create_obituary(
                format_type_for_display(expected_type), # Use TypeVar's repr as expected
                received_repr,                          # received_repr (positional)
                value,                                  # value (positional)
                path,                                   # path (positional)
                fail_msg                                # message (positional)
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_typevar: Fail (Mismatch found for value). Details: {final_obituary!r}. Exiting.")
            return False, final_obituary
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_typevar [ID:{func_id}]: Bound check passed.")

    # 3. Consistency Check (if bound/constraints passed or absent)
    existing_binding = get_typevar_binding(func_id, expected_type)
    value_type = type(value)
    # value_type_repr = format_type_for_display(value_type)

    if existing_binding is not None:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_typevar [ID:{func_id}]: Existing binding found: {existing_binding!r}")
        # Check if current value's type matches the established binding
        # Use check_type for consistency, handles complex types
        consistency_match, consistency_details = check_type(value, existing_binding, globalns, localns, path)

        if not consistency_match:
            # Consistency failure
            existing_binding_repr = format_type_for_display(existing_binding)
            received_repr = format_type_for_display(type(value)) # Need received type repr
            # Corrected fail_msg to include TypeVar repr (~T)
            fail_msg = f"TypeVar consistency violation: Expected {format_type_for_display(expected_type)} (Bound to: {existing_binding_repr}) but received {received_repr}"
            # --- ADDED: Include bound info if applicable --- #
            if bound:
                 bound_repr = format_type_for_display(bound)
                 fail_msg += f" (Bound to: {bound_repr})"
            # --- END ADDITION --- #
            # Create final obituary - use details from the failed consistency check if available
            # Ensure received_repr is calculated (done above)
            # received_repr = format_type_for_display(type(value))
            if consistency_details:
                final_obituary = consistency_details # Use the detailed inner failure
                # Update the message to reflect the overarching consistency failure
                final_obituary = dataclasses.replace(final_obituary, message=fail_msg)
                # <<< ADDED LINE BELOW >>>
                final_obituary = dataclasses.replace(final_obituary, expected_repr=format_type_for_display(expected_type))
            else:
                # Create a new obituary if no inner details were available
                final_obituary = _create_obituary(
                    format_type_for_display(expected_type), # Use TypeVar's repr as expected
                    received_repr,                          # received_repr (positional)
                    value,                                  # value (positional)
                    path,                                   # path (positional)
                    fail_msg                                # message (positional)
                )

            # If we used details from consistency check, update the message
            # This is now handled above when setting final_obituary
            # if consistency_details:
            #    final_obituary = dataclasses.replace(final_obituary, message=fail_msg)

            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_typevar: Fail (Consistency check failed). Details: {final_obituary!r}")
            return False, final_obituary
        else:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_typevar [ID:{func_id}]: Match (Consistency check passed).")
            return True, None # Consistent with existing binding
    else:
        # No existing binding, establish one
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_typevar [ID:{func_id}]: No existing binding. Binding {expected_type!r} to {value_type!r}.")
        bind_typevar(func_id, expected_type, value_type)
        return True, None # First encounter, implicitly consistent

def _check_forward_ref(
    value: Any,
    expected_type: Any,
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    path: List[Union[str, int]],
) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Handle `typing.ForwardRef` checks."""
    if not isinstance(expected_type, ForwardRef):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_forward_ref: Not a ForwardRef. Returning None as unhandled.")
        return None # Reverted: Indicate not handled

    # Performance guard: format_type_for_display
    expected_repr = "ForwardRef" # Default
    path_repr = "UnknownPath" # Default
    if _log.isEnabledFor(logging.DEBUG):
        expected_repr = format_type_for_display(expected_type)
        path_repr = _format_path(path)
        _log.debug(f"TRACE type_utils._check_forward_ref: Entering. Value={value!r}, Expected={expected_repr}, Path='{path_repr}'")

    type_name = expected_type.__forward_arg__
    resolved_type = None
    # Lazily create fail details using the original ref string representation
    received_repr = format_type_for_display(type(value))
    fail_obituary_func = lambda msg: _create_obituary(
        type_name,      # expected_repr (original ref string)
        received_repr,  # received_repr
        value,          # value
        path,           # path
        msg             # message
    )

    try:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_forward_ref: Resolving '{type_name}'...")
        # Performance guard: Resolution can be slow
        resolved_type = resolve_forward_ref(expected_type, globalns, localns)
        resolved_repr = format_type_for_display(resolved_type) # Cache for logging
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_forward_ref: Resolved '{type_name}' -> {resolved_repr}")

        # Recurse with the resolved type
        # Performance guard: Recursive call
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_forward_ref: Recursive call: check_type({value!r}, {resolved_repr}, ...)")
        # Pass the *original* globalns, localns, path to the recursive call
        match, details_obj = check_type( # Rename
            value=value,
            expected_type=resolved_type,
            globalns=globalns,
            localns=localns,
            path=path,
            # func_id is implicitly handled by localns within check_type if needed
        )
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_forward_ref: Recursive check returned: match={match}")
        if match:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_forward_ref: Match (Recursive check passed). Exiting.")
                return True, None
        else:
             # Always create a new Obituary for the ForwardRef failure
             # Use the original ref name (e.g., 'MyClass') as the expected representation
             fail_msg = f"Value does not match resolved forward reference type '{type_name}' (resolved to {resolved_repr})"
             # Append inner reason if available
             if details_obj and details_obj.message:
                 fail_msg += f": {details_obj.message}"

             # Corrected call to use positional args
             # Ensure received_repr is calculated (already done above)
             final_obituary = _create_obituary(
                 type_name,      # expected_repr (original ref string)
                 received_repr,  # received_repr
                 value,          # value
                 path,           # path
                 fail_msg        # message
             )
             if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_forward_ref: Fail (Recursive check failed). Details: {final_obituary!r}. Exiting.")
             return False, final_obituary

    except NameError as e:
        # If resolution fails, treat it as a type check failure
        err_msg = f"Could not resolve forward reference '{type_name}': {e}"
        _log.warning(f"TRACE type_utils._check_forward_ref: Fail ({err_msg}). Exiting.")
        return False, fail_obituary_func(err_msg)
    except Exception as e:
        err_msg = f"Unexpected error resolving forward reference '{type_name}': {e!r}"
        _log.error(f"TRACE type_utils._check_forward_ref: Fail ({err_msg}). Exiting.", exc_info=True)
        return False, fail_obituary_func(err_msg)

def _check_generic_alias(value: Any, expected_type: Any, globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Optional[Tuple[bool, Optional[Obituary]]]:
    """Dispatcher for generic alias types (List, Dict, Tuple, etc.)."""
    # func_id = localns.get("_func_id", "unknown") if localns else "unknown"
    # Get origin and args without resolving inner types yet
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    expected_repr = format_type_for_display(expected_type) # Cache for logging
    path_repr = _format_path(path) # Cache for logging

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_generic_alias: Entering. Value=<{type(value).__name__}>, Expected={expected_repr}, Path='{path_repr}'")

    # If it's not a generic alias based on _GenericAlias or Generic, pass to other handlers
    # Add check for hasattr(expected_type, '__origin__') for robustness
    is_gen_alias = isinstance(expected_type, typing._GenericAlias) if hasattr(typing, '_GenericAlias') else False
    is_typing_generic = hasattr(expected_type, '__origin__') and hasattr(expected_type, '__args__')

    if not (is_gen_alias or is_typing_generic):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: Not a generic alias ({expected_repr}). Returning None as unhandled.")
        return None

    # Basic check: Value must not be None if the type hint is a generic alias
    if value is None:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: Fail (Value is None, expected generic {expected_repr}).")
        fail_obituary = _create_obituary(expected_repr, value, path, "Value is None but expected a generic container type")
        return False, fail_obituary

    # Helper for creating failure obituary within this function
    def fail_obituary(message: str) -> Obituary:
        return _create_obituary(expected_repr, value, path, message)

    # Check if value type matches expected origin BEFORE dispatch using optimized check
    if not is_instance_optimized(value, origin):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: Fail (Value type {type(value).__name__} does not match expected origin {origin.__name__} using is_instance_optimized).")
        # Use the helper function defined earlier in _check_generic_alias
        origin_fail_obituary = fail_obituary(f"Value type {type(value).__name__} is not compatible with expected container origin {origin.__name__}")
        return False, origin_fail_obituary

    # Delegate to specific handlers based on the origin type
    if origin is None:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: No origin type found for expected type {expected_repr}. Skipping origin check.")
        return None

    if origin is tuple:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: Dispatching to _check_generic_tuple.")
        result_match, result_obituary = _check_generic_tuple(value, args, globalns, localns, path)
        return result_match, result_obituary
    elif is_instance_optimized(value, collections.abc.Mapping):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: Dispatching to _check_generic_mapping.")
        result_match, result_obituary = _check_generic_mapping(value, args, globalns, localns, path)
        return result_match, result_obituary
    elif is_instance_optimized(value, collections.abc.Sequence):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: Dispatching to _check_generic_sequence.")
        result_match, result_obituary = _check_generic_sequence(value, args, globalns, localns, path)
        return result_match, result_obituary
    elif is_instance_optimized(value, collections.abc.Set):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: Dispatching to _check_generic_set.")
        result_match, result_obituary = _check_generic_set(value, args, globalns, localns, path)
        return result_match, result_obituary
    elif origin in (
        collections.abc.AsyncGenerator,
        collections.abc.AsyncIterable,
        collections.abc.AsyncIterator, 
        collections.abc.Generator,
        collections.abc.Iterator, 
        collections.abc.Iterable
        ):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_alias: Found Generator/Iterator/Iterable type {expected_repr}. Match based on origin. Exiting.")
        return True, None

    _log.warning(f"TRACE type_utils._check_generic_alias: Unhandled generic alias type: {expected_repr}. Failing check.")
    return False, fail_obituary("Unsupported generic type structure")

def _check_simple_type(value: Any, expected_type: Any, path: List[Union[str, int]]) -> Tuple[bool, Optional[Obituary]]:
    """Check non-generic types using optimized isinstance."""
    path_repr = _format_path(path) # Cache for logging
    expected_type_repr = format_type_for_display(expected_type) # Cache for logging
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_simple_type: Entering. ValueType={type(value).__name__}, ExpectedType={expected_type_repr}, Path='{path_repr}'")

    try:
        # Handle cases like typing.List, typing.Dict directly
        origin = get_origin(expected_type)
        check_target = origin if origin else expected_type

        # Ensure check_target is a type or tuple of types for isinstance
        if not isinstance(check_target, (type, tuple)):
             # This might happen for things that aren't classes/types like TypeVars without bounds
             # In simple check, if it's not a type, it's likely a mismatch unless origin helped
             if origin is None:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_simple_type: Fail (Expected type is not a class/tuple and has no origin). Path='{path_repr}'")
                    obituary = _create_obituary(expected_type_repr, value, path, "Expected type is not a class or tuple of classes")
                    return False, obituary
             # If origin exists and isn't a type/tuple, that's also unusual here
             elif not isinstance(origin, (type, tuple)):
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_simple_type: Fail (Origin type is not a class/tuple). Path='{path_repr}'")
                    obituary = _create_obituary(expected_type_repr, value, path, "Origin of expected type is not a class or tuple of classes")
                    return False, obituary
             # else: check_target should be valid type(s) from origin

        if expected_type is int and isinstance(value, bool):
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_simple_type: Fail (bool value rejected for int hint). Path='{path_repr}'")
                obituary = _create_obituary(expected_type_repr, value, path, "Value is bool, expected int")
                return False, obituary

        # Use the potentially simplified check_target (e.g., list for List[int])
        is_match = isinstance(value, check_target)

        if is_match:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_simple_type: Match (isinstance check passed). Path='{path_repr}'")
                return True, None
        else:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_simple_type: Fail (isinstance check failed). Path='{path_repr}'")
            # Calculate received_repr and use correct _create_obituary signature
            received_repr = format_type_for_display(type(value))
            obituary = _create_obituary(
                expected_repr=expected_type_repr,
                received_repr=received_repr,
                value=value,
                path=path,
                message="Value is not an instance of expected type"
            )
            return False, obituary

    except TypeError as e:
        # isinstance can raise TypeError for non-class second arguments
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_simple_type: Fail (TypeError during isinstance: {e}). Path='{path_repr}'")
        # Calculate received_repr and use correct _create_obituary signature
        received_repr = format_type_for_display(type(value))
        obituary = _create_obituary(
            expected_repr=expected_type_repr,
            received_repr=received_repr,
            value=value,
            path=path,
            message=f"TypeError during type check: {e}"
        )
        return False, obituary

## ===== GENERIC TYPE CHECKERS ===== ##
def _check_generic_tuple(value: Tuple, args: Tuple[Any, ...], globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Tuple[bool, Optional[Obituary]]:
    """Handle `Tuple[X, Y, ...]` or `Tuple[X, ...]` checks."""
    if not args: 
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_generic_tuple: No args provided for tuple check. Match.")
        return True, None # Tuple without args (plain tuple)

    path_repr = _format_path(path)
    # FIX: Define expected_repr before length check
    expected_repr = format_type_for_display(Tuple[args])
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_generic_tuple: Entering. Value={value!r}, Args={args!r}, Path='{path_repr}'")

    # Handle variable-length tuples (e.g., Tuple[int, ...])
    is_variable_tuple = len(args) == 2 and args[1] is Ellipsis
    if is_variable_tuple:
        element_type = args[0]
        element_type_repr = format_type_for_display(element_type) # Cache for logging
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_tuple: Handling variable-length tuple. Element type={element_type_repr}")
        if element_type is Any:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("TRACE type_utils._check_generic_tuple: Element type is Any. Match.")
            return True, None

        for index, item in enumerate(value):
            item_path = path + [index]
            item_path_repr = _format_path(item_path) # Cache for logging
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_generic_tuple: Variable tuple loop [{index}]: Checking item={item!r}, Path='{item_path_repr}'")
            # Performance guard: Recursive call
            match, details_obj = check_type(item, element_type, globalns, localns, item_path) # Rename
            if not match:
                # --- MODIFIED: Create container obituary, use inner path --- #
                fail_msg = f"Incorrect element type at index {index} in variable-length tuple"

                final_expected_repr = details_obj.expected_repr if details_obj else element_type_repr
                final_received_repr = details_obj.received_repr if details_obj else format_type_for_display(type(item))
                final_received_value = details_obj.value if details_obj else item
                # Use path from inner details if available, otherwise use current item_path
                final_path = details_obj.path if details_obj else item_path

                final_obituary = _create_obituary(
                    expected_repr=final_expected_repr,
                    received_repr=final_received_repr,
                    value=final_received_value,
                    path=final_path, # Use potentially deeper path
                    message=details_obj.message if details_obj and details_obj.message else fail_msg # PRIORITIZE INNER MESSAGE
                )
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_generic_tuple: Fail (Variable tuple mismatch at index {index}). Details: {final_obituary!r}. Exiting.")
                return False, final_obituary
                # --- END MODIFICATION --- #
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_generic_tuple: Variable tuple loop [{index}]: Match.")

            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("TRACE type_utils._check_generic_tuple: Match (All variable tuple elements matched). Exiting.")
        return True, None

    # Handle fixed-length tuples (e.g., Tuple[int, str])
    _log.debug("TRACE type_utils._check_generic_tuple: Handling fixed-length tuple.")
    if len(value) != len(args):
        fail_msg = f"Expected {len(args)} elements, got {len(value)}"
        # FIX: Correct arguments for _create_obituary
        received_repr = format_type_for_display(type(value))
        final_obituary = _create_obituary(
            expected_repr,  # Correct arg
            received_repr,  # Correct arg (was missing)
            value,          # Correct arg
            path,           # Correct arg
            fail_msg        # Correct arg
        )
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_tuple: Fail (Length mismatch). Details: {final_obituary!r}. Exiting.")
        return False, final_obituary

    for index, (item, element_type) in enumerate(zip(value, args)):
        item_path = path + [index]
        item_path_repr = _format_path(item_path) # Cache for logging
        element_type_repr = format_type_for_display(element_type) # Cache for logging
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_tuple: Fixed tuple loop [{index}]: Checking item={item!r} against {element_type_repr}, Path='{item_path_repr}'")
        if element_type is Any: 
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("TRACE type_utils._check_generic_tuple: Fixed tuple loop [{index}]: Skipping Any.")
            continue
            
        # Performance guard: Recursive call
        match, details_obj = check_type(item, element_type, globalns, localns, item_path) # Rename
        if not match:
            # --- MODIFIED: Create container obituary, use inner path ---
            fail_msg = f"Incorrect element type at index {index} in fixed-length tuple"
            # Append inner reason? (Optional enhancement)
            # if details_obj and details_obj.message and "Value is not an instance" not in details_obj.message:
            #      fail_msg += f": {details_obj.message}"

            final_expected_repr = details_obj.expected_repr if details_obj else format_type_for_display(element_type) # Use specific element_type here
            final_received_repr = details_obj.received_repr if details_obj else format_type_for_display(type(item))
            final_received_value = details_obj.value if details_obj else item
            # Use path from inner details if available, otherwise use current item_path
            final_path = details_obj.path if details_obj else item_path

            final_obituary = _create_obituary(
                expected_repr=final_expected_repr,
                received_repr=final_received_repr,
                value=final_received_value,
                path=final_path, # Use potentially deeper path
                message=details_obj.message if details_obj and details_obj.message else fail_msg # PRIORITIZE INNER MESSAGE
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_generic_tuple: Fail (Fixed tuple mismatch at index {index}). Details: {final_obituary!r}. Exiting.")
            return False, final_obituary
            # --- END MODIFICATION ---
        else:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_generic_tuple: Fixed tuple loop [{index}]: Match.")

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug("TRACE type_utils._check_generic_tuple: Match (All fixed tuple elements matched). Exiting.")
    return True, None

def _check_generic_sequence(value: Sequence, args: Tuple[Any, ...], globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Tuple[bool, Optional[Obituary]]:
    """Handle `Sequence[T]` checks (applies to list, tuple, etc.)."""
    if not args: 
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_generic_sequence: No args provided for sequence check. Match.")
        return True, None # No specific element type to check (e.g., list)
        
    element_type = args[0]
    element_type_repr = format_type_for_display(element_type) # Cache for logging
    path_repr = _format_path(path) # Cache for logging
    
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_generic_sequence: Entering. Value=<{type(value).__name__} len={len(value)}>, ElementType={element_type_repr}, Path='{path_repr}'")
    
    if element_type is Any: 
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_generic_sequence: Element type is Any. Match. Exiting.")
        return True, None # Sequence[Any]

    for index, item in enumerate(value):
        item_path = path + [index]
        item_path_repr = _format_path(item_path)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"!!! _check_generic_sequence LOOP [{index}]: PRE-CHECK item={item!r} (type={type(item).__name__}), ExpectedElement={element_type_repr}, Path='{item_path_repr}'")
            _log.debug(f"TRACE type_utils._check_generic_sequence: Loop [{index}]: Checking item={item!r}, Path='{item_path_repr}'")

        match, details_obj = check_type(item, element_type, globalns, localns, item_path)
        if not match:
            if _log.isEnabledFor(logging.DEBUG):
                _log.info(f"!!! _check_generic_sequence FAILURE BLOCK ENTERED: Index={index}, Item={item!r}, Match={match}, Details={details_obj!r}")
                # Return the inner failure details directly
                _log.debug(f"TRACE type_utils._check_generic_sequence: Fail (Mismatch at index {index}). Returning inner details: {details_obj!r}. Exiting.")
            return False, details_obj
            
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_sequence: Match (All elements matched). Exiting.")
        # _log.info(f"!!! _check_generic_sequence RETURNING: (True, None) for path '{_format_path(path)}'")
        return True, None

def _check_generic_mapping(value: Mapping, args: Tuple[Any, ...], globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Tuple[bool, Optional[Obituary]]:
    """Handle `Mapping[K, V]` checks (applies to dict)."""
    if not args or len(args) != 2: 
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_generic_mapping: Invalid/missing args for mapping check. Match (no checks performed).")
        return True, None # Cannot check without KeyType and ValueType
        
    key_type, value_type = args
    key_type_repr = format_type_for_display(key_type) # Cache for logging
    value_type_repr = format_type_for_display(value_type) # Cache for logging
    path_repr = _format_path(path) # Cache for logging

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_generic_mapping: Entering. Value=<{type(value).__name__} len={len(value)}>, KeyType={key_type_repr}, ValueType={value_type_repr}, Path='{path_repr}'")

    # Optimization: If both key and value types are Any, no need to iterate
    if key_type is Any and value_type is Any: 
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_generic_mapping: KeyType and ValueType are Any. Match. Exiting.")
        return True, None

    for key, item in value.items():
        # key_path = path + [repr(key)] # Use repr(key) for path segment - Incorrect
        key_path = path + [key]
        key_path_repr = _format_path(key_path)

        # Check key type
        if key_type is not Any:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_generic_mapping: Checking Key {key!r} against {key_type_repr}, Path='{key_path_repr}'")
            # Performance guard: Recursive call
            match, inner_details_obj = check_type(key, key_type, globalns, localns, key_path) 
            if not match:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_generic_mapping: Fail (Key mismatch for key {key!r}). Creating specific key failure obituary.")
                    key_fail_obituary = _create_obituary(
                    expected_repr=inner_details_obj.expected_repr if inner_details_obj else format_type_for_display(key_type),
                    received_repr=inner_details_obj.received_repr if inner_details_obj else format_type_for_display(type(key)),
                    value=key,
                    path=key_path,
                    message=f"Incorrect type for key {key!r}"
                )
                return False, key_fail_obituary
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_generic_mapping: Key {key!r} matched.")

        # Check value type
        if value_type is not Any:
            value_path = path + [f'value({key!r})']
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_generic_mapping: Checking Value {item!r} for key {key!r} against {value_type_repr}, Path='{_format_path(value_path)}'")
            match, details_obj = check_type(item, value_type, globalns, localns, value_path)
            if not match:
                # Return the inner failure details directly
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_generic_mapping: Fail (Value mismatch for key {key!r}). Returning inner details: {details_obj!r}. Exiting.")
                return False, details_obj
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE type_utils._check_generic_mapping: Value {item!r} for key {key!r} matched.")
                    _log.debug(f"TRACE type_utils._check_generic_mapping: Finished checks for key {key!r}.")

    if _log.isEnabledFor(logging.DEBUG):    
        _log.debug("TRACE type_utils._check_generic_mapping: Match (All keys/values matched). Exiting.")
    return True, None

def _check_generic_set(value: Set, args: Tuple[Any, ...], globalns: Dict, localns: Optional[Dict], path: List[Union[str, int]]) -> Tuple[bool, Optional[Obituary]]:
    """Handle `AbstractSet[T]` or `Set[T]` checks."""
    if not args: 
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_generic_set: No args provided for set check. Match.")
        return True, None # No specific element type to check (e.g., set)
        
    element_type = args[0]
    element_type_repr = format_type_for_display(element_type) # Cache for logging
    path_repr = _format_path(path) # Cache for logging

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE type_utils._check_generic_set: Entering. Value=<{type(value).__name__} len={len(value)}>, ElementType={element_type_repr}, Path='{path_repr}'")

    if element_type is Any: 
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("TRACE type_utils._check_generic_set: Element type is Any. Match. Exiting.")
        return True, None # Set[Any]

    for item in value:
        # Path for set elements is tricky, use repr for uniqueness if possible
        try:
             item_repr_short = repr(item)
             if len(item_repr_short) > 30:
                 item_repr_short = item_repr_short[:27] + "..."
        except Exception:
             item_repr_short = hex(id(item)) # Fallback if repr fails

        # We still *pass* the specific item path to the recursive check for context,
        # but we might not use it in the *final* error message if the whole set fails.
        item_path = path + [f"elem({item_repr_short})"] # Indicate path is for a set element
        item_path_repr = _format_path(item_path) # Cache for logging
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE type_utils._check_generic_set: Checking item={item!r}, Path='{item_path_repr}'")

        # Performance guard: Recursive call
        match, details_obj = check_type(item, element_type, globalns, localns, item_path)

        if not match:
            if details_obj and details_obj.message:
                fail_msg = details_obj.message
            else:
                fail_msg = f"Incorrect element type found in set"

            # Prioritize inner expected_repr
            final_expected_repr = details_obj.expected_repr if details_obj else element_type_repr
            final_received_repr = details_obj.received_repr if details_obj else format_type_for_display(type(item))
            final_received_value = details_obj.value if details_obj else item

            # Note: We use the *element_type_repr* and the specific *item*
            # but we use the original *set's path* for the final obituary,
            # as the specific element's path within the set is less useful.
            # UPDATE: Now using inner details, but still use set's path for consistency?
            # Let's use the inner check's path (item_path) for detail, but keep set's path for overall message?
            # For now, let's use the inner details completely, including item_path.
            final_obituary = _create_obituary(
                expected_repr=final_expected_repr,
                received_repr=final_received_repr,
                value=final_received_value,
                path=item_path, # Use the specific item's path
                message=fail_msg
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_generic_set: Fail (Mismatch found for item {item!r}). Details: {final_obituary!r}. Exiting.")
            return False, final_obituary
        else:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE type_utils._check_generic_set: Item {item!r} matched.")

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug("TRACE type_utils._check_generic_set: Match (All elements matched). Exiting.")
    return True, None

## ===== MAIN CHECK FUNCTION ===== ##
_3_ARG_HANDLERS: Tuple[TypeCheckHandler3Args, ...] = (
    _check_any,
    _check_none,
    _check_callable,
    _check_literal,
    _check_protocol
)
_TYPE_HANDLERS_5ARGS: list[TypeCheckHandler5Args] = [
    _check_union,
    _check_typevar,
    _check_final,
    _check_newtype,
    _check_forward_ref,
    _check_generic_alias
]

def check_type(
    value: Any,
    expected_type: Any,
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
    path: Optional[List[Union[str, int]]] = None,
    func_id: Optional[str] = None,
    raise_error: bool = False,
) -> Tuple[bool, Optional[Obituary]]:
    """Main recursive function to check if a value matches an expected type annotation.

    Recursively handles various type constructs like Generics, Unions, Optionals,
    TypeVars, ForwardRefs, etc.

    Args:
        value: The value to check.
        expected_type: The type hint to check against.
        globalns: Global namespace for resolving ForwardRefs.
        localns: Local namespace for resolving ForwardRefs and tracking TypeVars.
                 Must include `_func_id` key for TypeVar consistency checks.
        path: Internal list tracking the path within nested data structures.
        func_id: Optional unique ID for TypeVar consistency tracking within a call.
                 If None and `localns` is provided, attempts to get from `localns['_func_id']`.
        raise_error: If True, raises YouDiedError immediately on failure.

    Returns:
        Tuple[bool, Optional[Obituary]]: (True, None) if match,
                                       (False, obituary_object) if mismatch.
        The Obituary object contains structured details about the failure.
    """
    # Ensure path is initialized for recursive calls and error reporting
    path = path if path is not None else []
    path_tuple = tuple(path) if path is not None else None # Use tuple for cache key

    # --- BEGIN Annotated Handling (Must be early!) ---
    # Check if Annotated is available and expected_type uses it
    if Annotated is not None:
        _origin = get_origin(expected_type)
        if _origin is Annotated:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE DETECTED Annotated: Path='{_format_path(path)}', Type={format_type_for_display(expected_type)} ---")
            _args = get_args(expected_type)
            if not _args:
                # This case is likely invalid Annotated usage, e.g. Annotated[]
                _log.warning(f"Annotated type has no arguments: {expected_type!r}. Failing check.")
                fail_msg = "Annotated type hint requires at least one type argument"
                expected_repr_err = "Annotated" # Simple repr for this specific error
                # Note: Not caching this specific structural failure of Annotated itself
                # Ensure path is included in details, even for this structural error
                return False, _create_obituary(expected_repr_err, value, path, fail_msg)

            # Check against the underlying type (first argument)
            underlying_type = _args[0]
            if _log.isEnabledFor(logging.DEBUG):
                underlying_repr = format_type_for_display(underlying_type)
                _log.debug(f"--- CHECK_TYPE RECURSING (Annotated): Path='{_format_path(path)}', Checking underlying type: {underlying_repr} ---")

            # Recursive call with the *underlying* type, ignoring annotations
            # Pass the original globalns, localns, path, func_id, raise_error
            # Cache check happens *inside* the recursive call if applicable
            return check_type(
                value=value,
                expected_type=underlying_type,
                globalns=globalns,
                localns=localns,
                path=path,
                func_id=func_id,
                raise_error=raise_error # Propagate raise_error setting
            )
    # --- END Annotated Handling ---

    # Ensure globalns is provided (essential for forward refs and hints)
    if globalns is None:
        _log.error("CRITICAL: check_type called without globalns!")
        # Decide on fallback: raise error, return False, or use empty dict?
        # Using empty dict might hide errors but allows execution to proceed.
        # Raising error is safer but might be too strict if called unexpectedly.
        # Returning False with details is a middle ground.
        return False, _create_obituary( # Use Obituary helper
            format_type_for_display(expected_type),
            value,
            path,
            "Internal Error: globalns not provided to check_type"
        )

    # Get function ID for TypeVar tracking if localns provided and func_id not passed
    # Prefer explicitly passed func_id if available
    if func_id is None and localns:
         func_id = localns.get("_func_id")

    # Initialize path if None (redundant now, path_tuple used for cache key)
    current_path = path if path is not None else []

    is_safe_to_cache = False # Default to False
    try:
        is_literal = hasattr(typing, 'Literal') and get_origin(expected_type) is Literal
        is_union, _ = is_union_type(expected_type)
        is_protocol = hasattr(typing, 'Protocol') and isinstance(expected_type, type) and issubclass(expected_type, typing.Protocol)
        is_newtype = hasattr(expected_type, '__supertype__')
        is_generic = is_generic_alias(expected_type)
        args = get_args(expected_type) if is_generic else None

        if expected_type in (int, str, float, bool, NoneType, Any):
            is_safe_to_cache = True
        elif is_literal:
            is_safe_to_cache = True
        elif is_generic and not args:
            # Allow caching unparameterized built-in generics (list, dict, etc.)
            origin = get_origin(expected_type)
            if origin in (list, dict, set, tuple, collections.abc.Sequence, collections.abc.Mapping, collections.abc.Set):
                is_safe_to_cache = True
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"--- CHECK_TYPE CACHE ALLOW: Unparameterized generic {format_type_for_display(expected_type)} ---")
        elif isinstance(expected_type, type) and not (is_generic or is_union or is_protocol or is_newtype):
             # Allow simple custom types, excluding known complex categories
             is_safe_to_cache = True
             if _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"--- CHECK_TYPE CACHE ALLOW: Simple type {format_type_for_display(expected_type)} ---")

        # Explicitly unsafe types override previous decision
        if isinstance(expected_type, (str, ForwardRef, TypeVar)) or is_union or is_protocol or is_newtype or (is_generic and args):
            is_safe_to_cache = False
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE CACHE SKIP: Complex/Contextual type {format_type_for_display(expected_type)} ---")

    except Exception as e:
        # Catch errors during safety check itself
        _log.warning(f"Error determining cache safety for {expected_type!r}: {e}. Defaulting to unsafe.", exc_info=True)
        is_safe_to_cache = False

        expected_repr = format_type_for_display(expected_type)

    if is_safe_to_cache:
        cache_key = (type(value), expected_type, func_id, path_tuple)
        with _check_type_cache_lock:
            cached_result = _check_type_cache_obituary.get(cache_key)
        if cached_result is not None:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE CACHE HIT: Key={cache_key}, Path='{_format_path(path)}' ---")
            return cached_result
        elif _log.isEnabledFor(logging.DEBUG):
             _log.debug(f"--- CHECK_TYPE CACHE MISS: Key={cache_key}, Path='{_format_path(path)}' ---")
    elif _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"--- CHECK_TYPE CACHE SKIP: Type={expected_type!r}, Path='{_format_path(path)}' ---")

    # --- Handle Optional[X] *before* main handlers --- #
    is_opt, opt_inner = is_optional_type(expected_type)
    if is_opt:
        if _log.isEnabledFor(logging.DEBUG):
             _log.debug(f"--- CHECK_TYPE SHORTCUT: Path='{_format_path(path)}', Detected Optional. Calling _check_optional directly.")

        # Call the modified _check_optional directly
        final_result = _check_optional(value, expected_type, opt_inner, globalns, localns, path)

        # --- Cache Write for Optional Result --- #
        if is_safe_to_cache and cache_key is not None:
            with _check_type_cache_lock:
                _check_type_cache_obituary[cache_key] = final_result
            if _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"+++ CHECK_TYPE CACHE WRITE (Optional Shortcut): Key={cache_key}, Result={final_result}, Path='{_format_path(path)}' +++")

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"--- CHECK_TYPE EXIT (Optional Shortcut): Path='{_format_path(path)}', Final Result={final_result!r} ---")

        # Raise error if requested and match failed
        if raise_error and not final_result[0]:
            # Raise the custom exception type
            # Construct message using the Obituary object
            raise YouDiedError(_construct_type_error(final_result[1]), 'optional_mismatch') # Example cause

        return final_result
    # --- End Optional Shortcut --- #

    if expected_type is Any:
        result = (True, None)
        if is_safe_to_cache and cache_key is not None:
             with _check_type_cache_lock:
                 _check_type_cache_obituary[cache_key] = result
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"--- CHECK_TYPE EXIT (Match: Any): Path='{_format_path(current_path)}'")
        return result

    if value is None:
        origin = get_origin(expected_type)
        allows_none = (
            expected_type is None
            or expected_type is NoneType
            or (origin is Union and NoneType in get_args(expected_type))
        )
        if allows_none:
            result = (True, None)
            if is_safe_to_cache and cache_key is not None:
                with _check_type_cache_lock:
                    _check_type_cache_obituary[cache_key] = result
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE EXIT (Match: None allowed): Path='{_format_path(current_path)}'")
            return result
        else:
            fail_obituary = _create_obituary(expected_repr if _log.isEnabledFor(logging.DEBUG) else format_type_for_display(expected_type), value, current_path, "Expected non-None value")
            result = (False, fail_obituary)
            if is_safe_to_cache and cache_key is not None:
                with _check_type_cache_lock:
                    _check_type_cache_obituary[cache_key] = result
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE EXIT (Fail: Expected non-None): Path='{_format_path(current_path)}', Details: {fail_obituary!r}")
            return result

    # Initialize default result
    match: bool = False
    details_obj: Optional[Obituary] = None
    final_result: Tuple[bool, Optional[Obituary]] = (False, None) # Define final_result early
    handler_found_match = False # Flag to track if any handler made a decision

    try:
        # --- 3-Argument Handlers --- #
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"--- CHECK_TYPE LOOP (3-arg): Path='{_format_path(path)}', Trying handlers: {[h.__name__ for h in _3_ARG_HANDLERS]}")
        for handler in _3_ARG_HANDLERS:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE HANDLER (3-arg): Path='{_format_path(path)}', Calling {handler.__name__}")
            handler_result = handler(value, expected_type, path)

            if handler_result is None:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"--- CHECK_TYPE HANDLER (3-arg): Path='{_format_path(path)}', {handler.__name__} returned: None (skipped)")
                continue

            if not isinstance(handler_result, tuple) or len(handler_result) != 2:
                _log.warning(f"Handler {handler.__name__} returned unexpected format: {handler_result!r}. Skipping handler.")
                continue

            match, details_obj = handler_result
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE HANDLER (3-arg): Path='{_format_path(path)}', {handler.__name__} returned: {(match, details_obj)}")
            if match or details_obj is not None:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"--- CHECK_TYPE DECISION (3-arg handler {handler.__name__}): Path='{_format_path(path)}', Result={(match, details_obj)}")
                final_result = (True, None) if match else (False, details_obj or _create_obituary(format_type_for_display(expected_type), value, path, f"Check failed in {handler.__name__}"))
                handler_found_match = True # Mark that a handler made a decision
                break # Exit the 3-arg loop

        # --- 5-Argument Handlers (only if no 3-arg handler matched) --- #
        if not handler_found_match:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE LOOP (5-arg): Path='{_format_path(path)}', Trying handlers: {[h.__name__ for h in _TYPE_HANDLERS_5ARGS]}")
            for handler in _TYPE_HANDLERS_5ARGS:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"--- CHECK_TYPE HANDLER (5-arg): Path='{_format_path(path)}', Calling {handler.__name__}")
                handler_result = handler(value, expected_type, globalns, localns, path)

                if handler_result is None:
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"5-arg Handler {handler.__name__} returned None (type not handled). Skipping.")
                    continue

                if not isinstance(handler_result, tuple) or len(handler_result) != 2:
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.warning(f"Handler {handler.__name__} returned unexpected format: {handler_result!r}. Skipping handler.")
                    continue

                match, details_obj = handler_result
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"--- CHECK_TYPE HANDLER (5-arg): Path='{_format_path(path)}', {handler.__name__} returned: {(match, details_obj)}")
                if match or details_obj is not None:
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"--- CHECK_TYPE DECISION (5-arg handler {handler.__name__}): Path='{_format_path(path)}', Result={(match, details_obj)}")
                    final_result = (True, None) if match else (False, details_obj or _create_obituary(format_type_for_display(expected_type), value, path, f"Check failed in {handler.__name__}"))
                    handler_found_match = True # Mark that a handler made a decision
                    break # Exit the 5-arg loop

        # --- Default Fallback (only if NO handler matched) --- #
        if not handler_found_match:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE DEFAULT FALLBACK: Path='{_format_path(path)}', No handler matched. Calling _check_simple_type.")
            match, details_obj = _check_simple_type(value, expected_type, path)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"--- CHECK_TYPE DEFAULT FALLBACK RESULT: Path='{_format_path(path)}', Result={(match, details_obj)}")
        final_result = (match, details_obj or (_create_obituary(format_type_for_display(expected_type), value, path, "Type failed default isinstance check") if not match else None))

    except Exception as e:
        # Catch unexpected errors during checking
        _log.error(f"!!! CHECK_TYPE ERROR: Path='{_format_path(path)}', Unexpected exception during check: {e!r}", exc_info=True)
        final_result = (False, _create_obituary(format_type_for_display(expected_type), value, path, f"Internal error during type check: {e!r}"))

    finally:
        # --- Cache Write --- #
        if is_safe_to_cache and cache_key is not None:
            with _check_type_cache_lock:
                _check_type_cache_obituary[cache_key] = final_result
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"+++ CHECK_TYPE CACHE WRITE: Key={cache_key}, Result={final_result}, Path='{_format_path(path)}' +++")

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"--- CHECK_TYPE EXIT: Path='{_format_path(path)}', Final Result={final_result!r} ---")

        # Raise error if requested and match failed
        if raise_error and not final_result[0]:
            # Raise the custom exception type
            # Construct message using the Obituary object
            raise YouDiedError(_construct_type_error(final_result[1]), 'check_type_failure')

        return final_result

# ===== PUBLIC API EXPORTS ===== #

# Explicitly list functions intended for use by other modules (decorator.py mostly)
__all__: Final[List[str]] = [
    'check_type',
    'get_resolved_type_hints',
    'get_cached_mro_set', # Exported for potential direct use or testing
    'is_instance_optimized', # Exported for potential direct use or testing
    'clear_typevar_bindings', # Needed by decorator wrapper
    'bind_typevar', # Needed for testing/potential advanced use
    'get_typevar_binding', # Needed for testing/potential advanced use
    '_get_caller_info', # Needed by decorator
    'NoneType', # Useful constant
    '_RETURN_ANNOTATION', # Constant needed by decorator
    # 'YouDiedError' # Removed from exports here, will be exported from error_utils
]