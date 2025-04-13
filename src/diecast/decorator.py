# ===== MODULE DOCSTRING ===== #
"""
Main decorator module for DieCast.

This module provides the decorator that implements runtime type checking
based on type annotations. It handles both synchronous and asynchronous
functions, as well as generators and their async counterparts.

The decorator performs the following checks:
1. Validates argument types against their annotations
2. Validates return values against their annotations
3. For generators, validates yielded values and final return value
4. Provides detailed error messages with stack traces when type violations occur

Usage:
    import diecast

    @diecast.diecast
    def process_data(items: list[int]) -> dict:
        return {"sum": sum(items)}

    @diecast.diecast
    async def fetch_data(url: str) -> dict:
        # async implementation
        pass

    @diecast.diecast
    def generate_numbers(n: int) -> Generator[int, None, None]:
        for i in range(n):
            yield i
"""
#-#

# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
import collections.abc
from typing import (
    Coroutine, AsyncGenerator, AsyncIterator, 
    Callable, Generator, Iterator,
    Type, TypeVar, Tuple,
    Optional, Final, Union, 
    Dict, List, Any, 
)
import threading
import functools
import inspect
import logging
import asyncio
import typing
import types
import sys
##-##

## ===== LOCAL ===== ##
from .error_utils import (
    generate_return_error_message, 
    generate_arg_error_message,
    _get_caller_info,
    Obituary
)
from .type_utils import (
    get_resolved_type_hints,
    format_type_for_display,
    clear_typevar_bindings,
    YouDiedError, 
    check_type
)
from .config import (
    _RETURN_ANNOTATION, _DIECAST_MARKER,
    _SELF_NAMES
)
from .logging import _log
##-##
#-#

# ===== GLOBALS ===== #

## ===== EXPORTS ===== ## 
__all__: Final[List[str]] = ['diecast', 'ignore']

# Global cache for specialized generic classes to avoid redundant creation
_SPECIALIZED_CLASS_CACHE: Dict[Tuple[Type, Any], Type] = {}
_SPECIALIZED_CLASS_CACHE_LOCK = threading.Lock()
##-##
#-#

# ===== FUNCTIONS ===== #

## ===== TYPE ERROR HANDLING ===== ##
def _handle_type_error(
    error_type: str, # 'argument', 'return', 'yield'
    func_info: Dict[str, Any],
    annotation: Any, # The type the check was performed against (potentially resolved)
    value: Any,
    obituary: Optional[Any], # Should be Obituary object
    caller_depth: int,
    param: Optional[inspect.Parameter] = None, # For arg errors
    arg_index: Optional[int] = None, # For arg errors
    is_kwarg: Optional[bool] = None, # For arg errors
    is_yield_check: bool = False, # Distinguish yield from return
    original_annotation: Optional[Any] = None # The original annotation (e.g., the TypeVar)
) -> None:
    """Centralized function to handle type errors: get info, generate message, raise."""
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE _handle_type_error: Received error_type='{error_type}', obituary={obituary!r}")
        _log.debug(f"TRACE _handle_type_error: Received error_type='{error_type}', obituary={obituary!r}")
        _log.debug(f"TRACE decorator._handle_type_error: Handling '{error_type}' error for '{func_info['func_name']}'")

    caller_info = _get_caller_info(depth=caller_depth)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._handle_type_error: Gathered caller_info: {caller_info!r}")

    if error_type == 'argument':
        error_msg = generate_arg_error_message(
            func_name=func_info['func_name'], func_module=func_info['func_module'], func_lineno=func_info['func_lineno'],
            func_class_name=func_info.get('func_class_name'),
            param=param, annotation=annotation, value=value,
            arg_index=arg_index, is_kwarg=is_kwarg,
            caller_info=caller_info, obituary=obituary,
            original_annotation=original_annotation # Pass down original annotation
        )
    elif error_type == 'return' or error_type == 'yield':
        error_msg = generate_return_error_message(
            func_name=func_info['func_name'], func_module=func_info['func_module'], func_lineno=func_info['func_lineno'],
            func_class_name=func_info.get('func_class_name'),
            annotation=annotation, value=value,
            caller_info=caller_info, obituary=obituary,
            is_yield_value=is_yield_check, # Use the flag passed in
            original_annotation=original_annotation # Pass down original annotation
        )
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"!!! _handle_type_error: Calling generate_return_error_message with annotation={annotation!r}, original_annotation={original_annotation!r}") # DEBUG LOG
    else:
        # Fallback for unknown error types
        _log.error(f"_handle_type_error called with unknown error_type: {error_type}")
        error_msg = f"Unknown type check error ({error_type}) in {func_info['func_name']}"

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._handle_type_error: Generated error message (len={len(error_msg)}).")
        _log.debug(f"TRACE decorator._handle_type_error: Raising YouDiedError with obituary: {repr(obituary)}") 
    raise YouDiedError(error_msg, obituary=obituary, cause=error_type)
##-##

## ===== FUNCTION INFO ===== ##
def _get_func_info(func: Callable) -> Dict[str, Any]:
    """Extracts and stores basic information about the decorated function for later use.

    Args:
        func: The function being decorated.

    Returns:
        A dictionary containing function name, module, line number, and the object itself.
    """
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._get_func_info: Entering with func={func!r}")
    try:
        first_line = func.__code__.co_firstlineno
    except AttributeError:
        _log.debug("TRACE decorator._get_func_info: func has no __code__.co_firstlineno, setting line to 'unknown'.")
        first_line = 'unknown' # Handle functions without __code__

    func_name = getattr(func, '__name__', 'unknown')
    func_module = getattr(func, '__module__', 'unknown')
    func_qualname = getattr(func, '__qualname__', func_name)
    func_class_name = func_qualname.rsplit('.', 1)[0] if '.' in func_qualname and func_qualname != func_name else None

    info = {
        'func_name': func_name,
        'func_module': func_module,
        'func_lineno': first_line,
        'func_object': func, # Store the actual function object reference
        'func_class_name': func_class_name
    }
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._get_func_info: Extracted info: {info!r}")
        _log.debug(f"TRACE decorator._get_func_info: Exiting")
    return info
##-##

## ===== ARGUMENT CHECKING ===== ##
def _check_arguments(
    sig: inspect.Signature,
    hints: Dict[str, Any],
    bound_args: inspect.BoundArguments,
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    func_info: Dict[str, Any],
    func_id: int,
    instance_map: Optional[Dict[TypeVar, Type]] = None # ADDED instance_map parameter
) -> None:
    """Check function arguments against type hints using the bound arguments.

    Uses `check_type` for validation and raises TypeError on mismatch.
    Relies on `_get_caller_info` and `generate_arg_error_message` for errors.
    Now uses an explicit `instance_map` provided by the wrapper for generic resolution.

    Args:
        sig: The inspect.Signature object for the function.
        hints: The dictionary of resolved type hints.
        bound_args: The inspect.BoundArguments object containing call arguments.
        globalns: Global namespace for type resolution.
        localns: Local namespace (may include _func_id for TypeVars).
        func_info: Dictionary with function details.
        func_id: The base function ID for TypeVar tracking.
        instance_map: Optional map from TypeVar -> resolved Type for generic instance methods. # ADDED DOCS

    Raises:
        TypeError: If any argument fails its type check.
    """
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_arguments: Entering for func_id={func_id} ('{func_info['func_name']}') with instance_map: {instance_map}") # UPDATED Log
        _log.debug(f"TRACE decorator._check_arguments: Bound arguments: {bound_args.arguments!r}")
        _log.debug(f"TRACE decorator._check_arguments: Hints: {hints!r}")
    
    caller_depth = 2
    effective_localns = (localns or {}).copy()
    effective_localns['_func_id'] = func_id
    bound_args.apply_defaults()

    ### ----- Determine if this is likely an instance method and get self/cls ----- ###
    first_param_name = next(iter(sig.parameters), None)
    instance_obj = None
    is_likely_instance_method = False
    if first_param_name is not None and first_param_name in _SELF_NAMES:
        if first_param_name in bound_args.arguments:
            instance_obj = bound_args.arguments[first_param_name]
            if instance_obj is not None and not inspect.isclass(instance_obj):
                is_likely_instance_method = True
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: Detected likely instance method context. Instance: {instance_obj!r}")
            else:
                 if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: First arg '{first_param_name}' is class or None. Assuming class/static method context for TypeVar resolution.")
        else:
             if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_arguments: First arg '{first_param_name}' not in bound args. Cannot determine instance context.")
    ###-###

    for i, (name, param) in enumerate(sig.parameters.items()):
        ### ----- Guards for self/cls/Any/missing hints ----- ###
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_arguments: Checking param #{i}: name='{name}', Parameter={param}")

        # Skip self/cls arguments implicitly
        if i == 0 and name in _SELF_NAMES:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_arguments: Skipping parameter '{name}' (self/cls arg check).")
            continue

        if name not in hints:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_arguments: Skipping parameter '{name}' (no type hint).")
            continue

        annotation = hints[name]
        if annotation is Any:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_arguments: Skipping parameter '{name}' (annotation is Any).")
            continue
        ###-###

        ### ----- TypeVar Resolution (USING PASSED instance_map) ----- ###
        effective_annotation = annotation # Start with original hint
        
        # PRIORITY 1: Use provided instance map
        if instance_map is not None and isinstance(annotation, TypeVar):
            resolved_type = instance_map.get(annotation)
            if resolved_type is not None and not isinstance(resolved_type, TypeVar): # Ensure it's resolved to a concrete type or unbound TypeVar
                effective_annotation = resolved_type
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: Resolved TypeVar {annotation!r} to {resolved_type!r} using PASSED instance map for param '{name}'.")
            # else: TypeVar not in map or mapped to another TypeVar - use original annotation

        # PRIORITY 2: Fallback to function-level consistency (handled in check_type -> _check_typevar)
        ###-###

        ### ----- Prepare local namespace (excluding current arg value) ----- ###
        current_arg_localns = effective_localns.copy()
        current_arg_localns.pop(name, None)
        ###-###
        
        ### ----- Handle different parameter kinds (Using effective_annotation) ----- ###
        if param.kind == param.VAR_POSITIONAL: # *args
            if name in bound_args.arguments:
                args_tuple = bound_args.arguments[name]
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: Checking *args parameter '{name}'. Value: {args_tuple!r}, Element Annotation: {effective_annotation!r}")
                if not isinstance(args_tuple, tuple):
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.warning(f"Expected tuple for *args parameter '{name}', but got {type(args_tuple)}. Skipping check.")
                    continue
                for idx, arg_value in enumerate(args_tuple):
                    path = [f"{name}[{idx}]"] # Construct path like args[0], args[1]
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._check_arguments: Checking *args element at index {idx}: value={arg_value!r} against {effective_annotation!r} at path {path!r}")
                    # Pass instance_map down to check_type
                    match, obituary = check_type(arg_value, effective_annotation, globalns, current_arg_localns, path=path, instance_map=instance_map)
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._check_arguments: check_type result: match={match}, details={obituary!r}")
                    if not match:
                        # !!! ADDED DEBUG LOG !!!
                        _handle_type_error(
                            error_type='argument',
                            func_info=func_info,
                            annotation=effective_annotation,
                            value=arg_value,
                            obituary=obituary,
                            caller_depth=caller_depth,
                            param=param, # Pass the *args param itself
                            arg_index=idx,
                            is_kwarg=False # Treat elements as positional within *args
                        )
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: *args parameter '{name}' not found in bound arguments.")

        elif param.kind == param.VAR_KEYWORD: # **kwargs
            if name in bound_args.arguments:
                kwargs_dict = bound_args.arguments[name]
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: Checking **kwargs parameter '{name}'. Value: {kwargs_dict!r}, Value Annotation: {effective_annotation!r}")
                if not isinstance(kwargs_dict, dict):
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.warning(f"Expected dict for **kwargs parameter '{name}', but got {type(kwargs_dict)}. Skipping check.")
                    continue
                for kwarg_key, kwarg_value in kwargs_dict.items():
                    path = [f"{name}[{kwarg_key!r}]" ] # Construct path like kwargs['key']
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._check_arguments: Checking **kwargs value for key '{kwarg_key}': value={kwarg_value!r} against {effective_annotation!r} at path {path!r}")
                        _log.debug(f"TRACE decorator._check_arguments [**kwargs]: BEFORE check_type for key='{kwarg_key}'.")
                    # Pass instance_map down to check_type
                    match, obituary = check_type(kwarg_value, effective_annotation, globalns, current_arg_localns, path=path, instance_map=instance_map)
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._check_arguments [**kwargs]: AFTER check_type for key='{kwarg_key}'. Result: match={match}, details={obituary!r}")
                        _log.debug(f"TRACE decorator._check_arguments: check_type result: match={match}, details={obituary!r}")
                    if not match:
                        _handle_type_error(
                            error_type='argument',
                            func_info=func_info,
                            annotation=effective_annotation, # Use resolved type in error
                            value=kwarg_value,
                            obituary=obituary,
                            caller_depth=caller_depth,
                            param=param, # Pass the **kwargs param itself
                            arg_index=None, # Index not applicable
                            is_kwarg=True
                        )
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: **kwargs parameter '{name}' not found in bound arguments.")

        else: # POSITION_OR_KEYWORD, KEYWORD_ONLY, POSITIONAL_ONLY
            if name in bound_args.arguments:
                value = bound_args.arguments[name]
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: Checking value={value!r} against annotation={effective_annotation!r}")
                # Pass instance_map down to check_type
                match, obituary = check_type(value, effective_annotation, globalns, current_arg_localns, path=[name], instance_map=instance_map)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: check_type result: match={match}, details={obituary!r}")
                if not match:
                    # !!! ADDED DEBUG LOG !!!
                    is_kwarg = name in bound_args.kwargs
                    _handle_type_error(
                        error_type='argument',
                        func_info=func_info,
                        annotation=effective_annotation, # Use resolved type in error
                        value=value,
                        obituary=obituary,
                        caller_depth=caller_depth,
                        param=param,
                        arg_index=i,
                        is_kwarg=is_kwarg
                    )
            else:
                # This might happen if an argument is missing but has a default
                # that wasn't applied correctly, or if binding failed silently.
                if _log.isEnabledFor(logging.DEBUG):
                    _log.warning(f"Argument '{name}' missing from bound_args.arguments despite apply_defaults(). Skipping check.")
            ###-###

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_arguments: All argument checks passed for func_id={func_id} ('{func_info['func_name']}'). Exiting.")
##-##

## ===== RETURN VALUE CHECKING ===== ##
def _check_return_value(
    result: Any,
    hints: Dict[str, Any],
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    func_info: Dict[str, Any],
    func_id: int,
    instance_map: Optional[Dict[TypeVar, Type]] = None # ADDED instance_map parameter
) -> Any:
    """Checks the return value type against annotations or wraps generators.

    Handles regular return values, sync/async generators, and coroutine results.
    Uses `check_type` for validation and `_analyze_stack_and_raise` on failure.
    Now uses an explicit `instance_map` provided by the wrapper for generic resolution.

    Args:
        result: The value returned by the decorated function.
        hints: The dictionary of resolved type hints for the function.
        globalns: Global namespace for type resolution.
        localns: Local namespace for type resolution (may include _func_id).
        func_info: Dictionary containing information about the decorated function.
        func_id: The base function ID for TypeVar tracking.
        instance_map: Optional map from TypeVar -> resolved Type for generic instance methods. # ADDED DOCS

    Returns:
        The original result if the check passes, or a wrapped generator.

    Raises:
        TypeError: If the return value fails its type check.
    """
    ### ----- Defensive guards for unhandlableable cases ----- ###
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: Entering for func_id={func_id} ('{func_info['func_name']}') with instance_map: {instance_map}") # UPDATED Log
        _log.debug(f"TRACE decorator._check_return_value: Result={result!r}, Type={type(result).__name__}, Hints={hints!r}")
    
    if _RETURN_ANNOTATION not in hints:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: No return annotation found. Skipping check.")
            _log.debug(f"TRACE decorator._check_return_value: Exiting, returning original result.")
        return result

    return_annotation = hints[_RETURN_ANNOTATION]
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: Found return annotation: {return_annotation!r}")
    
    if return_annotation is Any:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: Return annotation is Any. Skipping check.")
            _log.debug(f"TRACE decorator._check_return_value: Exiting, returning original result.")
        return result
    ###-###

    ### ----- BEGIN TypeVar Resolution (USING PASSED instance_map) ----- ###
    effective_return_annotation = return_annotation
    
    # PRIORITY 1: Use provided instance map
    if instance_map is not None and isinstance(return_annotation, TypeVar):
        resolved_type = instance_map.get(return_annotation)
        if resolved_type is not None and not isinstance(resolved_type, TypeVar): # Ensure it's resolved
            effective_return_annotation = resolved_type
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Resolved return TypeVar {return_annotation!r} to {resolved_type!r} using PASSED instance map.")

    # PRIORITY 2: Fallback to function-level consistency (handled in check_type -> _check_typevar)
    ###-###

    ### ----- Prepare local namespace (excluding current return value) ----- ###
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE _check_return_value: Received instance_map: {instance_map!r}")
        _log.debug(f"TRACE _check_return_value: Original return_annotation: {return_annotation!r}")
        _log.debug(f"TRACE _check_return_value: Effective return_annotation after map lookup: {effective_return_annotation!r}")
    origin = typing.get_origin(effective_return_annotation) # Use potentially resolved annotation
    args_return = typing.get_args(effective_return_annotation) # Use potentially resolved annotation
    caller_depth = 2

    # Prepare local namespace with function ID for check_type
    effective_localns = (localns or {}).copy()
    effective_localns['_func_id'] = func_id
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: Effective localns for check_type: {effective_localns!r}")
    ###-###

    ### ----- Generator Handling (using effective_return_annotation) ----- ###
    # Define sets of sync and async generator/iterator types
    SYNC_GENERATOR_TYPES = {Generator, collections.abc.Generator, Iterator, collections.abc.Iterator}
    ASYNC_GENERATOR_TYPES = {AsyncGenerator, collections.abc.AsyncGenerator, AsyncIterator, collections.abc.AsyncIterator}

    is_sync_gen_hint = origin in SYNC_GENERATOR_TYPES
    is_sync_gen_result = isinstance(result, collections.abc.Generator)
    
    if is_sync_gen_hint or is_sync_gen_result:
        # Extract Yield and Return types based on the hint origin
        yield_type = Any
        ret_type = Any
        if is_sync_gen_hint and args_return:
            yield_type = args_return[0]
            # Only Generator types have a return type argument (at index 2)
            if origin in {Generator, collections.abc.Generator} and len(args_return) > 2:
                ret_type = args_return[2]
        
        effective_yield_type, effective_ret_type = yield_type, ret_type
        
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: Sync Gen/Iter Effective Types: yield={effective_yield_type!r}, return={effective_ret_type!r}")

        # Check if the result matches the hint expectation (is it a generator?)
        if not is_sync_gen_result:
             if is_sync_gen_hint:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_return_value: Mismatch: Hinted Generator/Iterator but result is not a Generator ({type(result).__name__}). Raising YouDiedError.")
                match_ignored, obituary = check_type(result, effective_return_annotation, globalns, effective_localns, instance_map=instance_map)
                _handle_type_error(
                    error_type='return',
                    func_info=func_info,
                    annotation=effective_return_annotation, 
                    value=result,
                    obituary=obituary,
                    caller_depth=caller_depth,
                    is_yield_check=False
                )
             else:
                  # Result is generator, hint isn't - this shouldn't happen if hint resolution is correct, but proceed safely
                  if _log.isEnabledFor(logging.DEBUG):
                      _log.debug(f"TRACE decorator._check_return_value: Result is Generator, hint is not. Proceeding to standard return check (likely error).")
                  pass # Fall through to standard check
        else:
            # Result is a generator, and hint was either a generator type or Any/unspecified
            if effective_yield_type is not Any or effective_ret_type is not Any:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_return_value: Wrapping sync generator '{func_info['func_name']}' with effective types.")
                wrapped_gen = _diecast_wrap_generator_sync(
                    result, yield_type, ret_type,
                    globalns, effective_localns, func_info, func_id, instance_map=instance_map
                )
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_return_value: Exiting, returning wrapped sync generator: {wrapped_gen!r}")
                return wrapped_gen
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_return_value: Sync generator for '{func_info['func_name']}' effective yield/return types are Any. Returning original generator.")
                return result
    ###-###

    ### ----- Async Generator Handling ----- ###
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE _check_return_value: Calling check_type for return value. Value={result!r}, Annotation={effective_return_annotation!r}")
    is_async_gen_hint = origin in ASYNC_GENERATOR_TYPES
    is_async_gen_result = isinstance(result, collections.abc.AsyncGenerator)
    
    if is_async_gen_hint or is_async_gen_result:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE _check_return_value: Calling check_type for return value. Value={result!r}, Annotation={effective_return_annotation!r}")
        # Extract Yield type based on the hint origin
        yield_type = Any
        if is_async_gen_hint and args_return:
            yield_type = args_return[0]
            # Note: AsyncGenerators also have a SendType at index 1, which we ignore.
            # AsyncIterators only have YieldType.
        
        effective_yield_type = yield_type
        
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: Async Gen/Iter Effective Types: yield={effective_yield_type!r}")

        # Check if the result matches the hint expectation (is it an async generator?)
        if not is_async_gen_result:
            if is_async_gen_hint:
                _log.debug(f"TRACE decorator._check_return_value: Mismatch: Hinted AsyncGenerator/Iterator but result is not an AsyncGenerator ({type(result).__name__}). Raising YouDiedError.")
                match_ignored, obituary = check_type(result, effective_return_annotation, globalns, effective_localns, instance_map=instance_map)
                _handle_type_error(
                    error_type='return',
                    func_info=func_info,
                    annotation=effective_return_annotation, 
                    value=result,
                    obituary=obituary,
                    caller_depth=caller_depth,
                    is_yield_check=False
                )
            else:
                # Result is async generator, hint isn't - proceed safely
                if _log.isEnabledFor(logging.DEBUG):
                     _log.debug(f"TRACE decorator._check_return_value: Result is AsyncGenerator, hint is not. Proceeding to standard return check (likely error).")
                pass # Fall through to standard check
        else:
            # Result is an async generator, and hint was appropriate or Any/unspecified
            if effective_yield_type is not Any:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_return_value: Wrapping async generator '{func_info['func_name']}' with effective types.")
                wrapped_agen = _diecast_wrap_generator_async(
                    result, yield_type,
                    globalns, effective_localns, func_info, func_id, instance_map=instance_map
                )
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_return_value: Exiting, returning wrapped async generator: {wrapped_agen!r}")
                return wrapped_agen
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_return_value: Async generator for '{func_info['func_name']}' effective yield type is Any. Returning original async generator.")
                return result
    ###-###

    ### ----- Coroutine Result Handling ----- ###
    # --- Determine the actual type to check against ---
    check_against_type = effective_return_annotation # Start with the potentially resolved annotation
    is_coroutine_result_check = False # Not relevant here, but kept for structure
    
    # If the original hint was a Coroutine, extract the inner return type
    # Note: We use the *original* origin here, not the potentially resolved one.
    original_origin = typing.get_origin(return_annotation)
    if original_origin is Coroutine and args_return:
        # Coroutine[T, U, V] -> V is return type at index -1
        coroutine_return_type = args_return[-1]
        
        # Resolve the inner coroutine return type if it's a TypeVar
        effective_coroutine_return_type = coroutine_return_type
        if instance_map is not None and isinstance(coroutine_return_type, TypeVar):
            resolved_inner = instance_map.get(coroutine_return_type)
            if resolved_inner is not None and not isinstance(resolved_inner, TypeVar):
                effective_coroutine_return_type = resolved_inner

        check_against_type = effective_coroutine_return_type # Update check target
        
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: Hint is Coroutine. Checking result against effective inner type: {check_against_type!r}")
        if check_against_type is Any:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Coroutine inner type is Any. Skipping check.")
            return result # Don't check if inner type is Any
    # else: Not a coroutine hint, check_against_type remains effective_return_annotation

    if _log.isEnabledFor(logging.DEBUG):
         _log.debug(f"TRACE decorator._check_return_value: Final type check against: {check_against_type!r}")
    ###-###

    ### ----- Standard Return Value Check ----- ###
    match, obituary = check_type(result, check_against_type, globalns, effective_localns, instance_map=instance_map)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: check_type result: match={match}, details={obituary!r}")

    if not match:
        # Create a new Obituary with the correct path for return errors,
        # copying details from the one returned by check_type.
        if obituary:
            final_obituary = Obituary(
                expected_repr=obituary.expected_repr,
                received_repr=obituary.received_repr,
                value=obituary.value,
                path=['return'], # Explicitly set path for return errors
                message=obituary.message # Keep the original detailed message
            )
        else:
            # Fallback if check_type somehow returned match=False but obituary=None
            expected_repr = format_type_for_display(check_against_type)
            received_repr = format_type_for_display(type(result))
            final_obituary = Obituary(
                expected_repr=expected_repr,
                received_repr=received_repr,
                value=result,
                path=['return'],
                message=f"Return value type mismatch (Expected: {expected_repr}, Received: {received_repr})"
            )

        _handle_type_error(
            error_type='return',
            func_info=func_info,
            annotation=check_against_type, # The type we actually checked against
            value=result,
            obituary=final_obituary, # Pass the NEW obituary
            caller_depth=caller_depth,
            is_yield_check=False # Explicitly False for return value checks
        )

    # Check passed
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: Type check PASSED for return value of '{func_info['func_name']}'.")
        _log.debug(f"TRACE decorator._check_return_value: Exiting, returning original result.")
    return result
    ###-###
##-##

## ===== GENERATOR WRAPPERS ===== ##
def _diecast_wrap_generator_sync(
    gen: Generator,
    yield_type: Union[Type, TypeVar],
    ret_type: Union[Type, TypeVar],
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    func_info: Dict[str, Any],
    func_id: int,
    instance_map: Optional[Dict[TypeVar, Type]]
) -> Generator:
    """Wraps a synchronous generator to check yielded values and the final return value.

    Args:
        gen: The original generator object.
        yield_type: The expected type of yielded values.
        ret_type: The expected type of the generator's return value.
        globalns: Global namespace for type resolution.
        localns: Local namespace for type resolution (includes _func_id).
        func_info: Dictionary containing information about the decorated function.
        func_id: The base function ID for TypeVar tracking.
        instance_map: Optional[Dict[TypeVar, Type]] = None

    Yields:
        Values from the original generator after type checking.

    Returns:
        The final return value of the generator after type checking.

    Raises:
        TypeError: If a yielded value or the return value violates type hints.
    """
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Entering wrapper for func_id={func_id} ('{func_info['func_name']}') with instance_map: {instance_map}") # UPDATED Log
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Expected yield={yield_type!r}, return={ret_type!r}")
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Namespaces: globalns keys={list(globalns.keys())!r}, localns={localns!r}")
    
    effective_yield_type = yield_type
    if instance_map is not None and isinstance(yield_type, TypeVar):
        resolved = instance_map.get(yield_type)
        if resolved is not None and not isinstance(resolved, TypeVar):
            effective_yield_type = resolved
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Resolved yield TypeVar {yield_type!r} to {resolved!r} via instance_map.")

    effective_ret_type = ret_type
    if instance_map is not None and isinstance(ret_type, TypeVar):
        resolved = instance_map.get(ret_type)
        if resolved is not None and not isinstance(resolved, TypeVar):
            effective_ret_type = resolved
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Resolved return TypeVar {ret_type!r} to {resolved!r} via instance_map.")

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Effective types: yield={effective_yield_type!r}, return={effective_ret_type!r}")

    caller_depth_yield = 3 # wrapper -> next(gen) -> yield -> user code -> user code's caller
    caller_depth_return = 3 # Similar depth for return check
    gen_index = 0
    return_value_from_stop_iteration = None

    # Iterate through the generator
    while True:
        try:
            # 1. Get the next value first
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Calling next(gen)")
            value = next(gen)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Received value index {gen_index}: {value!r}")

            # 2. Check the yielded value *after* getting it, *before* yielding it
            # Use the resolved effective_yield_type
            if effective_yield_type is not Any:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Checking yielded value against {effective_yield_type!r}")
                effective_localns = (localns or {}).copy()
                effective_localns['_func_id'] = func_id
                match, obituary = check_type(value, effective_yield_type, globalns, effective_localns, path=[f"{_RETURN_ANNOTATION}[Yield]"], instance_map=instance_map)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: check_type result for yield: match={match}, details={obituary!r}")
                if not match:
                    # Raise the error HERE, before yielding the bad value
                    _handle_type_error(
                        error_type='yield',
                        func_info=func_info,
                        annotation=effective_yield_type, # What was checked against
                        value=value,
                        obituary=obituary,
                        caller_depth=caller_depth_yield,
                        is_yield_check=True,
                        original_annotation=yield_type # The original TypeVar hint
                    )

            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Yielding value: {value!r}")
            gen_index += 1 # Increment the index after successful yield check
            yield value

        except StopIteration as e:
            # Generator finished, capture return value
            return_value_from_stop_iteration = e.value
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Caught StopIteration. Return value: {return_value_from_stop_iteration!r}")
            # 4. Check return value type if needed (inside the StopIteration handler)
            # Use the resolved effective_ret_type
            if effective_ret_type is not Any:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Checking return value against {effective_ret_type!r}")
                effective_localns = (localns or {}).copy()
                effective_localns['_func_id'] = func_id
                match, obituary = check_type(return_value_from_stop_iteration, effective_ret_type, globalns, effective_localns, path=[f"{_RETURN_ANNOTATION}[Return]"], instance_map=instance_map)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: check_type result for return: match={match}, details={obituary!r}")
                if not match:
                    _handle_type_error(
                        error_type='return',
                        func_info=func_info,
                        annotation=effective_ret_type, # What was checked against
                        value=return_value_from_stop_iteration,
                        obituary=obituary,
                        caller_depth=caller_depth_return,
                        is_yield_check=False, # Explicitly False
                        original_annotation=ret_type # The original TypeVar hint
                    )
            else:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Return type is Any. Skipping check.")
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Breaking loop after StopIteration.")
            return return_value_from_stop_iteration # Return value to signal completion

        # Handle exceptions propagated via throw()
        except GeneratorExit as e:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Caught GeneratorExit. Propagating exception.")
            gen.close()
            raise e
        except Exception as e:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Caught unexpected exception. Propagating exception.")
                try:
                    gen.throw(e)
                except StopIteration as si_after_throw:
                    # If throw() causes StopIteration, handle return value check
                    return_value_from_stop_iteration = si_after_throw.value
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Caught StopIteration after throw. Return value: {return_value_from_stop_iteration!r}")
                    if effective_ret_type is not Any:
                        effective_localns = (localns or {}).copy()
                        effective_localns['_func_id'] = func_id
                        match, obituary = check_type(return_value_from_stop_iteration, effective_ret_type, globalns, effective_localns, path=[f"{_RETURN_ANNOTATION}[Return]"], instance_map=instance_map)
                        if not match:
                            _handle_type_error(
                                error_type='return',
                                func_info=func_info,
                                annotation=effective_ret_type, # What was checked against
                                value=return_value_from_stop_iteration,
                                obituary=obituary,
                                caller_depth=caller_depth_return,
                                is_yield_check=False, # Explicitly False
                                original_annotation=ret_type # The original TypeVar hint
                            )
                    return return_value_from_stop_iteration # Return value to signal completion after throw
                except Exception as e_inner_throw:
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Exception occurred inside generator during throw(): {e_inner_throw!r}")
                    raise e_inner_throw
        finally:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Exiting wrapper finally block.")

async def _diecast_wrap_generator_async(
    agen: AsyncGenerator,
    yield_type: Union[Type, TypeVar],
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    func_info: Dict[str, Any],
    func_id: int,
    instance_map: Optional[Dict[TypeVar, Type]] = None # ADDED
) -> AsyncGenerator:
    """Wraps an asynchronous generator to check yielded values.

    This is now an async generator function itself.

    Args:
        agen: The original async generator object.
        yield_type: The expected type of yielded values.
        globalns: Global namespace for type resolution.
        localns: Local namespace for type resolution (includes _func_id).
        func_info: Dictionary containing information about the decorated function.
        func_id: The base function ID for TypeVar tracking.
        instance_map: Optional map from TypeVar -> resolved Type for generic instance methods.

    Yields:
        Values from the original async generator after type checking.

    Raises:
        TypeError: If a yielded value violates type hints.
    """
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Entering wrapper for func_id={func_id} ('{func_info['func_name']}') with instance_map: {instance_map}") # UPDATED Log
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Expected yield={yield_type!r}")
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Namespaces: globalns keys={list(globalns.keys())!r}, localns={localns!r}")

    effective_yield_type = yield_type
    if instance_map is not None and isinstance(yield_type, TypeVar):
        resolved = instance_map.get(yield_type)
        if resolved is not None and not isinstance(resolved, TypeVar):
            effective_yield_type = resolved
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Resolved yield TypeVar {yield_type!r} to {resolved!r} via instance_map.")

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Effective types: yield={effective_yield_type!r}")

    caller_depth_yield = 3 # wrapper -> anext(agen) -> yield -> user code -> user code's caller
    agen_index = 0
    while True:
        try:
            # 1. Get the next value first
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Calling anext(agen)")
            value = await anext(agen)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Received value index {agen_index}: {value!r}")

            # 2. Check the yielded value *after* getting it, *before* yielding it
            # Use the resolved effective_yield_type
            if effective_yield_type is not Any:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Checking yielded value against {effective_yield_type!r}")
                effective_localns = (localns or {}).copy()
                effective_localns['_func_id'] = func_id
                match, obituary = check_type(value, effective_yield_type, globalns, effective_localns, path=[f"{_RETURN_ANNOTATION}[Yield]"], instance_map=instance_map)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_async: check_type result: match={match}, details={obituary!r}")
                if not match:
                    # Raise the error HERE, before yielding the bad value
                    _handle_type_error(
                        error_type='yield',
                        func_info=func_info,
                        annotation=effective_yield_type, # What was checked against
                        value=value,
                        obituary=obituary,
                        caller_depth=caller_depth_yield,
                        is_yield_check=True,
                        original_annotation=yield_type # The original TypeVar hint
                    )
            # 3. Yield the (now checked) value from this wrapper
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Yielding value: {value!r}")
            agen_index += 1
            yield value

        except StopAsyncIteration:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Caught StopAsyncIteration. Generator finished normally.")
            break # Exit the while loop

        # Handle exceptions propagated via throw()
        except GeneratorExit:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Caught GeneratorExit. Propagating exception.")
            await agen.close()
            raise

        except Exception as e:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Caught unexpected exception. Propagating exception.")
            try:
                await agen.athrow(e)
            except StopAsyncIteration:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Caught StopAsyncIteration after throw. Propagating exception.")
                break
            except Exception as e_inner_throw:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Exception occurred inside generator during throw(): {e_inner_throw!r}")
                raise e_inner_throw
        finally:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Exiting wrapper finally block.")
##-##

## ===== CORE WRAPPERS ===== ##
def _async_gen_caller_wrapper(
    func: Callable,
    sig: inspect.Signature,
    hints: Dict[str, Any],
    globalns: Dict[str, Any],
    func_info: Dict[str, Any],
    instance_map: Optional[Dict[TypeVar, Type]] = None # ADDED parameter
) -> Callable:
    """Wraps the *call* to an async generator function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        potential_instance = None # Initialize before conditional assignment
        func_id = id(func_info['func_object'])
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Entering ASYNC GEN wrapper for func_id={func_id} ('{func_info['func_name']}')")
            _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Args={args!r}, Kwargs={kwargs!r}")

        # Define localns here, potentially capturing closure vars if needed later
        # For now, an empty dict suffices as TypeVars are handled via func_id
        localns = {}

        bound_args = None
        try:
            bound_args = sig.bind(*args, **kwargs)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Arguments bound successfully.")
            # <<< MODIFIED: Pass instance_map >>>
            _check_arguments(sig, hints, bound_args, globalns, localns, func_info, func_id, instance_map=instance_map)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Argument checks passed.")
                _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Calling original async generator function: {func_info['func_name']}")

            # --- Call the original async generator function ---
            # This call returns the async generator object itself.
            original_agen = func(*bound_args.args, **bound_args.kwargs)

            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Original async generator created: {original_agen!r}")
                _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Passing async generator to _check_return_value for yield wrapping.")

            # --- Pass the async generator object for wrapping/checking ---
            # _check_return_value will identify it as an async generator and apply
            # the _diecast_wrap_generator_async wrapper.
            # <<< MODIFIED: Pass instance_map >>>
            wrapped_agen = _check_return_value(original_agen, hints, globalns, localns, func_info, func_id, instance_map=instance_map)

            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Exiting, returning wrapped async generator: {wrapped_agen!r}")
            return wrapped_agen # Return the async generator (potentially wrapped)

        except YouDiedError as e: # Explicitly catch and re-raise YouDiedErrors
             _log.debug(f"TRACE decorator._async_gen_caller_wrapper: Re-raising YouDiedError ({e.cause}).")
             raise e
        except TypeError as e: # Catch binding errors etc.
             _log.error(f"TypeError during DieCast async gen wrapper setup for {func_info['func_name']}: {e!r}", exc_info=True)
             # Try to provide more context if possible
             if "missing" in str(e) or "unexpected" in str(e):
                 cause = 'binding_error'
             else:
                 cause = 'internal_error'
             raise YouDiedError(f"TypeError during argument binding or execution: {e}", obituary=None, cause=cause) from e
        except Exception as e:
            # Log and re-raise any other unexpected exceptions
            _log.error(f"Unexpected error during DieCast async gen wrapper for {func_info['func_name']}: {e!r}", exc_info=True)
            raise YouDiedError(f"Exception during function execution: {e}", obituary=None, cause='internal_error') from e
        finally:
            # Clear TypeVar bindings associated with this specific call context
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._async_gen_caller_wrapper: finally block in argument checker. Clearing bindings for base_func_id={func_id}")
            clear_typevar_bindings(func_id)
    return wrapper

def _sync_wrapper(
    func: Callable,
    sig: inspect.Signature,
    hints: Dict[str, Any],
    globalns: Dict[str, Any],
    func_info: Dict[str, Any],
    instance_map: Optional[Dict[TypeVar, Type]] = None # ADDED parameter
) -> Callable:
    """Internal wrapper for synchronous functions."""
    base_func_id = id(func_info['func_object']) # Use func_object's id

    @functools.wraps(func_info['target_func'])
    def wrapper(*args, **kwargs):
        potential_instance = None # Initialize before conditional assignment
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._sync_wrapper: ENTERING for base_func_id={base_func_id} ('{func_info['func_name']}') with args={args!r}, kwargs={kwargs!r}")

        runtime_localns = {}

        bound_args = None
        result = None
        try:
            ### ----- Bind arguments and determine context ID ----- ###
            try:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                runtime_localns = bound_args.arguments.copy()

                if _log.isEnabledFor(logging.DEBUG):
                     _log.debug(f"TRACE decorator._sync_wrapper: Bound arguments: {bound_args.arguments!r}")
                     _log.debug(f"TRACE decorator._sync_wrapper: Runtime localns: {runtime_localns!r}")
            except TypeError as e:
                _log.warning(f"TypeError binding arguments for {func_info['func_name']}: {e}")
                raise # Re-raise binding errors
            ###-###

            ### ----- Check arguments, passing context_id ----- ###
            if any(p.annotation != inspect.Parameter.empty for p in sig.parameters.values()):
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._sync_wrapper: Calling _check_arguments with context_id={base_func_id}")
                _check_arguments(sig, hints, bound_args, globalns, runtime_localns, func_info, base_func_id, instance_map=instance_map)
            elif _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"TRACE decorator._sync_wrapper: Skipping _check_arguments (no annotated parameters)")
            ###-###

            ### ----- Execute the original function ----- ###
            try:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._sync_wrapper: Executing target_func: {func_info['func_name']}")
                result = func_info['target_func'](*args, **kwargs)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._sync_wrapper: target_func returned: {result!r}")
            except Exception as e:
                _log.error(f"Exception during function execution '{func_info['func_name']}': {e!r}", exc_info=True)
                raise
            ###-###

            ### ----- Check return value, passing context_id ----- ###
            checked_result = result # Initialize with original result
            if _RETURN_ANNOTATION in hints:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._sync_wrapper: Calling _check_return_value with context_id={base_func_id}")
                checked_result = _check_return_value(result, hints, globalns, runtime_localns, func_info, base_func_id, instance_map=instance_map)
                # checked_result now holds original result or wrapped generator
                if _log.isEnabledFor(logging.DEBUG):
                     _log.debug(f"TRACE decorator._sync_wrapper: Return value checked. Result from _check_return_value: {checked_result!r}")
            elif _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"TRACE decorator._sync_wrapper: Skipping _check_return_value (no return annotation)")

            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._sync_wrapper: EXITING NORMALLY for context_id={base_func_id}")
            return checked_result # Return the final (potentially wrapped) result

        finally:
            ### ----- Clean up TypeVar bindings for this specific context ----- ###
            # Ensure context_id is resolved to base_func_id if it's a tuple
            final_context_id = base_func_id
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._sync_wrapper: MAIN finally block. Clearing bindings for ID={final_context_id}")
            clear_typevar_bindings(final_context_id) # Restore call
            ###-###

    wrapper._diecast_id = base_func_id # Store base ID for potential identification
    return wrapper

def _async_wrapper(
    func: Callable,
    sig: inspect.Signature,
    hints: Dict[str, Any],
    globalns: Dict[str, Any],
    func_info: Dict[str, Any],
    instance_map: Optional[Dict[TypeVar, Type]] = None # ADDED parameter
) -> Callable:
    """Internal wrapper for asynchronous functions."""
    base_func_id = id(func_info['func_object'])

    @functools.wraps(func_info['target_func']) # <<< FIX: Wrap target_func
    async def wrapper(*args, **kwargs):
        # Removed misplaced log line that used 'cls' before assignment
        if _log.isEnabledFor(logging.DEBUG):
             _log.debug(f"TRACE decorator._async_wrapper: ENTERING for base_func_id={base_func_id} ('{func_info['func_name']}') with args={args!r}, kwargs={kwargs!r}")

        potential_instance = None # Initialize before conditional assignment
        runtime_localns = {}

        bound_args = None
        result = None
        try:
            ### ----- Bind arguments and determine context ID ----- ###
            try:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                runtime_localns = bound_args.arguments.copy()

                if _log.isEnabledFor(logging.DEBUG):
                     _log.debug(f"TRACE decorator._async_wrapper: Bound arguments: {bound_args.arguments!r}")
                     _log.debug(f"TRACE decorator._async_wrapper: Runtime localns: {runtime_localns!r}")
            except TypeError as e:
                _log.warning(f"TypeError binding arguments for {func_info['func_name']}: {e}")
                raise
            ###-###

            ### ----- Check arguments, passing context_id ----- ###
            if any(p.annotation != inspect.Parameter.empty for p in sig.parameters.values()):
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._async_wrapper: Calling _check_arguments with context_id={base_func_id}")
                _check_arguments(sig, hints, bound_args, globalns, runtime_localns, func_info, base_func_id, instance_map=instance_map)
            elif _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._async_wrapper: Skipping _check_arguments (no annotated parameters)")
            ###-###

            ### ----- Execute the original async function ----- ###
            try:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._async_wrapper: Executing target_func: {func_info['func_name']}")
                result = await func_info['target_func'](*args, **kwargs)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._async_wrapper: target_func returned: {result!r}")
            except Exception as e:
                _log.error(f"Exception during async function execution '{func_info['func_name']}': {e!r}", exc_info=True)
                raise
            ###-###

            ### ----- Check return value, passing context_id ----- ###
            checked_result = result # Initialize
            if _RETURN_ANNOTATION in hints:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._async_wrapper: Calling _check_return_value with context_id={base_func_id}")
                # <<< MODIFIED: Pass instance_map >>>
                checked_result = _check_return_value(result, hints, globalns, runtime_localns, func_info, base_func_id, instance_map=instance_map)
                if _log.isEnabledFor(logging.DEBUG):
                     _log.debug(f"TRACE decorator._async_wrapper: Return value checked. Result from _check_return_value: {checked_result!r}")
            elif _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"TRACE decorator._async_wrapper: Skipping _check_return_value (no return annotation)")

            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._async_wrapper: EXITING NORMALLY for context_id={base_func_id}")
            return checked_result
            ###-###

        finally:
            # --- Clean up TypeVar bindings for this specific context --- #
            # Ensure context_id is resolved to base_func_id if it's a tuple
            final_context_id = base_func_id
            if _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"TRACE decorator._async_wrapper: MAIN finally block. Clearing bindings for ID={final_context_id}")
            clear_typevar_bindings(final_context_id)

    wrapper._diecast_id = base_func_id # Store base ID
    return wrapper
##-##

## ===== CLASS DECORATION LOGIC ===== ##
def _apply_diecast_to_method(
    cls: Type, # MODIFIED: Class object
    name: str, # MODIFIED: Attribute name
    attr: Callable, # MODIFIED: Original attribute value
    # cls_name: str, # REMOVED: Redundant
    instance_map: Optional[Dict[TypeVar, Type]] # Keep for generic path, None for non-generic
) -> None: # MODIFIED: Returns None, modifies class in-place
    """Applies the core diecast wrapping logic to a method.

    Used for non-generic classes and for specialized generic subclasses.
    Modifies the class `cls` by replacing the attribute `name` with
    a wrapped version if applicable.

    Args:
        cls: The class object to modify.
        name: The name of the attribute (method) on the class.
        attr: The original attribute value (the method, classmethod, staticmethod).
        instance_map: The resolved TypeVar map for generic specialization, or None.
    """
    # Correctly inserted log statement
    _log.debug(f"TRACE _apply_diecast_to_method: Entering for method '{name}' on class '{cls.__name__}'. Received instance_map: {instance_map!r}")
    cls_name = cls.__name__
    method_name = name # Use the actual attribute name

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._apply_diecast_to_method: Attempting to wrap '{method_name}' on class '{cls_name}' with instance_map={instance_map}")

    # Check ignore marker on the attribute itself
    if getattr(attr, _DIECAST_MARKER, False):
        _log.debug(f"TRACE decorator._apply_diecast_to_method: Skipping method '{method_name}' (marked with _DIECAST_MARKER)")
        return # Return original if ignored

    target_func = None
    is_classmethod = isinstance(attr, classmethod)
    is_staticmethod = isinstance(attr, staticmethod)

    if inspect.isfunction(attr): # Regular method
        target_func = attr
    elif is_classmethod or is_staticmethod:
        try:
            target_func = attr.__func__
            # Check marker on underlying function too
            if getattr(target_func, _DIECAST_MARKER, False):
                _log.debug(f"TRACE decorator._apply_diecast_to_method: Skipping method '{method_name}' (underlying function marked)")
                return # Return original if underlying func is ignored
        except AttributeError:
            _log.debug(f"TRACE decorator._apply_diecast_to_method: Could not get __func__ for '{method_name}', skipping.")
            return
    else:
        # Not a standard callable method type we handle (could be a descriptor, property, etc.)
        _log.debug(f"TRACE decorator._apply_diecast_to_method: Skipping attribute '{method_name}' (not a function/classmethod/staticmethod).")
        return # Return original if not a method we handle

    if not target_func or not callable(target_func): # Added callable check for safety
        _log.debug(f"TRACE decorator._apply_diecast_to_method: No valid target function identified for '{method_name}', skipping.")
        return # Return original if no target func

    # Check for annotations
    # Use get_resolved_type_hints to handle forward refs etc. within the method
    # We need the function's global namespace here.
    func_globals = getattr(target_func, "__globals__", None)
    if func_globals is None:
        _log.warning(f"Could not get __globals__ for {method_name} in {cls_name}. Type hint resolution may fail. Skipping.")
        return
    # For methods being decorated *on the class definition*, localns is typically not needed
    # or easily available. It's primarily for resolving forward refs *during a call*.
    # Pass None for localns here. get_resolved_type_hints handles this.
    hints = get_resolved_type_hints(target_func, globalns=func_globals, localns=None)

    if not hints:
        _log.debug(f"TRACE decorator._apply_diecast_to_method: Skipping method '{method_name}' (no resolved type hints found).")
        return # Return original if no hints

    # --- Get Signature ---
    try:
        sig = inspect.signature(target_func)
    except (ValueError, TypeError) as e: # Handle cases like builtins without signature or other errors
        _log.warning(f"Could not get signature for {method_name} in {cls_name}: {e!r}. Skipping.")
        return

    # --- Create func_info dictionary ---
    # Pass the *target_func* (the actual underlying function)
    func_info = _get_func_info(target_func)
    # Store the target function itself, as wrappers need it for execution
    func_info['target_func'] = target_func
    # Ensure the ID used by wrappers refers to the target_func
    func_info['func_object'] = target_func
    # Store original name for potential future use (e.g., better error messages)
    # func_info['original_attribute_name'] = method_name
    # Correct the qualname based on class context
    # Use target_func.__name__ as the method's name part
    func_info['func_qualname'] = f"{cls_name}.{target_func.__name__}"
    func_info['func_class_name'] = cls_name

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._apply_diecast_to_method: Applying core wrapper to '{method_name}' (target: '{target_func.__name__}'). func_info={func_info!r}")
        _log.debug(f"TRACE decorator._apply_diecast_to_method: Signature={sig}")
        _log.debug(f"TRACE decorator._apply_diecast_to_method: Hints={hints!r}")
        _log.debug(f"TRACE decorator._apply_diecast_to_method: Globals keys={list(func_globals.keys())!r}")

    # Apply the appropriate core wrapper based on the function type
    # Pass target_func and all required info (sig, hints, globalns, func_info)
    # instance_map is NOT passed here; it's handled internally by check functions.
    wrapped_func = None
    _log.debug(f"TRACE decorator._apply_diecast_to_method: About to call wrapper. instance_map = {instance_map!r}")
    _log.debug(f"TRACE decorator._apply_diecast_to_method: About to call wrapper. instance_map = {instance_map!r}")
    # Pass instance_map down to the core wrappers
    if asyncio.iscoroutinefunction(target_func):
        wrapped_func = _async_wrapper(target_func, sig, hints, func_globals, func_info, instance_map=instance_map)
    elif inspect.isasyncgenfunction(target_func):
        wrapped_func = _async_gen_caller_wrapper(target_func, sig, hints, func_globals, func_info, instance_map=instance_map)
    else: # Regular synchronous function
        wrapped_func = _sync_wrapper(target_func, sig, hints, func_globals, func_info, instance_map=instance_map)

    # Ensure wrapping didn't fail silently (shouldn't happen with current wrappers)
    if wrapped_func is None:
         _log.error(f"Failed to obtain wrapped function for {method_name} in {cls_name}. Skipping.")
         return

    # Re-wrap classmethods and staticmethods correctly
    # Update the attribute on the *class* object (`cls`) using the original `name`
    try:
        final_method = None
        if is_classmethod:
            final_method = classmethod(wrapped_func)
            setattr(cls, name, final_method)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._apply_diecast_to_method: Set classmethod '{name}' on {cls_name}")
        elif is_staticmethod:
            final_method = staticmethod(wrapped_func)
            setattr(cls, name, final_method)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._apply_diecast_to_method: Set staticmethod '{name}' on {cls_name}")
        else: # Regular instance method
            final_method = wrapped_func
            setattr(cls, name, final_method)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._apply_diecast_to_method: Set regular method '{name}' on {cls_name}")

        # --- ADDED: Set marker AFTER successful setattr ---
        if final_method is not None:
            try:
                setattr(final_method, _DIECAST_MARKER, True)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._apply_diecast_to_method: Set marker on final method '{name}' for class {cls_name}")
            except AttributeError:
                 _log.warning(f"Could not set {_DIECAST_MARKER} on final method for {name} in {cls_name} (type: {type(final_method).__name__})")
        # --- END ADDED ---

    except Exception as e:
        _log.error(f"Failed to set wrapped method '{name}' on class {cls_name}: {e!r}", exc_info=True)
        # Decide whether to raise or just log
##-##

## ===== CORE DECORATOR LOGIC ===== ##
def diecast(obj: Union[Callable, Type]) -> Union[Callable, Type]:
    """Applies runtime type checking to a function, method, or class.

    If applied to a function or method:
    - Wraps it to check argument and return types based on annotations.
    - Handles sync/async functions and generators.
    - Respects `@ignore`.

    If applied to a class:
    - Generic Classes: Injects `__class_getitem__` to enable type checking
      on specialized instances (e.g., `MyGeneric[int]()`). Methods are wrapped
      with the resolved type map during specialization.
    - Non-Generic Classes: Iterates through methods defined in the class
      and applies the `diecast` wrapper to each, respecting `@ignore`.

    Returns:
        The wrapped function/method or the modified class.
    """
    if _log.isEnabledFor(logging.DEBUG):
        obj_repr = getattr(obj, '__qualname__', getattr(obj, '__name__', repr(obj)))
        _log.debug(f"TRACE decorator.diecast: Decorator called on: {obj_repr} (type: {type(obj).__name__})")

    if getattr(obj, _DIECAST_MARKER, False):
        _log.debug(f"TRACE decorator.diecast: Object '{obj_repr}' already marked with {getattr(obj, _DIECAST_MARKER)}. Returning original.")
        return obj

    # --- Handle Classes --- #
    if inspect.isclass(obj):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator.diecast: Detected class: '{obj.__name__}'. Delegating to _decorate_class.")
        return _decorate_class(obj) # MODIFIED: Call renamed function

    # --- Handle Functions/Methods --- #
    if not callable(obj):
        # Should ideally not happen if used as a standard decorator
        _log.warning(f"diecast applied to non-callable, non-class object: {obj_repr}. Returning original.")
        return obj

    # --- Prepare common info needed for all wrappers --- #
    target_func = obj
    is_classmethod = isinstance(obj, classmethod)
    is_staticmethod = isinstance(obj, staticmethod)
    
    # If it's a classmethod or staticmethod, get the underlying function
    if is_classmethod or is_staticmethod:
        try:
            target_func = obj.__func__
        except AttributeError:
            _log.error(f"Could not retrieve underlying function for {obj_repr}. Returning original.")
            return obj # Cannot proceed

    # --- Check for __wrapped__ and determine function for checks ---
    func_for_checks = target_func # Default to the function we received/extracted
    original_func = getattr(target_func, '__wrapped__', None)
    if original_func is not None and callable(original_func):
        _log.debug(f"TRACE decorator.diecast: Found '__wrapped__'. Using '{getattr(original_func, '__name__', repr(original_func))}' for signature/hints.")
        func_for_checks = original_func
    # --- END __wrapped__ check ---

    # Get signature, hints, etc. using func_for_checks
    try:
        sig = inspect.signature(func_for_checks)
        globalns = getattr(func_for_checks, "__globals__", None)
        if globalns is None:
            _log.warning(f"Could not get __globals__ for {obj_repr} (checking {func_for_checks!r}). Type hint resolution may fail.")
            globalns = {}
        # Pass None for localns, it's not typically needed at decoration time
        hints = get_resolved_type_hints(func_for_checks, globalns=globalns, localns=None)
        func_info = _get_func_info(func_for_checks) # Get info from the function used for checks
        func_info['target_func'] = target_func # Store the *actual* function to call (the outermost one received)
        func_info['func_object'] = func_for_checks # Store the function used for checks/ID generation
        # Store original decorator type if needed later (optional)
        # func_info['original_decorator'] = 'classmethod' if is_classmethod else 'staticmethod' if is_staticmethod else None

    except (ValueError, TypeError) as e:
        _log.warning(f"Could not get signature or resolve hints for {obj_repr}: {e!r}. Returning original.")
        return obj # Cannot proceed without signature/hints
        
    # --- Select and apply the correct wrapper --- #
    wrapped = None
    if asyncio.iscoroutinefunction(target_func):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator.diecast: Wrapping async function '{func_info['func_name']}'")
        wrapped = _async_wrapper(target_func, sig, hints, globalns, func_info)
    elif inspect.isasyncgenfunction(target_func):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator.diecast: Wrapping async generator function '{func_info['func_name']}'")
        wrapped = _async_gen_caller_wrapper(target_func, sig, hints, globalns, func_info)
    # REMOVED incorrect branch for isgeneratorfunction - handled by else
    else: # Regular synchronous function OR synchronous generator function
        if _log.isEnabledFor(logging.DEBUG):
            log_msg = "sync function/method"
            if inspect.isgeneratorfunction(target_func):
                log_msg = "sync generator function"
            _log.debug(f"TRACE decorator.diecast: Wrapping {log_msg} '{func_info['func_name']}'")
        wrapped = _sync_wrapper(target_func, sig, hints, globalns, func_info)

    # --- Re-apply classmethod/staticmethod if necessary --- #
    final_wrapper = wrapped
    if is_classmethod:
        final_wrapper = classmethod(wrapped)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator.diecast: Re-applied @classmethod to wrapper for '{func_info['func_name']}'")
    elif is_staticmethod:
        final_wrapper = staticmethod(wrapped)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator.diecast: Re-applied @staticmethod to wrapper for '{func_info['func_name']}'")

    # Mark the final wrapper to prevent re-decoration
    if final_wrapper is not None and final_wrapper is not obj: # Check if wrapping occurred
        try:
            setattr(final_wrapper, _DIECAST_MARKER, True)
            if _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"TRACE decorator.diecast: Set marker on final wrapper for {func_info['func_name']}")
        except AttributeError:
             _log.warning(f"Could not set {_DIECAST_MARKER} on final wrapper for {func_info['func_name']} (type: {type(final_wrapper).__name__})")

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator.diecast: Finished wrapping function/method '{func_info['func_name']}'. Returning final wrapper.")
    return final_wrapper # Return the potentially re-decorated final wrapper
##-##

## ===== IGNORE DECORATOR ===== ##
def ignore(func_or_cls: Callable) -> Callable:
    """Decorator to mark a class, function, or method to be ignored by @diecast and mold.

    Args:
        func: The class, function, or method to mark.

    Returns:
        The original class, function, or method marked with an internal flag.
    """
    func_name = getattr(func_or_cls, '__name__', 'unknown')
    if _log.isEnabledFor(logging.DEBUG):
        if inspect.isclass(func_or_cls):
            _log.debug(f"TRACE decorator.ignore: Entering for class='{func_name}' ({func_or_cls!r})")
        else:
            _log.debug(f"TRACE decorator.ignore: Entering for func='{func_name}' ({func_or_cls!r})")
    # Set the single marker to indicate this function should be skipped
    setattr(func_or_cls, _DIECAST_MARKER, True)
    if _log.isEnabledFor(logging.DEBUG):
        if inspect.isclass(func_or_cls):
            _log.debug(f"TRACE decorator.ignore: Set {_DIECAST_MARKER}=True for class '{func_name}'.")
            _log.debug("TRACE decorator.ignore: Exiting, returning original class")
        else:
            _log.debug(f"TRACE decorator.ignore: Set {_DIECAST_MARKER}=True for function '{func_name}'.")
            _log.debug(f"TRACE decorator.ignore: Exiting, returning original function.")
    return func_or_cls
##-##

## ===== INTERNAL HELPERS ===== ##
def _decorate_class(cls: Type) -> Type: # RENAMED function
    """Decorates a class, handling both generic and non-generic cases.

    For Generic classes: Injects __class_getitem__ to handle specialization and
                         wraps methods on the specialized subclass.
    For Non-Generic classes: Directly wraps methods defined on the class.
    """
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._decorate_class: Entering for class '{cls.__name__}'") # UPDATED log message

    if hasattr(cls, "__parameters__") and cls.__parameters__:
        _log.debug(f"Class {cls.__name__} is Generic. Decorating methods.")

        ### ----- Capture original __class_getitem__ before overwriting ----- ###
        original_getitem = getattr(cls, '__class_getitem__', None)
        _log.debug(f"TRACE _decorate_class: Captured original __class_getitem__: {original_getitem!r}")
        if original_getitem is None:
             # If no original, try getting from typing.Generic (or superclass)
             # This is important for standard Generic behavior
             if sys.version_info >= (3, 7):
                 # In 3.7+, Generic itself provides __class_getitem__
                 try:
                     original_getitem = typing.Generic.__dict__['__class_getitem__'].__get__(cls)
                 except (KeyError, AttributeError):
                      _log.warning(f"Could not find fallback __class_getitem__ for {cls.__name__}")
                      # Define a minimal fallback if absolutely necessary, 
                      # though this might indicate an issue with the class itself.
                      def _minimal_fallback_getitem(key):
                          from typing import _GenericAlias
                          return _GenericAlias(cls, key)
                      original_getitem = _minimal_fallback_getitem
             else:
                 # Older Python versions might need _GenericAlias directly
                 def _legacy_fallback_getitem(key):
                     from typing import _GenericAlias
                     return _GenericAlias(cls, key)
                 original_getitem = _legacy_fallback_getitem
        ###-###
                 
        ### ----- Define and inject DieCast __class_getitem__ ----- ###
        @classmethod
        def __class_getitem__(cls_arg, key): # Use cls_arg to avoid clash with outer cls
            """Dynamically create and cache specialized subclass with type map."""
            _log.debug(f"DIECAST_GENERIC: ENTERING injected __class_getitem__ for {cls_arg.__name__} with key={key!r}") # TEMP COMMENT

            # Cache lookup
            cache_key = (cls_arg, key)
            with _SPECIALIZED_CLASS_CACHE_LOCK:
                cached_class = _SPECIALIZED_CLASS_CACHE.get(cache_key)
            if cached_class:
                return cached_class

            # Call original __class_getitem__ (captured above)
            try:
                base_specialized_type = original_getitem(key)
            except TypeError as e:
                _log.error(f"Error calling original __class_getitem__ for {cls_arg.__name__} with key {key!r}: {e}")
                raise
            except Exception as e:
                _log.exception(f"Unexpected error calling original __class_getitem__ for {cls_arg.__name__} with key {key!r}: {e}")
                raise

            # Resolve the type map based on the *original* class parameters and the key
            parameters = None
            if hasattr(cls_arg, "__parameters__") and cls_arg.__parameters__:
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"DIECAST_GENERIC: Found __parameters__ directly on {cls_arg.__name__}: {cls_arg.__parameters__}")
                parameters = cls_arg.__parameters__
            else:
                for base in reversed(cls_arg.__mro__[1:]):
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"DIECAST_GENERIC: Checking base {base!r} for __parameters__")
                    if hasattr(base, "__parameters__") and base.__parameters__:
                        if _log.isEnabledFor(logging.DEBUG):
                            _log.debug(f"DIECAST_GENERIC: Found __parameters__ in base {base.__name__}: {base.__parameters__}")
                        parameters = base.__parameters__
                        break
                    if hasattr(base, "__orig_bases__"):
                        if _log.isEnabledFor(logging.DEBUG):
                            _log.debug(f"DIECAST_GENERIC: Checking __orig_bases__ of base {base.__name__}")
                        for orig_base in base.__orig_bases__:
                            if hasattr(orig_base, "__parameters__") and orig_base.__parameters__:
                                if _log.isEnabledFor(logging.DEBUG):
                                    _log.debug(f"DIECAST_GENERIC: Found __parameters__ in orig_base {orig_base!r}: {orig_base.__parameters__}") 
                                parameters = orig_base.__parameters__
                                break
            args = key if isinstance(key, tuple) else (key,)
            if not parameters:
                _log.warning(f"Class {cls_arg.__name__} has no __parameters__ for specialization mapping.")
                resolved_map = {}
            elif len(parameters) != len(args):
                _log.warning(f"Parameter/Argument count mismatch for {cls_arg.__name__}[{key!r}]")
                resolved_map = {}
            else:
                resolved_map = dict(zip(parameters, args))

            # _log.debug(f"DIECAST_GENERIC: Calculated resolved_map: {resolved_map!r}") # TEMP COMMENT
            specialized_name = f"{cls_arg.__name__}_DiecastSpecialized_{key!r}"

            try:
                specialized_subclass = types.new_class(
                    specialized_name,
                    (base_specialized_type,), 
                    {}, # Pass empty dict for keywords
                    lambda ns: ns.update({'__module__': cls_arg.__module__})
                )
                _log.debug(f"DIECAST_GENERIC: Created specialized subclass: {specialized_subclass.__name__} inheriting from {base_specialized_type!r}") # TEMP COMMENT
                # Correctly inserted log statement after class creation
                _log.debug(f"TRACE _decorate_generic_class.__class_getitem__: Calculated resolved_map: {resolved_map!r} for key {key!r}") # TEMP COMMENT
                setattr(specialized_subclass, '_diecast_type_map', resolved_map) # Map of TypeVar -> Resolved Type
                setattr(specialized_subclass, '_diecast_generic_alias', base_specialized_type) # Store the standard alias it represents
                setattr(specialized_subclass, '_DIECAST_SPECIALIZED_GENERIC', True) # Mark the specialized subclass
                # _log.debug(f"DIECAST_GENERIC: Set _diecast attributes on {specialized_subclass.__name__}") # TEMP COMMENT
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"Created dynamic subclass {specialized_subclass.__name__} with map: {resolved_map}") # TEMP COMMENT
            except Exception as e:
                    _log.exception(f"Failed to create dynamic subclass for {cls_arg.__name__}[{key!r}]: {e}")
                    return base_specialized_type

            # --- REVISED: Wrap methods defined in the ORIGINAL generic class ---
            _log.debug(f"DIECAST_GENERIC: Wrapping methods defined in {cls_arg.__name__} onto {specialized_subclass.__name__}") # TEMP COMMENT
            for name, original_attr in cls_arg.__dict__.items():
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"DIECAST_GENERIC: Checking attribute '{name}' in {cls_arg.__name__}.__dict__") # TEMP COMMENT
                # Check if it's a method-like attribute we should consider
                is_method_like = isinstance(original_attr, (types.FunctionType, staticmethod, classmethod))
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"DIECAST_GENERIC: Attribute '{name}' is {'method-like' if is_method_like else 'not method-like'} in {cls_arg.__name__}.__dict__")
                if not is_method_like:
                    # _log.debug(f"DIECAST_GENERIC: Skipping '{name}' (not method-like in {cls_arg.__name__}.__dict__)") # TEMP COMMENT
                    continue

                # Determine the underlying function and check for ignore marker
                original_target_func = None
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"DIECAST_GENERIC: Checking original_attr {original_attr} for '{name}' in {cls_arg.__name__}.__dict__")
                if inspect.isfunction(original_attr):
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"DIECAST_GENERIC: Found function attribute '{name}' in {cls_arg.__name__}.__dict__")
                    original_target_func = original_attr
                elif isinstance(original_attr, (classmethod, staticmethod)):
                    original_target_func = getattr(original_attr, '__func__', None)

                # Check marker or abstract on the *original* underlying function
                if original_target_func:
                    if getattr(original_target_func, _DIECAST_MARKER, False):
                        _log.debug(f"DIECAST_GENERIC: Skipping method '{name}' on {specialized_subclass.__name__} (original marked with @ignore)") # TEMP COMMENT
                        continue
                    if getattr(original_target_func, '__isabstractmethod__', False):
                        _log.debug(f"DIECAST_GENERIC: Skipping abstract method '{name}' on specialized class {specialized_subclass.__name__}")
                        continue

                # Check marker on the attribute itself (redundant if underlying check works, but safe)
                if getattr(original_attr, _DIECAST_MARKER, False):
                    _log.debug(f"DIECAST_GENERIC: Skipping method '{name}' on {specialized_subclass.__name__} (original attribute already marked)") # TEMP COMMENT
                    continue

                if not hasattr(specialized_subclass, name):
                    setattr(specialized_subclass, name, original_attr)

                # Apply diecast to the method *on the specialized subclass*
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"DIECAST_GENERIC: Applying diecast to method '{name}' on specialized class {specialized_subclass.__name__} using map {resolved_map!r}") # TEMP COMMENT
                _apply_diecast_to_method(specialized_subclass, name, original_attr, instance_map=resolved_map)

            with _SPECIALIZED_CLASS_CACHE_LOCK:
                _SPECIALIZED_CLASS_CACHE[cache_key] = specialized_subclass
            if _log.isEnabledFor(logging.DEBUG) and _SPECIALIZED_CLASS_CACHE.get(cache_key):
                _log.debug(f"TRACE decorator.py _decorate_class.__class_getitem__: Successfully cached specialized class {specialized_name}")

            return specialized_subclass
        ###-###

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE _decorate_class: Setting wrapped __class_getitem__ on {cls.__name__}")
        # Inject the custom __class_getitem__ into the original class (cls)
        # Use setattr for safety, though direct assignment is common
        setattr(cls, '__class_getitem__', __class_getitem__)
        setattr(cls, '_DIECAST_GENERIC', True) # Mark the original generic class
        return cls # Return early for generic classes, wrapping happens in __class_getitem__

    else: # Non-Generic Class: Decorate methods directly on the original class
        _log.debug(f"Class {cls.__name__} is not Generic. Decorating methods directly.")
        # Iterate through attributes defined directly in the class dictionary
        for name, attr in cls.__dict__.items():
            # Determine the underlying function and check for ignore marker
            target_func = None
            is_method_like = False
            if inspect.isfunction(attr):
                target_func = attr
                is_method_like = True
            elif isinstance(attr, (classmethod, staticmethod)):
                target_func = getattr(attr, '__func__', None)
                is_method_like = True
            # Add check for properties if needed in the future

            if not is_method_like or target_func is None:
                _log.debug(f"TRACE _decorate_class: Skipping non-method attribute '{name}' in non-generic class {cls.__name__}")
                continue # Skip non-methods or things we couldn't get a function from

            # Check ignore marker or abstract on the underlying function
            if getattr(target_func, _DIECAST_MARKER, False):
                _log.debug(f"TRACE _decorate_class: Skipping method '{name}' in non-generic class {cls.__name__} (marked with _DIECAST_MARKER)")
                continue
            if getattr(target_func, '__isabstractmethod__', False):
                _log.debug(f"TRACE _decorate_class: Skipping abstract method '{name}' in non-generic class {cls.__name__}")
                continue

            # Check if the method has any type hints (no point wrapping if not)
            try:
                # Use the target_func's globals for resolving hints within the method
                func_globals = getattr(target_func, "__globals__", None)
                if func_globals is None:
                     _log.warning(f"Could not get __globals__ for {cls.__name__}.{name}. Type hint resolution may fail. Skipping.")
                     continue
                hints = get_resolved_type_hints(target_func, globalns=func_globals, localns=None)
                if not hints:
                    _log.debug(f"TRACE _decorate_class: Skipping method '{name}' in non-generic class {cls.__name__} (no type hints)")
                    continue
            except Exception as e:
                 _log.warning(f"Could not resolve type hints for {cls.__name__}.{name}: {e}. Skipping decoration.")
                 continue

            # Apply the diecast wrapper directly to the method on the original class
            # Pass instance_map=None because there's no specialization map
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE _decorate_class: Applying diecast to method '{name}' on non-generic class {cls.__name__}")
            _apply_diecast_to_method(cls, name, attr, instance_map=None)

    _log.debug(f"TRACE decorator._decorate_class: Exiting for class '{cls.__name__}'")
    return cls
##-##
#-#