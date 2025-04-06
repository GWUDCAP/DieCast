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

# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
import collections.abc
from typing import (
    AsyncGenerator, AsyncIterator, Generator, Coroutine, Iterator, Callable,
    Optional, Final, Type, Dict, List, Any
)
import functools
import inspect
import logging
import typing

## ===== LOCAL ===== ##
from .error_utils import (
    generate_return_error_message, generate_arg_error_message,
    _get_caller_info
)
from .type_utils import (
    get_resolved_type_hints, clear_typevar_bindings,
    YouDiedError, check_type 
)
from .config import (
    _RETURN_ANNOTATION, _DIECAST_MARKER,
    _SELF_NAMES
)
from .logging import _log


# ===== GLOBALS ===== #

## ===== EXPORTS ===== ## 
__all__: Final[List[str]] = ['diecast', 'ignore']

# ===== FUNCTIONS ===== #

## ===== TYPE ERROR HANDLING ===== ##
def _handle_type_error(
    error_type: str, # 'argument', 'return', 'yield'
    func_info: Dict[str, Any],
    annotation: Any,
    value: Any,
    obituary: Optional[Any], # Should be Obituary object
    caller_depth: int,
    param: Optional[inspect.Parameter] = None, # For arg errors
    arg_index: Optional[int] = None, # For arg errors
    is_kwarg: Optional[bool] = None, # For arg errors
    is_yield_check: bool = False # Distinguish yield from return
) -> None:
    """Centralized function to handle type errors: get info, generate message, raise."""
    if _log.isEnabledFor(logging.DEBUG):
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
            caller_info=caller_info, obituary=obituary
        )
    elif error_type == 'return' or error_type == 'yield':
        error_msg = generate_return_error_message(
            func_name=func_info['func_name'], func_module=func_info['func_module'], func_lineno=func_info['func_lineno'],
            func_class_name=func_info.get('func_class_name'),
            annotation=annotation, value=value,
            caller_info=caller_info, obituary=obituary,
            is_yield_value=is_yield_check # Use the flag passed in
        )
    else:
        # Fallback for unknown error types
        _log.error(f"_handle_type_error called with unknown error_type: {error_type}")
        error_msg = f"Unknown type check error ({error_type}) in {func_info['func_name']}"

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._handle_type_error: Generated error message (len={len(error_msg)}).")
    # Raise the correct exception with cause and obituary
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._handle_type_error: Raising YouDiedError with obituary: {obituary!r}")
    raise YouDiedError(error_msg, obituary=obituary, cause=error_type)

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


## ===== ARGUMENT CHECKING ===== ##
def _check_arguments(
    sig: inspect.Signature,
    hints: Dict[str, Any],
    bound_args: inspect.BoundArguments,
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    func_info: Dict[str, Any]
) -> None:
    """Check function arguments against type hints using the bound arguments.

    Uses `check_type` for validation and raises TypeError on mismatch.
    Relies on `_get_caller_info` and `generate_arg_error_message` for errors.

    Args:
        sig: The inspect.Signature object for the function.
        hints: The dictionary of resolved type hints.
        bound_args: The inspect.BoundArguments object containing call arguments.
        globalns: Global namespace for type resolution.
        localns: Local namespace (may include _func_id for TypeVars).
        func_info: Dictionary with function details.

    Raises:
        TypeError: If any argument fails its type check.
    """
    func_id = id(func_info['func_object'])
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_arguments: Entering for func_id={func_id} ('{func_info['func_name']}')")
        _log.debug(f"TRACE decorator._check_arguments: Signature: {sig}")
        _log.debug(f"TRACE decorator._check_arguments: Hints: {hints!r}")
        _log.debug(f"TRACE decorator._check_arguments: Bound Args: args={bound_args.args!r}, kwargs={bound_args.kwargs!r}")
        _log.debug(f"TRACE decorator._check_arguments: Namespaces: globalns keys={list(globalns.keys())!r}, localns={localns!r}")

    # Apply defaults to ensure all arguments have a value
    bound_args.apply_defaults()

    caller_depth = 3 # _sync/_async_wrapper -> call -> _check_arguments -> check_type -> caller

    for i, (name, param) in enumerate(sig.parameters.items()):
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_arguments: Checking param #{i}: name='{name}', Parameter={param}")

        # Skip self/cls arguments implicitly
        if i == 0 and name in _SELF_NAMES:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_arguments: Skipping parameter '{name}' (assuming self/cls).")
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

        # Prepare local namespace with function ID for TypeVar tracking
        effective_localns = (localns or {}).copy()
        effective_localns['_func_id'] = func_id
        # Remove specific argument value from localns for check_type to avoid self-referencing issues
        effective_localns.pop(name, None)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_arguments: Effective localns for check_type: {effective_localns!r}")

        # --- Handle different parameter kinds --- #
        if param.kind == param.VAR_POSITIONAL: # *args
            if name in bound_args.arguments:
                args_tuple = bound_args.arguments[name]
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: Checking *args parameter '{name}'. Value: {args_tuple!r}, Element Annotation: {annotation!r}")
                if not isinstance(args_tuple, tuple):
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.warning(f"Expected tuple for *args parameter '{name}', but got {type(args_tuple)}. Skipping check.")
                    continue
                for idx, arg_value in enumerate(args_tuple):
                    path = [f"{name}[{idx}]"] # Construct path like args[0], args[1]
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._check_arguments: Checking *args element at index {idx}: value={arg_value!r} against {annotation!r} at path {path!r}")
                    match, obituary = check_type(arg_value, annotation, globalns, effective_localns, path=path)
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._check_arguments: check_type result: match={match}, details={obituary!r}")
                    if not match:
                        _handle_type_error(
                            error_type='argument',
                            func_info=func_info,
                            annotation=annotation,
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
                    _log.debug(f"TRACE decorator._check_arguments: Checking **kwargs parameter '{name}'. Value: {kwargs_dict!r}, Value Annotation: {annotation!r}")
                if not isinstance(kwargs_dict, dict):
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.warning(f"Expected dict for **kwargs parameter '{name}', but got {type(kwargs_dict)}. Skipping check.")
                    continue
                for kwarg_key, kwarg_value in kwargs_dict.items():
                    path = [f"{name}[{kwarg_key!r}]" ] # Construct path like kwargs['key']
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._check_arguments: Checking **kwargs value for key '{kwarg_key}': value={kwarg_value!r} against {annotation!r} at path {path!r}")
                        _log.debug(f"TRACE decorator._check_arguments [**kwargs]: BEFORE check_type for key='{kwarg_key}'.")
                    match, obituary = check_type(kwarg_value, annotation, globalns, effective_localns, path=path)
                    if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._check_arguments [**kwargs]: AFTER check_type for key='{kwarg_key}'. Result: match={match}, details={obituary!r}")
                        _log.debug(f"TRACE decorator._check_arguments: check_type result: match={match}, details={obituary!r}")
                    if not match:
                        _handle_type_error(
                            error_type='argument',
                            func_info=func_info,
                            annotation=annotation, # Annotation for the values
                            value=kwarg_value,
                            obituary=obituary,
                            caller_depth=caller_depth,
                            param=param, # Pass the **kwargs param itself
                            # We need to indicate it's a kwarg item, maybe pass kwarg_key?
                            # Let's pass param name and is_kwarg=True; error msg generator needs update
                            # Current error generator uses param.name which is just 'kwargs'
                            # This needs more thought on how best to represent **kwargs failure origin
                            # For now, use existing logic, but this is imperfect.
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
                    _log.debug(f"TRACE decorator._check_arguments: Checking value={value!r} against annotation={annotation!r}")
                match, obituary = check_type(value, annotation, globalns, effective_localns, path=[name])
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._check_arguments: check_type result: match={match}, details={obituary!r}")
                if not match:
                    is_kwarg = name in bound_args.kwargs
                    _handle_type_error(
                        error_type='argument',
                        func_info=func_info,
                        annotation=annotation,
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

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_arguments: All argument checks passed for func_id={func_id} ('{func_info['func_name']}'). Exiting.")


## ===== RETURN VALUE CHECKING ===== ##
def _check_return_value(
    result: Any,
    hints: Dict[str, Any],
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    func_info: Dict[str, Any],
    is_async_generator: bool = False
) -> Any:
    """Checks the return value type against annotations or wraps generators.

    Handles regular return values, sync/async generators, and coroutine results.
    Uses `check_type` for validation and `_analyze_stack_and_raise` on failure.

    Args:
        result: The value returned by the decorated function.
        hints: The dictionary of resolved type hints for the function.
        globalns: Global namespace for type resolution.
        localns: Local namespace for type resolution (may include _func_id).
        func_info: Dictionary containing information about the decorated function.
        is_async_generator: Flag indicating if the result is from an async generator.

    Returns:
        The original result if the check passes, or a wrapped generator.

    Raises:
        TypeError: If the return value fails its type check.
    """
    func_id = id(func_info['func_object'])
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: Entering for func_id={func_id} ('{func_info['func_name']}')")
        _log.debug(f"TRACE decorator._check_return_value: Result={result!r}, Type={type(result).__name__}, Hints={hints!r}")
        _log.debug(f"TRACE decorator._check_return_value: Namespaces: globalns keys={list(globalns.keys())!r}, localns={localns!r}")
    
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

    origin = typing.get_origin(return_annotation)
    args_return = typing.get_args(return_annotation)
    caller_depth = 2

    # Prepare local namespace with function ID for TypeVar tracking
    effective_localns = (localns or {}).copy()
    effective_localns['_func_id'] = func_id
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: Effective localns for check_type: {effective_localns!r}")

    # --- Generator Handling ---
    is_sync_gen_hint = origin is Generator or origin is Iterator
    # Need to check actual result type *before* trying to wrap
    is_sync_gen_result = isinstance(result, collections.abc.Iterator) 
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: Sync Generator Check: hint={is_sync_gen_hint}, result={is_sync_gen_result}")

    if is_sync_gen_hint or (is_sync_gen_result and not is_sync_gen_hint): # Handle cases where hint isn't Generator but result is
        yield_type = args_return[0] if args_return else Any
        ret_type = args_return[2] if len(args_return) > 2 else Any
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: Sync Gen Types: yield={yield_type!r}, return={ret_type!r}")

        if not is_sync_gen_result:
             if _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"TRACE decorator._check_return_value: Mismatch: Hinted Generator/Iterator but result is not an Iterator ({type(result).__name__}). Raising YouDiedError.")
                 # Type hint was Generator/Iterator, but result wasn't one
                 # Performance: Only call _get_caller_info and check_type if check fails
                 match_ignored, obituary = check_type(result, return_annotation, globalns, effective_localns)
                 _handle_type_error(
                    error_type='return',
                    func_info=func_info,
                    annotation=return_annotation, # The overall hint Generator[...]
                    value=result,
                    obituary=obituary,
                    caller_depth=caller_depth,
                    is_yield_check=False
                 )

        # Wrap the generator if there are types to check
        if yield_type is not Any or ret_type is not Any:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Wrapping sync generator.")
            wrapped_gen = _diecast_wrap_generator_sync(result, yield_type, ret_type, globalns, effective_localns, func_info)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Exiting, returning wrapped sync generator.")
            return wrapped_gen
        else:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Sync generator yield/return types are Any. Returning original generator.")
                _log.debug(f"TRACE decorator._check_return_value: Exiting, returning original result.")
            return result

    # --- Async Generator Handling ---
    is_async_gen_hint = origin is AsyncGenerator or origin is AsyncIterator
    # Need to check actual result type *before* trying to wrap
    is_async_gen_result = isinstance(result, collections.abc.AsyncIterator)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: Async Generator Check: hint={is_async_gen_hint}, result={is_async_gen_result}")

    if is_async_gen_hint or (is_async_gen_result and not is_async_gen_hint):
        yield_type = args_return[0] if args_return else Any
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: Async Gen Types: yield={yield_type!r}")
        
        if not is_async_gen_result:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Mismatch: Hinted AsyncGenerator/Iterator but result is not an AsyncIterator ({type(result).__name__}). Raising YouDiedError.")
                # Type hint was AsyncGenerator/Iterator, but result wasn't one
                # Performance: Only call _get_caller_info and check_type if check fails
                match_ignored, obituary = check_type(result, return_annotation, globalns, effective_localns)
                _handle_type_error(
                    error_type='return',
                    func_info=func_info,
                    annotation=return_annotation, # The overall hint AsyncGenerator[...]
                    value=result,
                    obituary=obituary,
                    caller_depth=caller_depth,
                    is_yield_check=False
                )

        # Wrap the async generator if yield type needs checking
        if yield_type is not Any:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Wrapping async generator.")
            # FIX: Directly call the async generator wrapper with the result (agen object)
            wrapped_agen = _diecast_wrap_generator_async(result, yield_type, globalns, effective_localns, func_info)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Exiting, returning wrapped async generator.")
            return wrapped_agen # Return the wrapped async generator
        else:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._check_return_value: Async generator yield type is Any. Returning original async generator.")
                _log.debug(f"TRACE decorator._check_return_value: Exiting, returning original result.")
            return result # Return the original async generator object

    # --- Coroutine Result Handling (Check awaited result's type) ---
    # Note: The actual result passed here IS the awaited result from the wrapper
    check_against_type = return_annotation
    is_coroutine_result_check = False
    if origin is Coroutine and args_return:
        # If the hint is Coroutine[X,Y,Z], we check the result against Z
        check_against_type = args_return[-1] # Coroutine result type is the last arg
        is_coroutine_result_check = True
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: Hint is Coroutine. Checking result against inner type: {check_against_type!r}")
        if check_against_type is Any:
             if _log.isEnabledFor(logging.DEBUG):
                 _log.debug(f"TRACE decorator._check_return_value: Coroutine inner type is Any. Skipping check.")
                 _log.debug(f"TRACE decorator._check_return_value: Exiting, returning original result.")
             return result
    else:
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._check_return_value: Standard return value check against: {check_against_type!r}")


    # --- Standard Return Value Check ---
    match, obituary = check_type(result, check_against_type, globalns, effective_localns)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._check_return_value: check_type result: match={match}, details={obituary!r}")

    if not match:
        _handle_type_error(
            error_type='return',
            func_info=func_info,
            annotation=check_against_type, # The type we actually checked against
            value=result,
            obituary=obituary,
            caller_depth=caller_depth,
            is_yield_check=False
        )

    # Check passed
    _log.debug(f"TRACE decorator._check_return_value: Type check PASSED for return value.")
    _log.debug(f"TRACE decorator._check_return_value: Exiting, returning original result.")
    return result

## ===== GENERATOR WRAPPERS ===== ##
def _diecast_wrap_generator_sync(
    gen: Generator,
    yield_type: Type,
    ret_type: Type,
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    func_info: Dict[str, Any]
) -> Generator:
    """Wraps a synchronous generator to check yielded values and the final return value.

    Args:
        gen: The original generator object.
        yield_type: The expected type of yielded values.
        ret_type: The expected type of the generator's return value.
        globalns: Global namespace for type resolution.
        localns: Local namespace for type resolution (includes _func_id).
        func_info: Dictionary containing information about the decorated function.

    Yields:
        Values from the original generator after type checking.

    Returns:
        The final return value of the generator after type checking.

    Raises:
        TypeError: If a yielded value or the return value violates type hints.
    """
    func_id = id(func_info['func_object'])
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Entering wrapper for func_id={func_id} ('{func_info['func_name']}')")
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Expected yield={yield_type!r}, return={ret_type!r}")
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Namespaces: globalns keys={list(globalns.keys())!r}, localns={localns!r}")
    
    caller_depth_yield = 3 # wrapper -> next(gen) -> yield -> user code -> user code's caller
    caller_depth_return = 2 # wrapper -> return -> caller
    gen_index = 0
    try:
        while True:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Calling next(gen) for index {gen_index}")
            # Get next value from the generator
            try:
                value = next(gen)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Received value: {value!r}")
            except StopIteration as e:
                 _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Caught StopIteration.")
                 # Check return value and properly return it (PEP 479)
                 return_value = e.value
                 if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Generator finished. Return value: {return_value!r}")
                 if ret_type is not Any:
                     _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Checking return value against {ret_type!r}")
                     match, obituary = check_type(return_value, ret_type, globalns, localns)
                     if _log.isEnabledFor(logging.DEBUG):
                        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: check_type result: match={match}, details={obituary!r}")
                     if not match:
                         _handle_type_error(
                            error_type='return',
                            func_info=func_info,
                            annotation=ret_type,
                            value=return_value,
                            obituary=obituary,
                            caller_depth=caller_depth_return,
                            is_yield_check=False
                         )
                 
                 # Return the value directly rather than re-raising StopIteration (PEP 479)
                 if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Returning value: {return_value!r}")
                 return return_value
            except Exception as e_inner:
                 _log.error(f"TRACE decorator._diecast_wrap_generator_sync: Exception during next(gen): {e_inner!r}", exc_info=True)
                 raise # Re-raise other exceptions immediately

            # Check yielded value type if needed
            if yield_type is not Any:
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Checking yielded value against {yield_type!r}")
                match, obituary = check_type(value, yield_type, globalns, localns)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: check_type result: match={match}, details={obituary!r}")
                if not match:
                    _handle_type_error(
                        error_type='yield',
                        func_info=func_info,
                        annotation=yield_type,
                        value=value,
                        obituary=obituary,
                        caller_depth=caller_depth_yield,
                        is_yield_check=True
                    )
            else:
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Yield type is Any. Skipping check.")
            
            # Yield the checked value
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Yielding value: {value!r}")
            yield value
            gen_index += 1

    except YouDiedError as e: # Explicitly catch and re-raise YouDiedErrors
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Re-raising YouDiedError.")
        raise e
    except TypeError as e: # Explicitly catch and re-raise other TypeErrors (e.g., from isinstance checks)
         _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Re-raising TypeError.")
         raise e
    except Exception as e:
        # Log and re-raise any other unexpected exceptions during generation
        _log.error(f"Unexpected error during DieCast sync generator wrapper for {func_info['func_name']}: {e!r}", exc_info=True)
        _log.debug(f"TRACE decorator._diecast_wrap_generator_sync: Exiting wrapper due to unexpected exception.")
        # For internal errors, obituary might not be relevant or available
        raise YouDiedError(f"Exception during function execution: {e}", obituary=None, cause='internal_error') from e

async def _diecast_wrap_generator_async(
    agen: AsyncGenerator,
    yield_type: Type,
    globalns: Dict[str, Any],
    localns: Optional[Dict[str, Any]],
    func_info: Dict[str, Any]
) -> AsyncGenerator:
    """Wraps an asynchronous generator to check yielded values.

    This is now an async generator function itself.

    Args:
        agen: The original async generator object.
        yield_type: The expected type of yielded values.
        globalns: Global namespace for type resolution.
        localns: Local namespace for type resolution (includes _func_id).
        func_info: Dictionary containing information about the decorated function.

    Yields:
        Values from the original async generator after type checking.

    Raises:
        TypeError: If a yielded value violates type hints.
    """
    func_id = id(func_info['func_object'])
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Entering wrapper for func_id={func_id} ('{func_info['func_name']}')")
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Expected yield={yield_type!r}")
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Namespaces: globalns keys={list(globalns.keys())!r}, localns={localns!r}")

    caller_depth_yield = 3 # wrapper -> anext(agen) -> yield -> user code -> user code's caller
    agen_index = 0
    try:
        # Iterate through the original async generator
        async for value in agen:
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Received value index {agen_index}: {value!r}")
            # Check yielded value type if needed
            if yield_type is not Any:
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Checking yielded value against {yield_type!r}")
                # Prepare local namespace with function ID for TypeVar tracking for this specific check
                effective_localns = (localns or {}).copy()
                effective_localns['_func_id'] = func_id
                match, obituary = check_type(value, yield_type, globalns, effective_localns)
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(f"TRACE decorator._diecast_wrap_generator_async: check_type result: match={match}, details={obituary!r}")
                if not match:
                    _handle_type_error(
                        error_type='yield',
                        func_info=func_info,
                        annotation=yield_type,
                        value=value,
                        obituary=obituary,
                        caller_depth=caller_depth_yield,
                        is_yield_check=True
                    )
            else:
                 _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Yield type is Any. Skipping check.")

            # Yield the checked value from this wrapper
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Yielding value: {value!r}")
            yield value
            agen_index += 1

    except StopAsyncIteration:
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Caught StopAsyncIteration. Generator finished normally.")
        # Async generator finished normally.
        pass # Just finish cleanly
    except YouDiedError: # Explicitly catch and re-raise YouDiedErrors
        _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Re-raising YouDiedError.")
        raise
    except Exception as e:
        # Log and re-raise any other exceptions during async generation
        # Avoid f-string formatting in error log if possible, use args
        _log.error("Unexpected error during DieCast async generator wrapper for %s: %r", func_info['func_name'], e, exc_info=True)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Exiting wrapper due to unexpected exception.")
        # For internal errors, obituary might not be relevant or available
        raise YouDiedError(f"Exception during function execution: {e}", obituary=None, cause='internal_error') from e
    finally:
        # Ensure TypeVar bindings are cleared when the generator is exhausted or an error occurs
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Clearing TypeVar bindings in finally block for func_id={func_id}")
        clear_typevar_bindings(func_id)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"TRACE decorator._diecast_wrap_generator_async: Exiting wrapper finally block.")


## ===== CORE WRAPPERS ===== ##
# Note: These are defined *inside* the diecast decorator
# Their definitions are omitted here as they are part of diecast's implementation.
# def _sync_wrapper(...):
# async def _async_wrapper(...):
# async def _async_gen_caller_wrapper(...):

## ===== PUBLIC DECORATORS ===== ##
def ignore(func: Callable) -> Callable:
    """Decorator to mark a function or method to be ignored by @diecast and mold.
    # ... existing code ...
    """
    # ... existing code ...

def diecast(func: Callable) -> Callable:
    """Decorator to enforce type hints at runtime for function/method arguments and return values.
    # ... existing code ...
    """
    # ... existing code ...
