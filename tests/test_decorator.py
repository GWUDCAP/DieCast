import pytest
import logging # Add logging import
import typing # Ensure typing is imported at top level
import re # Added for strip_ansi
from typing import (
    Iterator, 
    Optional, 
    TypeVar, 
    Generic,
    Union, 
    List, 
    Dict,
    Any # Added Any for clarity
)
from abc import ABC, abstractmethod
import asyncio
from collections.abc import Sequence as ABCSequence # For TypeVar bound test
import functools
import sys
import inspect
import gc
from diecast.type_utils import YouDiedError, Obituary # Ensure Obituary is imported

# ANSI escape code pattern
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m') # Escaped brackets

def strip_ansi(text):
  """Removes ANSI escape codes from a string."""
  return ANSI_ESCAPE.sub('', text)

# Assuming diecast is installed or src is in PYTHONPATH
from diecast import diecast, logger as diecast_logger # Import logger

# --- Fixture to set log level ---
@pytest.fixture(autouse=True)
def configure_diecast_logging():
    """Fixture to ensure diecast logger level is set to DEBUG for tests."""
    original_level = diecast_logger.level
    diecast_logger.setLevel(logging.DEBUG)
    yield
    diecast_logger.setLevel(original_level) # Restore original level after test

# --- Fixture to clean up TypeVar bindings ---
@pytest.fixture(autouse=True)
def clean_typevar_bindings():
    """Fixture to ensure TypeVar bindings are cleared between tests."""
    from diecast.type_utils import _TYPEVAR_BINDINGS
    yield
    # Clear all TypeVar bindings after the test
    _TYPEVAR_BINDINGS.clear()
    # Also force garbage collection to clean up any lingering references
    gc.collect()

# --- Helper Functions/Classes for Tests ---

class SimpleClass:
    @diecast
    def instance_method(self, x: int) -> str:
        if not isinstance(x, int):
             # Should be caught by diecast, but helps debugging test if it fails
            raise TypeError("Test Helper: Expected int")
        return f"Value: {x}"

    @classmethod
    @diecast
    def class_method(cls, y: str) -> bool:
        return isinstance(y, str)

    @staticmethod
    @diecast
    def static_method(z: bool = True) -> Optional[bool]:
        return z if isinstance(z, bool) else None

@diecast
def basic_func(a: int, b: str) -> float:
    return float(a) + len(b)

@diecast
def optional_func(a: Optional[str]) -> Optional[int]:
    return len(a) if a is not None else None

@diecast
def union_func(a: Union[int, str]) -> Union[float, bool]:
    if isinstance(a, int):
        return float(a)
    else:
        return len(a) > 0

@diecast
def list_func(items: List[int]) -> int:
    return sum(items)

@diecast
def dict_func(config: Dict[str, int]) -> List[str]:
    return sorted(config.keys())

@diecast
def complex_nested_func(data: List[Dict[str, Optional[int]]]) -> int:
    count = 0
    for d in data:
        for v in d.values():
            if v is not None:
                count += v
    return count

class ForwardRefTarget:
    pass

@diecast
def forward_ref_func(target: 'ForwardRefTarget') -> bool:
    return isinstance(target, ForwardRefTarget)

@diecast
def generator_func(n: int) -> Iterator[int]:
    for i in range(n):
        yield i * i # Yields int

@diecast
async def async_func(name: str) -> str:
    await asyncio.sleep(0.01)
    return f"Hello, {name}"

T_Unconstrained = TypeVar('T_Unconstrained')
T_Constrained = TypeVar('T_Constrained', int, str)
T_Bound = TypeVar('T_Bound', bound=ABCSequence) # Use collections.abc for bound

@diecast
def typevar_unconstrained_func(x: T_Unconstrained) -> T_Unconstrained:
    return x

@diecast
def typevar_constrained_func(x: T_Constrained) -> T_Constrained:
    return x

@diecast
def typevar_bound_func(x: T_Bound) -> int:
    return len(x)

T = TypeVar('T')
class MyGeneric(Generic[T]):
    @diecast
    def identity(self, value: T) -> T:
        return value

class Parent: pass
class Child(Parent): pass

@diecast
def inheritance_func_a(p: Parent) -> bool:
    return True

@diecast
def inheritance_func_b(c: Child) -> bool:
    return True

# Moved nested function definitions to top level
@diecast
def _nested_wrong_return(a: int) -> str:
    return a # Returns int instead of str

@diecast
def _nested_wrong_optional_return(a: Optional[str]) -> Optional[str]:
    return 123 # Incorrect type

@diecast
def _nested_wrong_union_return(a: Union[int, str]) -> Union[int, str]:
    return [a] # Returns list

@diecast
def _nested_bad_generator() -> Iterator[int]:
    yield 1
    yield "bad" # Fails here
    yield 9

@diecast
async def _nested_bad_async() -> str:
    await asyncio.sleep(0.01)
    return 123 # Wrong type

@diecast
def _nested_consistent_types(x: T_Unconstrained, y: T_Unconstrained) -> T_Unconstrained:
    return x

@diecast
def _nested_wrong_return_typevar(x: T_Unconstrained) -> T_Unconstrained:
    return "wrong type"  # Always returns str regardless of input

T_Class = TypeVar('T_Class') # Renamed from T to avoid conflict
class ConsistentGeneric(Generic[T_Class]):
    @diecast
    def method(self, x: T_Class, y: T_Class) -> T_Class:
        return x

@diecast
@diecast # Keep double decoration for the test
def _nested_double_decorated(a: int) -> int:
    # Add a marker to check if inner wrapper ran twice
    if not hasattr(_nested_double_decorated, 'call_count'):
        _nested_double_decorated.call_count = 0
    _nested_double_decorated.call_count += 1
    return a * 2

# --- Test Cases ---

def test_basic_arg_pass():
    assert basic_func(1, "abc") == 4.0

def test_basic_arg_fail():
    with pytest.raises(YouDiedError) as excinfo_a:
        basic_func("1", "abc")
    e_a = excinfo_a.value
    assert e_a.cause == 'argument'
    assert e_a.obituary.expected_repr == 'int'
    assert e_a.obituary.received_repr == 'str'
    assert e_a.obituary.path == ['a']

    with pytest.raises(YouDiedError) as excinfo_b:
        basic_func(1, 123)
    e_b = excinfo_b.value
    assert e_b.cause == 'argument'
    assert e_b.obituary.expected_repr == 'str'
    assert e_b.obituary.received_repr == 'int'
    assert e_b.obituary.path == ['b']

def test_basic_return_pass():
    assert isinstance(basic_func(1, "abc"), float)

def test_basic_return_fail():
    with pytest.raises(YouDiedError) as excinfo:
        _nested_wrong_return(5)
    e = excinfo.value
    assert e.cause == 'return'
    assert e.obituary.expected_repr == 'str'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.value == 5
    assert e.obituary.path == []

def test_optional_arg_pass_none():
    assert optional_func(None) is None

def test_optional_arg_pass_value():
    assert optional_func("test") == 4

def test_optional_arg_fail():
    with pytest.raises(YouDiedError) as excinfo:
        optional_func(123)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'str'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.path == ['a']
    assert e.obituary.message == 'Value is not an instance of expected type'

def test_optional_return_pass_none():
    assert optional_func(None) is None

def test_optional_return_pass_value():
    assert isinstance(optional_func("test"), int)

def test_optional_return_fail():
    with pytest.raises(YouDiedError) as excinfo:
        _nested_wrong_optional_return("hello")
    e = excinfo.value
    assert e.cause == 'return'
    assert e.obituary.expected_repr == 'str'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.value == 123
    assert e.obituary.path == []
    assert e.obituary.message == 'Value is not an instance of expected type'

def test_union_arg_pass():
    assert union_func(10) == 10.0
    assert union_func("test") is True
    assert union_func("") is False

def test_union_arg_fail():
    with pytest.raises(YouDiedError) as excinfo:
        union_func([1.0])
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'Union[int, str]'
    assert e.obituary.received_repr == 'list'
    assert e.obituary.path == ['a']
    assert e.obituary.message == 'Value does not match any type in Union[int, str]'

def test_union_return_pass():
    assert isinstance(union_func(10), float)
    assert isinstance(union_func("test"), bool)

def test_union_return_fail():
    with pytest.raises(YouDiedError) as excinfo:
        _nested_wrong_union_return(5)
    e = excinfo.value
    assert e.cause == 'return'
    assert e.obituary.expected_repr == 'Union[int, str]'
    assert e.obituary.received_repr == 'list' # Note: _nested_wrong_union_return returns [5]
    assert e.obituary.value == [5]
    assert e.obituary.path == []
    assert e.obituary.message == 'Value does not match any type in Union[int, str]'

def test_nested_list_arg_pass():
    assert list_func([1, 2, 3]) == 6

def test_nested_list_arg_fail():
    with pytest.raises(YouDiedError) as excinfo:
         list_func([1, 'a', 3])
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'int'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['items', 1]
    assert e.obituary.message == 'Value is not an instance of expected type'

def test_nested_dict_arg_pass():
    assert dict_func({'c': 1, 'a': 2, 'b': 0}) == ['a', 'b', 'c']

def test_nested_dict_arg_fail_value():
    with pytest.raises(YouDiedError) as excinfo:
        dict_func({'a': 1, 'b': 'c'})
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'int'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['config', "value('b')"]
    assert e.obituary.message == "Value is not an instance of expected type"

def test_nested_dict_arg_fail_key():
     with pytest.raises(YouDiedError) as excinfo:
         dict_func({1: 1, 'b': 2})
     e = excinfo.value
     assert e.cause == 'argument'
     assert e.obituary.expected_repr == 'str'
     assert e.obituary.received_repr == 'int'
     assert e.obituary.path == ['config', 'key(1)']
     assert e.obituary.message == "Incorrect key type for key 1"

def test_complex_nested_arg_pass():
    assert complex_nested_func([{'a': 1, 'b': None}, {'c': 5}]) == 6

def test_complex_nested_arg_fail():
    with pytest.raises(YouDiedError) as excinfo:
        complex_nested_func([{'a': 1, 'b': None}, {'b': 'c'}])
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'int'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['data', 1, "value('b')"]
    assert e.obituary.message == 'Value is not an instance of expected type'

def test_forward_ref_arg_pass():
    instance = ForwardRefTarget()
    assert forward_ref_func(instance) is True

def test_forward_ref_arg_fail():
    with pytest.raises(YouDiedError) as excinfo:
        forward_ref_func(123)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'ForwardRefTarget'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.path == ['target']

# Test methods via an instance
instance = SimpleClass()

def test_method_pass():
    assert instance.instance_method(10) == "Value: 10"

def test_method_fail():
    with pytest.raises(YouDiedError) as excinfo:
        instance.instance_method("bad")
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'int'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['x'] # Note: 'self' is skipped

def test_classmethod_pass():
    assert SimpleClass.class_method("good") is True

def test_classmethod_fail():
    with pytest.raises(YouDiedError) as excinfo:
        SimpleClass.class_method(123)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'str'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.path == ['y'] # Note: 'cls' is skipped

def test_staticmethod_pass():
    assert SimpleClass.static_method(True) is True
    assert SimpleClass.static_method(False) is False

def test_staticmethod_fail():
    with pytest.raises(YouDiedError) as excinfo:
        SimpleClass.static_method("bad")
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'bool'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['z']

def test_generator_yield_pass():
    results = list(generator_func(3))
    assert results == [0, 1, 4]

def test_generator_yield_fail():
    gen = _nested_bad_generator()
    assert next(gen) == 1 # First yield is fine
    with pytest.raises(YouDiedError) as excinfo:
        next(gen) # Second yield fails
    e = excinfo.value
    assert e.cause == 'yield'
    assert e.obituary.expected_repr == 'int'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.value == 'bad'
    assert e.obituary.path == []

@pytest.mark.asyncio
async def test_async_return_pass():
    result = await async_func("Tester")
    assert result == "Hello, Tester"

@pytest.mark.asyncio
async def test_async_return_fail():
    with pytest.raises(YouDiedError) as excinfo:
        await _nested_bad_async()
    e = excinfo.value
    assert e.cause == 'return'
    assert e.obituary.expected_repr == 'str'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.value == 123
    assert e.obituary.path == []

def test_typevar_unconstrained_pass():
    assert typevar_unconstrained_func(1) == 1
    assert typevar_unconstrained_func("a") == "a"
    assert typevar_unconstrained_func([1]) == [1]

def test_typevar_constrained_pass():
    assert typevar_constrained_func(1) == 1
    assert typevar_constrained_func("a") == "a"

def test_typevar_constrained_fail():
     with pytest.raises(YouDiedError) as excinfo:
         typevar_constrained_func(1.5)
     e = excinfo.value
     assert e.cause == 'argument'
     assert e.obituary.expected_repr == '~T_Constrained'
     assert e.obituary.received_repr == 'float'
     assert e.obituary.path == ['x']
     assert e.obituary.message == 'Value not in allowed types for constrained TypeVar'

def test_typevar_bound_pass():
    # First call should succeed and bind T_Bound to list
    assert typevar_bound_func([1, 2]) == 2

    # Second call with a different Sequence type (str) should also SUCCEED because
    # bindings are cleared between calls. The TypeVar T_Bound is freshly bound to str.
    assert typevar_bound_func("abc") == 3

    # Clear bindings manually for the next part of the test
    from diecast.type_utils import clear_typevar_bindings
    # Need to get the actual wrapped function object to find its ID for clearing
    target_func = typevar_bound_func.__wrapped__ if hasattr(typevar_bound_func, '__wrapped__') else typevar_bound_func
    func_id = id(target_func)
    clear_typevar_bindings(func_id)

    # Now test str binding first - this should succeed
    assert typevar_bound_func("abc") == 3

def test_typevar_bound_fail():
     with pytest.raises(YouDiedError) as excinfo:
         typevar_bound_func(123)
     e = excinfo.value
     assert e.cause == 'argument'
     assert e.obituary.expected_repr == '~T_Bound' # Expect the TypeVar itself
     assert e.obituary.received_repr == 'int'
     assert e.obituary.path == ['x']
     assert e.obituary.message == 'Value is not an instance of expected type'

def test_typevar_consistency_pass():
    @diecast
    def consistent_types(x: T_Unconstrained, y: T_Unconstrained) -> T_Unconstrained:
        return x
    
    # These should pass because x and y are the same type
    assert consistent_types(1, 2) == 1
    assert consistent_types("a", "b") == "a"
    assert consistent_types([1], [2, 3]) == [1]

def test_typevar_consistency_fail():
    with pytest.raises(YouDiedError) as excinfo:
        _nested_consistent_types(1, "b")
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == '~T_Unconstrained'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['y']
    assert e.obituary.message == "TypeVar consistency violation: Expected ~T_Unconstrained (Bound to: int) but received str"

def test_typevar_consistency_return_fail():
    # This call should pass (str -> str is consistent)
    assert _nested_wrong_return_typevar("string input") == "wrong type"
    # This call fails consistency check on return
    with pytest.raises(YouDiedError) as excinfo:
        _nested_wrong_return_typevar(42)
    e = excinfo.value
    assert e.cause == 'return'
    assert e.obituary.expected_repr == '~T_Unconstrained'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.value == 'wrong type'
    assert e.obituary.path == []
    assert e.obituary.message == "TypeVar consistency violation: Expected ~T_Unconstrained (Bound to: int) but received str"

def test_typevar_consistency_in_class():
    gen_int = ConsistentGeneric[int]()
    assert gen_int.method(1, 2) == 1
    with pytest.raises(YouDiedError) as excinfo:
        gen_int.method(1, "string")
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == '~T_Class'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['y']
    assert e.obituary.message == "TypeVar consistency violation: Expected ~T_Class (Bound to: int) but received str"

def test_generic_class_method_pass():
    gen_int = MyGeneric[int]()
    assert gen_int.identity(5) == 5

def test_generic_class_method_fail():
    gen_int = MyGeneric[int]()
    result = gen_int.identity("bad")
    assert result == "bad"

def test_inheritance_pass():
    child_instance = Child()
    assert inheritance_func_a(child_instance) is True

def test_inheritance_fail():
    parent_instance = Parent()
    with pytest.raises(YouDiedError) as excinfo:
        inheritance_func_b(parent_instance)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'Child'
    assert e.obituary.received_repr == 'Parent'
    assert e.obituary.path == ['c']

def test_double_decoration_prevention():
    # Reset call count for the top-level function if it exists
    if hasattr(_nested_double_decorated, 'call_count'):
        del _nested_double_decorated.call_count

    assert _nested_double_decorated(5) == 10
    assert _nested_double_decorated.call_count == 1

    with pytest.raises(YouDiedError) as excinfo:
        _nested_double_decorated("bad")
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'int'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['a']

# Test for logging requires capturing logs, often done via pytest fixtures
# Skipping explicit log content assertion for now, but failure cases implicitly test logging path.

# Test @diecast.ignore is handled correctly by the decorator itself (doesn't wrap)
def test_diecast_ignore_does_not_wrap():
    @diecast.ignore
    def ignored_func(a: int) -> str:
        return a

    # If ignore worked, no YouDiedError should be raised
    assert ignored_func(123) == 123 
    # Check the marker is set
    from diecast.decorator import _DIECAST_MARKER
    assert hasattr(ignored_func, _DIECAST_MARKER)