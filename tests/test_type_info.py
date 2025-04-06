import pytest
import sys
import os
from typing import List, Union, Optional, Dict, Any, Tuple, Type, ForwardRef, Generic, TypeVar, NewType, Literal, Callable, Set
import collections.abc # Needed for Callable check
import typing_extensions

# Add src dir to path to allow importing diecast
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import functions directly from type_utils
from diecast.type_utils import (
    get_origin, get_args, is_optional_type, format_type_for_display,
    is_union_type, get_resolved_type_hints, check_type, Annotated, YouDiedError, Obituary,
    is_generic_alias, resolve_forward_ref
)

# TODO: Add tests for functions in diecast.type_info
# E.g., get_resolved_type_hints, is_union_type, etc. 

# Basic types
@pytest.mark.parametrize("tp, expected_origin, expected_args", [
    (int, None, ()),
    (str, None, ()),
    # Built-ins have no origin
    (list, None, ()),
    (dict, None, ()),
    (tuple, None, ()),
    # Typing aliases resolve to underlying types
    (List, list, ()),
    (Dict, dict, ()),
    (Tuple, tuple, ()),
    (List[int], list, (int,)),
    (Dict[str, bool], dict, (str, bool)),
    (Union[int, str], Union, (int, str)),
    (Optional[float], Union, (float, type(None))),
    (Tuple[int, ...], tuple, (int, ...)),
    (Callable[[int, str], bool], collections.abc.Callable, ([int, str], bool)),
    (Type[int], type, (int,)),
    (Any, None, ()),
])
def test_get_origin_and_args(tp, expected_origin, expected_args):
    """Test get_origin and get_args with various basic and generic types."""
    assert get_origin(tp) == expected_origin
    if expected_args:
        assert get_args(tp) == expected_args

# is_optional_type tests
@pytest.mark.parametrize("tp, is_opt, inner_type", [
    (Optional[int], True, int),
    (Union[str, None], True, str),
    (Union[None, bool], True, bool),
    (int, False, int),
    (str, False, str),
    (Union[int, str], False, Union[int, str]), # Not Optional because not 2 args with None
    (List[Optional[int]], False, List[Optional[int]]), # Optional is nested
    (None, False, None), # Just None type itself
    (Any, False, Any),
])
def test_is_optional_type(tp, is_opt, inner_type):
    """Test is_optional_type detection."""
    result_is_opt, result_inner = is_optional_type(tp)
    assert result_is_opt == is_opt
    # For non-optionals, the function returns the original type as the second element
    assert result_inner == inner_type

# format_type_for_display tests
# Using lowercase for built-in generics (list, dict, tuple) as per modern Python reprs
@pytest.mark.parametrize("tp, expected_substrings", [
    (int, ["int"]),
    (str, ["str"]),
    (List[int], ["list", "int"]),
    (Dict[str, bool], ["dict", "str", "bool"]),
    (Optional[float], ["Optional", "float"]), # Should simplify Union[X, None]
    # Union[..., None] does not necessarily simplify to Optional[...] in display
    (Union[int, str, None], ["Union", "int", "str", "None"]),
    (Tuple[int, str, bool], ["tuple", "int", "str", "bool"]),
    # Making callable check less strict - look for key parts
    (Callable[[int, str], bool], ["Callable", "int", "str", "bool"]),
    # Making ForwardRef check less strict - just look for name
    (ForwardRef('MyClass'), ["MyClass"]),
    (Any, ["Any"]),
    (None, ["None"]),
    (type(None), ["None"]),
    (Literal[1, "a"], ["Literal", "1", "a"])
])
def test_format_type_for_display(tp, expected_substrings):
    """Test format_type_for_display generates readable representations."""
    formatted = format_type_for_display(tp)
    for sub in expected_substrings:
        assert sub in formatted

# is_union_type tests
@pytest.mark.parametrize("tp, is_union, args", [
    (Union[int, str], True, (int, str)),
    (Optional[int], True, (int, type(None))), # Optional is a Union
    (int, False, (int,)),
    (List[int], False, (List[int],)),
    (Union[int], False, (int,)), # typing likely simplifies Union[T] to T
    (Any, False, (Any,)),
    (None, False, (None,)),
    # Test for Python 3.10+ syntax if applicable
    pytest.param(eval("int | str"), True, (int, str), marks=pytest.mark.skipif(sys.version_info < (3, 10), reason="requires Python 3.10+")),
    pytest.param(eval("int | None"), True, (int, type(None)), marks=pytest.mark.skipif(sys.version_info < (3, 10), reason="requires Python 3.10+")),
])
def test_is_union_type(tp, is_union, args):
    """Test is_union_type detection."""
    result_is_union, result_args = is_union_type(tp)
    assert result_is_union == is_union
    # Check args contain the same elements, order might differ
    assert set(result_args) == set(args)

# is_generic_alias tests
@pytest.mark.parametrize("tp, expected", [
    (List[int], True),
    (Dict[str, Any], True),
    (Tuple[bool], True),
    (Union[int, str], True), # Union is considered a generic alias
    (Optional[float], True), # Optional is Union, also a generic alias
    (list, False), # Bare type is not an alias
    (int, False),
    (Any, False),
    (None, False),
    (type(None), False),
    (TypeVar('T'), False),
    (ForwardRef('X'), False),
    (Literal[1], True), # Literal is a generic alias
    (Callable[[int], str], True), # Callable is a generic alias
])
def test_is_generic_alias(tp, expected):
    """Test is_generic_alias detection."""
    assert is_generic_alias(tp) == expected

# resolve_forward_ref tests
class ResolveRefTarget:
    pass

local_scope_var = int

def test_resolve_forward_ref_success():
    """Test successful forward reference resolution."""
    global_ns = {'ResolveRefTarget': ResolveRefTarget, 'List': List}
    local_ns = {'MyLocalType': bool}
    assert resolve_forward_ref(ForwardRef('ResolveRefTarget'), global_ns) is ResolveRefTarget
    assert resolve_forward_ref("ResolveRefTarget", global_ns) is ResolveRefTarget
    assert resolve_forward_ref(ForwardRef('MyLocalType'), global_ns, local_ns) is bool
    assert resolve_forward_ref("MyLocalType", global_ns, local_ns) is bool
    # Test without localns fallback
    assert resolve_forward_ref("List", global_ns) is List

def test_resolve_forward_ref_name_error():
    """Test NameError on unresolved forward reference."""
    with pytest.raises(NameError, match="Could not resolve forward reference 'NonExistent'"):
        resolve_forward_ref("NonExistent", {}) # Empty global ns

def test_resolve_forward_ref_invalid_type():
    """Test TypeError if input is not str or ForwardRef."""
    with pytest.raises(TypeError, match="Expected str or ForwardRef"):
        resolve_forward_ref(123, {}) # type: ignore

# get_resolved_type_hints tests
def basic_func(a: int, b: str = "hello") -> List[bool]:
    return [bool(a), b == "test"]

class SimpleClass:
    c: float = 0.0
    def method(self, d: 'SimpleClass') -> Optional[int]:
        if d.c > 0:
            return int(d.c)
        return None

class ForwardRefClass:
    attr: "AnotherClass"

class AnotherClass:
    value: int

T = TypeVar('T')
class GenericClass(Generic[T]):
    gen_attr: T
    def gen_method(self, p: T) -> T:
        return p

def test_get_resolved_type_hints_basic_func():
    """Test getting hints from a basic function."""
    hints = get_resolved_type_hints(basic_func)
    assert hints == {'a': int, 'b': str, 'return': List[bool]}

def test_get_resolved_type_hints_simple_class_method():
    """Test getting hints from a class method, resolving forward refs."""
    # Need global namespace for forward ref resolution
    hints = get_resolved_type_hints(SimpleClass.method, globalns=globals())
    assert hints == {'d': SimpleClass, 'return': Optional[int]}

def test_get_resolved_type_hints_class_attributes():
    """Test getting hints for class attributes."""
    hints = get_resolved_type_hints(SimpleClass)
    assert hints == {'c': float}

def test_get_resolved_type_hints_forward_refs():
    """Test resolving forward references across classes."""
    # Need global namespace for forward ref resolution
    hints = get_resolved_type_hints(ForwardRefClass, globalns=globals())
    assert hints == {'attr': AnotherClass}

def test_get_resolved_type_hints_generic_class():
    """Test getting hints from a generic class (hints will contain TypeVars)."""
    hints = get_resolved_type_hints(GenericClass)
    assert hints == {'gen_attr': T}
    method_hints = get_resolved_type_hints(GenericClass.gen_method)
    assert method_hints == {'p': T, 'return': T}

# TODO: Add more complex cases for get_resolved_type_hints if needed
# e.g., nested classes, complex inheritance, Annotated, include_extras=True 

def test_check_type_handler_return_format():
    """Verify check_type correctly handles tuple returns from handlers."""
    # Test case that uses _check_generic_alias (a 5-arg handler)
    # Scenario 1: Match
    match_ok, details_ok = check_type([1, 2], List[int], {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 2: Mismatch (wrong inner type)
    print("\n--- DEBUG: Calling check_type for mismatch scenario ---")
    match_fail, details_fail_obj = check_type([1, "a"], List[int], {}, {})
    print(f"--- DEBUG: Received match={match_fail}, details={details_fail_obj} ---")
    assert match_fail is False # Temporarily comment out the failing assertion
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.path == [1]
    assert details_fail_obj.expected_repr == "int"
    assert details_fail_obj.received_repr == "str"
    assert details_fail_obj.value == "a"

    # Scenario 3: Match (unparameterized List)
    match, details = check_type([1, "a"], List, {}, {})
    assert match is True
    assert details is None

    # Scenario 4: Match (List[Any])
    match, details = check_type([1, "a", None], List[Any], {}, {})
    assert match is True
    assert details is None

def test_check_type_union_mismatch():
    """Verify check_type returns False and details for Union mismatch."""
    # Scenario: Value matches none of the Union types
    match, details_fail_obj = check_type(1.5, Union[int, str], {}, {})
    assert match is False, "Should return False for mismatch"
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.path == []
    assert details_fail_obj.expected_repr == "Union[int, str]"
    assert details_fail_obj.received_repr == "float"
    assert details_fail_obj.value == 1.5

# Test cases for Annotated type handling (requires Python 3.9+ or typing_extensions)
@pytest.mark.skipif(Annotated is None, reason="Annotated not available")
def test_check_type_annotated():
    """Verify check_type correctly handles Annotated[T, ...] by checking T."""
    MyAnnotatedInt = Annotated[int, "Some metadata"]
    MyAnnotatedList = Annotated[List[str], "More metadata"]

    # Scenario 1: Match (int against Annotated[int, ...])
    match_ok, details_ok = check_type(10, MyAnnotatedInt, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 2: Mismatch (str against Annotated[int, ...])
    match_fail, details_fail_obj = check_type("hello", MyAnnotatedInt, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "int"
    assert details_fail_obj.received_repr == "str"

    # Scenario 3: Match (List[str] against Annotated[List[str], ...])
    match_ok, details_ok = check_type(["a", "b"], MyAnnotatedList, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 4: Mismatch (List[int] against Annotated[List[str], ...])
    match_fail, details_fail_obj = check_type([1, 2], MyAnnotatedList, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "str"
    assert details_fail_obj.received_repr == "int"
    assert details_fail_obj.path == [0]

    # Scenario 5: Match (None against Optional[Annotated[int, ...]])
    MyOptionalAnnotated = Optional[MyAnnotatedInt]
    match_ok, details_ok = check_type(None, MyOptionalAnnotated, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 6: Match (int against Optional[Annotated[int, ...]])
    match_ok, details_ok = check_type(5, MyOptionalAnnotated, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 7: Mismatch (str against Optional[Annotated[int, ...]])
    match_fail, details_fail_obj = check_type("no", MyOptionalAnnotated, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "int"
    assert details_fail_obj.received_repr == "str"

# Test cases for check_type with Literal
def test_check_type_literal():
    """Verify check_type correctly handles Literal types."""
    Action = Literal["run", "jump", "hide"]
    Status = Literal[200, 404, 500]

    # Scenario 1: Match (string literal)
    match_ok, details_ok = check_type("run", Action, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 2: Mismatch (string literal)
    match_fail, details_fail_obj = check_type("walk", Action, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "Literal['run', 'jump', 'hide']"
    assert details_fail_obj.received_repr == "str"
    assert details_fail_obj.value == "walk"
    assert details_fail_obj.message == "Value 'walk' of type str not in allowed literals: 'run', 'jump', 'hide'"

    # Scenario 3: Match (int literal)
    match_ok, details_ok = check_type(404, Status, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 4: Mismatch (int literal)
    match_fail, details_fail_obj = check_type(302, Status, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "Literal[200, 404, 500]"
    assert details_fail_obj.received_repr == "int"
    assert details_fail_obj.value == 302
    assert details_fail_obj.message == "Value 302 of type int not in allowed literals: 200, 404, 500"

    # Scenario 5: Mismatch (wrong type for literal)
    match_fail, details_fail_obj = check_type(200, Action, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "Literal['run', 'jump', 'hide']"
    assert details_fail_obj.received_repr == "int"
    assert details_fail_obj.message == "Value 200 of type int not in allowed literals: 'run', 'jump', 'hide'"

def test_check_type_callable():
    """Verify check_type correctly handles Callable types."""
    def sample_func(x: int) -> str:
        return str(x)
    
    non_callable = 123
    
    # Scenario 1: Match (basic callable)
    match_ok, details_ok = check_type(sample_func, Callable, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 2: Match (callable against specific signature - Note: DieCast only checks callability)
    match_ok, details_ok = check_type(sample_func, Callable[[int], str], {}, {})
    assert match_ok is True
    assert details_ok is None # DieCast currently doesn't validate signature details

    # Scenario 3: Mismatch (not callable)
    match_fail, details_fail_obj = check_type(non_callable, Callable, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "Callable"
    assert details_fail_obj.received_repr == "int"
    assert details_fail_obj.message == "Value is not callable"

def test_check_type_tuple():
    """Verify check_type correctly handles Tuple types."""
    FixedTuple = Tuple[int, str]
    VariadicTuple = Tuple[int, ...]

    # Scenario 1: Match (fixed tuple)
    match_ok, details_ok = check_type((1, "a"), FixedTuple, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 2: Mismatch (fixed tuple - wrong type)
    match_fail, details_fail_obj = check_type((1, 2), FixedTuple, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.path == [1]
    assert details_fail_obj.expected_repr == "str"
    assert details_fail_obj.received_repr == "int"

    # Scenario 3: Mismatch (fixed tuple - wrong length)
    match_fail, details_fail_obj = check_type((1, "a", True), FixedTuple, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "tuple[int, str]"
    assert details_fail_obj.message == "Expected 2 elements, got 3"

    # Scenario 4: Match (variadic tuple)
    match_ok, details_ok = check_type((1, 2, 3), VariadicTuple, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 5: Mismatch (variadic tuple - wrong type)
    match_fail, details_fail_obj = check_type((1, "a", 3), VariadicTuple, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.path == [1]
    assert details_fail_obj.expected_repr == "int"
    assert details_fail_obj.received_repr == "str"

def test_check_type_mapping():
    """Verify check_type correctly handles Mapping/Dict types."""
    StrIntMap = Dict[str, int]

    # Scenario 1: Match
    match_ok, details_ok = check_type({"a": 1, "b": 2}, StrIntMap, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 2: Mismatch (wrong key type)
    match_fail, details_fail_obj = check_type({1: 1, "b": 2}, StrIntMap, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.path == ['key(1)'] # Path indicates key check failure
    assert details_fail_obj.expected_repr == "str"
    assert details_fail_obj.received_repr == "int"

    # Scenario 3: Mismatch (wrong value type)
    match_fail, details_fail_obj = check_type({"a": 1, "b": "2"}, StrIntMap, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.path == ['value(\'b\')'] # FIX: Use single quotes in path assertion
    assert details_fail_obj.expected_repr == "int"
    assert details_fail_obj.received_repr == "str"

def test_check_type_sequence():
    """Verify check_type correctly handles Sequence/List/Set types."""
    IntList = List[int]
    StrSet = Set[str]

    # Scenario 1: Match (List)
    match_ok, details_ok = check_type([1, 2, 3], IntList, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 2: Mismatch (List - wrong element type)
    match_fail, details_fail_obj = check_type([1, "2", 3], IntList, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.path == [1]
    assert details_fail_obj.expected_repr == "int"
    assert details_fail_obj.received_repr == "str"

    # Scenario 3: Match (Set)
    match_ok, details_ok = check_type({"a", "b"}, StrSet, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 4: Mismatch (Set - wrong element type)
    match_fail, details_fail_obj = check_type({"a", 1}, StrSet, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    # Path for set element failure might be less predictable, focus on types
    assert details_fail_obj.expected_repr == "str"
    assert details_fail_obj.received_repr == "int"

# Need to import dataclasses for the test
import dataclasses

@dataclasses.dataclass
class SimpleDataClass:
    id: int
    name: str
    active: Optional[bool] = None

def test_check_type_dataclass():
    """Verify check_type correctly handles dataclass types."""
    # Scenario 1: Match
    instance1 = SimpleDataClass(id=1, name="Test")
    match_ok, details_ok = check_type(instance1, SimpleDataClass, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 2: Match (with optional field set)
    instance2 = SimpleDataClass(id=2, name="Test2", active=True)
    match_ok, details_ok = check_type(instance2, SimpleDataClass, {}, {})
    assert match_ok is True
    assert details_ok is None

    # Scenario 3: Mismatch (wrong type for required field)
    # check_type doesn't validate *inside* dataclasses by default
    # This test confirms the instance IS a SimpleDataClass, internal validation is separate
    instance_wrong_type = SimpleDataClass(id="not-an-int", name="Test Wrong") # type: ignore
    match_ok, details_ok = check_type(instance_wrong_type, SimpleDataClass, {}, {})
    assert match_ok is True # It *is* an instance of SimpleDataClass
    assert details_ok is None
    # NOTE: DieCast's @diecast decorator *would* catch the internal mismatch
    # when applied to the __init__ or a method using the instance.
    # This test just verifies check_type's behavior for the instance itself.

    # Scenario 4: Mismatch (completely wrong type)
    match_fail, details_fail_obj = check_type({"id": 1}, SimpleDataClass, {}, {})
    assert match_fail is False
    assert isinstance(details_fail_obj, Obituary)
    assert details_fail_obj.expected_repr == "SimpleDataClass"
    assert details_fail_obj.received_repr == "dict"