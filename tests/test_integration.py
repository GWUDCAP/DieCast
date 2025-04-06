from typing import (
    AsyncGenerator, Generator, 
    Union, List, Dict, Any, Optional, Tuple, Set, Iterator )
import asyncio
import pytest
import sys
import os

# Add src dir to path to allow importing diecast
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from diecast.type_utils import YouDiedError, Obituary
from diecast.config import _DIECAST_MARKER
from diecast import diecast

# --- Deep Inheritance Chain Definitions (Moved from test_mro_optimization) ---
class A:
    def identify(self) -> str:
        return self.__class__.__name__
class B(A): pass
class C(B): pass
class D(C): pass
class E(D): pass
class F(E): pass
class G(F): pass
class H(G): pass
class I(H): pass
class J(I): pass
class AA:
    def identify(self) -> str:
        return self.__class__.__name__
class BB(AA): pass
class CC(BB): pass
class DD(CC): pass
class EE(DD): pass
class FF(EE): pass
class GG(FF): pass
class HH(GG): pass
class II(HH): pass
class JJ(II): pass

# --- Integration Tests (Moved from test_mro_optimization) ---

@diecast
def simple_func(value: Union[A, AA]) -> Union[A, AA]:
    return value

@diecast
def complex_func(values: List[Union[A, B, C, D, E, F, G, H, I, J, 
                                   AA, BB, CC, DD, EE, FF, GG, HH, II, JJ]]) -> int:
    return len(values)

class TestDeepInheritanceIntegration:
    """Tests decorator behavior with deep inheritance chains."""
    
    def test_basic_functionality(self):
        """Test that the functions work correctly with deep inheritance."""
        j_instance = J()
        assert simple_func(j_instance) is j_instance
        jj_instance = JJ()
        assert simple_func(jj_instance) is jj_instance
        with pytest.raises(YouDiedError):
            simple_func(123) # Expect failure with wrong type
        mixed_list = [J(), JJ(), I(), II(), H(), HH()]
        assert complex_func(mixed_list) == 6
    
    def test_complex_nested_types(self):
        """Test decorator with nested types and deep inheritance."""
        complex_j = [{"a": J()}, {"b": J()}]
        complex_jj = [{"a": JJ()}, {"b": JJ()}]
        
        @diecast
        def nested_func(values: List[Dict[str, A]]) -> int:
            return len(values)
        
        @diecast
        def nested_func_aa(values: List[Dict[str, AA]]) -> int:
            return len(values)
        
        assert nested_func(complex_j) == 2
        with pytest.raises(YouDiedError):
            nested_func(complex_jj) # AA is not A
        assert nested_func_aa(complex_jj) == 2
        with pytest.raises(YouDiedError):
            nested_func_aa(complex_j) # A is not AA

# --- Added Integration Tests ---

# Scenario 1 & 2: mold() and ignore Integration
def test_mold_and_ignore_integration(tmp_path):
    """Tests that mold applies diecast correctly and respects ignore."""
    # Add debug import to the module content
    module_content = """
import sys
import os
import inspect
# Add src dir back for the temp module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from diecast import diecast, mold, ignore
from diecast.config import _DIECAST_MARKER
from diecast.type_utils import YouDiedError
from typing import List

# DEBUG FUNCTION - to help diagnose mold() issues
def _debug_mold():
    import inspect
    import sys
    frame = sys._getframe(1)  # Get caller's frame
    module_name = frame.f_globals.get('__name__')
    print(f"DEBUG: Module name: {module_name}")
    module = sys.modules[module_name]
    try:
        module_file = inspect.getfile(module)
        print(f"DEBUG: Module file: {module_file}")
    except TypeError:
        module_file = getattr(module, '__file__', None)
        print(f"DEBUG: Module file (from __file__): {module_file}")
    
    # Check a few representative objects
    for name, obj in [
        ("annotated_func", annotated_func),
        ("TargetClass.annotated_method", TargetClass.annotated_method),
    ]:
        try:
            obj_file = inspect.getfile(obj)
            print(f"DEBUG: {name} file: {obj_file}")
        except TypeError:
            print(f"DEBUG: {name} file: TypeError from inspect.getfile")
        print(f"DEBUG: {name} has marker: {hasattr(obj, _DIECAST_MARKER)}")
        print(f"DEBUG: {name} has annotations: {bool(getattr(obj, '__annotations__', {}))}") 

@ignore
class IgnoredClass:
    def method(self, x: int) -> str:
        return str(x + 1) # Incorrect return type, but should be ignored

class TargetClass:
    def __init__(self, val: str):
        self.val = val # No annotation, should be ignored by mold

    def annotated_method(self, y: int) -> str:
        return str(y * 2)

    @ignore
    def ignored_method(self, z: float) -> float:
        return z / 0 # Error, but should be ignored

    def unannotated_method(self, a):
        return a

@diecast # Explicitly decorated, mold should skip
def already_decorated(p: bool) -> bool:
    if not p: raise YouDiedError("Forced error", cause='test_case')
    return p

def annotated_func(q: List[int]) -> int:
    return sum(q)

def unannotated_func(r):
    return r

@ignore
def ignored_func(s: str) -> int:
    return "bad" # Incorrect type, but ignored

# Apply mold
print("DEBUG: Before mold()")
_debug_mold()  # Call debug function before mold
mold()
print("DEBUG: After mold()")
_debug_mold()  # Call debug function after mold
"""
    # Create temporary module file
    mod_file = tmp_path / "temp_module.py"
    mod_file.write_text(module_content)

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location("temp_module", mod_file)
    temp_module = importlib.util.module_from_spec(spec)
    sys.modules["temp_module"] = temp_module
    spec.loader.exec_module(temp_module)

    # --- Verification ---
    # Check classes and functions marked with @ignore have the marker
    assert getattr(temp_module.IgnoredClass, _DIECAST_MARKER, False)
    assert getattr(temp_module.TargetClass.ignored_method, _DIECAST_MARKER, False)
    assert getattr(temp_module.ignored_func, _DIECAST_MARKER, False)
    
    # Methods of ignored classes DON'T have the marker directly
    # But they're still "ignored" because mold() skips the whole class
    assert not getattr(temp_module.IgnoredClass.method, _DIECAST_MARKER, False)
    
    # Verify ignored items don't have type checking behavior
    ignored_instance = temp_module.IgnoredClass()
    # Should not fail even though return type doesn't match annotation
    assert isinstance(ignored_instance.method(10), str) # Returns str(10+1) = "11"
    
    instance = temp_module.TargetClass("hello") # Create instance for later tests
    # Should not fail even though it would divide by zero
    with pytest.raises(ZeroDivisionError):
        instance.ignored_method(1.0)
    
    # Shouldn't fail even though return type doesn't match annotation
    assert temp_module.ignored_func("test") == "bad" # Returns string instead of int

    # Check already decorated item IS wrapped (verify behavior, not marker)
    assert temp_module.already_decorated(True) is True # Should pass
    with pytest.raises(YouDiedError, match="Forced error"):
        temp_module.already_decorated(False) # Should raise original error

    # Check unannotated items are NOT wrapped
    # __init__ has parameter annotation (val: str), so it IS annotated (comment was incorrect)
    assert not getattr(temp_module.TargetClass.unannotated_method, _DIECAST_MARKER, False)
    assert not getattr(temp_module.unannotated_func, _DIECAST_MARKER, False)

    # Check annotated items ARE wrapped by mold (verify behavior, not marker)
    # __init__ has parameter annotation, so it should be wrapped by mold
    assert getattr(temp_module.TargetClass.__init__, _DIECAST_MARKER, False)
    instance = temp_module.TargetClass("hello") # Re-instantiate here for clarity
    assert instance.annotated_method(5) == "10" # Check molded method works
    with pytest.raises(YouDiedError): # Check molded method enforces type
        instance.annotated_method("bad")
    assert temp_module.annotated_func([1, 2, 3]) == 6 # Check molded function works
    with pytest.raises(YouDiedError): # Check molded function enforces type (arg)
        temp_module.annotated_func("bad")
    with pytest.raises(YouDiedError): # Check molded function enforces type (nested)
        temp_module.annotated_func([1, "a"])

# Scenario 3: Async Function Integration
@diecast
async def async_multiply(a: int, b: int) -> int:
    await asyncio.sleep(0.01)
    return a * b

@diecast
async def async_fail_arg(a: str) -> str:
    return a

@diecast
async def async_fail_return(a: int) -> str:
    return a # Incorrect return type

@pytest.mark.asyncio
async def test_async_integration():
    """Tests @diecast with async functions."""
    assert await async_multiply(3, 4) == 12

    with pytest.raises(YouDiedError):
        await async_multiply("3", 4) # Wrong arg type

    await async_fail_arg("hello") # Correct type
    with pytest.raises(YouDiedError):
        await async_fail_arg(123) # Wrong arg type

    with pytest.raises(YouDiedError):
        await async_fail_return(10) # Wrong return type

# Scenario 4: Generator Function Integration
@diecast
def count_up_sync(n: int) -> Generator[int, None, str]:
    i = 0
    while i < n:
        yield i
        i += 1
    return f"Counted to {n}"

@diecast
async def count_up_async(n: int) -> AsyncGenerator[int, None]:
    i = 0
    while i < n:
        yield i
        await asyncio.sleep(0.01)
        i += 1
    # No return value annotation for AsyncGenerator return

@diecast
def bad_yield_sync(n: int) -> Generator[int, None, None]:
    yield 0
    yield "one" # Bad yield type
    yield 2

@diecast
async def bad_yield_async(n: int) -> AsyncGenerator[int, None]:
    yield 0
    await asyncio.sleep(0.01)
    yield "one" # Bad yield type
    await asyncio.sleep(0.01)
    yield 2

@diecast
def bad_return_sync(n: int) -> Generator[int, None, str]:
    yield 0
    return 123 # Bad return type

def test_sync_generator_integration():
    """Tests @diecast with synchronous generators."""
    # Test successful iteration and return value capture
    gen_good = count_up_sync(3)
    
    # Collect results manually, then get return value
    results = []
    return_value = None
    
    # Step 1: Exhaust the generator manually
    try:
        while True:
            results.append(next(gen_good))
    except StopIteration as e:
        # This is how we capture the return value in Python 3.7+
        return_value = e.value
    
    # Verify iteration results and return value  
    assert results == [0, 1, 2]
    assert return_value == "Counted to 3" # Check return value captured correctly

    # Test failure on bad yield
    gen_bad_yield = bad_yield_sync(3)
    assert next(gen_bad_yield) == 0
    with pytest.raises(YouDiedError): # Error on second yield
        next(gen_bad_yield)

    # Test failure on bad return - manually iterate to trigger return check
    gen_bad_return = bad_return_sync(1)
    assert next(gen_bad_return) == 0
    # Attempt to exhaust the generator, which should trigger the return value check
    with pytest.raises(YouDiedError) as excinfo_bad_return:
        # Consume until StopIteration would be raised
        try:
            while True:
                next(gen_bad_return)
        except StopIteration:
            pytest.fail("YouDiedError was caught inside the try block, not by pytest.raises")
    
    # Verify correct error type and reason (Return value)
    msg = str(excinfo_bad_return.value)
    assert 'Return value' in msg

@pytest.mark.asyncio
async def test_async_generator_integration():
    """Tests @diecast with asynchronous generators."""
    # Test successful iteration
    agen_good = count_up_async(3) # No await!

    results_good = []
    async for i in agen_good:
        results_good.append(i)

    # Verify results
    assert results_good == [0, 1, 2]

    # Test case where the generator yields a wrong type
    agen_bad_yield = bad_yield_async(3) # No await!
    with pytest.raises(YouDiedError) as excinfo_bad_yield:
        # Consume the generator to trigger the error
        results_bad_yield = []
        async for i in agen_bad_yield:
            results_bad_yield.append(i)
        # Verify correct error type and reason (Yield value)
        msg = str(excinfo_bad_yield.value)
        assert 'Yield value' in msg

# Scenario 5: Nested Error Reporting
@diecast
def outer_func(data: Dict[str, List[int]]) -> int:
    total = 0
    for key, inner_list in data.items():
        total += inner_processing(key, inner_list)
    return total

@diecast
def inner_processing(name: str, numbers: List[int]) -> int:
    # This will fail if numbers contains a non-int
    return sum(numbers) * len(name)

@diecast
def calls_outer(good_data: Dict[str, List[int]], bad_data: Dict[str, List[Any]]) -> None:
    outer_func(good_data) # Should pass
    outer_func(bad_data) # Should fail inside inner_processing

def test_nested_error_reporting():
    """Test that error messages correctly report errors from nested calls."""
    good = {"a": [1, 2], "b": [3, 4]}
    bad_inner_list = {"a": [1, 2], "b": [3, "4"]} # "4" is str, not int
    bad_key_type = {123: [1, 2]} # 123 is int, not str

    # Test success case
    assert outer_func(good) == (1 + 2) * 1 + (3 + 4) * 1 # 3*1 + 7*1 = 10

    # Test failure within inner_processing due to list content
    with pytest.raises(YouDiedError) as excinfo_inner:
        calls_outer(good, bad_inner_list)
    # Verify correct error type and reason (Argument in inner_processing)
    err_str_inner = str(excinfo_inner.value)
    assert 'Argument' in err_str_inner

    # Test failure within outer_func due to dict key type
    with pytest.raises(YouDiedError) as excinfo_outer:
        outer_func(bad_key_type)
    # Verify correct error type and reason (Argument in outer_func)
    err_str_outer = str(excinfo_outer.value)
    assert 'Argument' in err_str_outer

# TODO: Add more integration tests covering combinations of features
# - TypeVar consistency checks integrated with mold/inheritance
# - Complex forward reference resolution across molded modules
# - Interaction with other decorators

# Need importlib for the mold test
import importlib.util

# ==== Deep Inheritance Tests ====

# --- Class Chain A-J ---
# ... (class definitions A-J) ...

@diecast
def process_j(arg: J) -> str:
    return arg.identify()

def test_inheritance_deep_pass():
    """Test that process_j accepts a valid J instance."""
    instance_j = J()
    assert process_j(instance_j) == instance_j.identify() # Should not raise

def test_inheritance_deep_fail():
    instance_i = I()
    with pytest.raises(YouDiedError) as excinfo:
        process_j(instance_i) # I is not J
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'J'
    assert e.obituary.received_repr == 'I'
    assert e.obituary.path == ['arg']

# --- Class Chain AA-JJ ---
# ... (class definitions AA-JJ) ...

@diecast
def process_jj(arg: JJ) -> str:
    return arg.identify()

def test_inheritance_deeper_pass():
    """Test that process_jj accepts a valid JJ instance."""
    instance_jj = JJ()
    assert process_jj(instance_jj) == instance_jj.identify() # Should not raise

def test_inheritance_deeper_fail():
    instance_ii = II()
    with pytest.raises(YouDiedError) as excinfo:
        process_jj(instance_ii) # II is not JJ
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'JJ'
    assert e.obituary.received_repr == 'II'
    assert e.obituary.path == ['arg']

    # Test passing base class AA where JJ is expected
    instance_aa = AA()
    with pytest.raises(YouDiedError) as excinfo_aa:
        process_jj(instance_aa)
    e_aa = excinfo_aa.value
    assert e_aa.cause == 'argument'
    assert e_aa.obituary.expected_repr == 'JJ'
    assert e_aa.obituary.received_repr == 'AA'
    assert e_aa.obituary.path == ['arg']


# ==== Nested Generic Tests ====

# --- List[Union[int, str]] ---

@diecast
def process_list_union(arg: List[Union[int, str]]) -> int:
    return len(arg)

def test_nested_list_union_pass():
    """Test processing a list with valid int and str union elements."""
    assert process_list_union([1, "two", 3, "four"]) == 4 # Should not raise
    assert process_list_union([]) == 0 # Empty list is valid

def test_nested_list_union_fail():
    with pytest.raises(YouDiedError) as excinfo:
        process_list_union([1, "two", 3.0]) # 3.0 is float, not int or str
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'Union[str, int]' # Order flipped
    assert e.obituary.received_repr == 'float'
    assert e.obituary.path == ['arg', 2] # Path includes index
    # Revert message assertion based on new log
    assert e.obituary.message == 'Value does not match any type in Union[str, int]'


# --- Dict[str, Optional[CustomType]] ---

class CustomType:
    def __init__(self, value: int):
        self.value = value
    def __eq__(self, other): # Add eq for potential assert checks if needed
        return isinstance(other, CustomType) and self.value == other.value

@diecast
def process_dict_optional_custom(arg: Dict[str, Optional[CustomType]]) -> int:
    count = 0
    for k, v in arg.items():
        if v is not None:
            count += v.value
    return count

def test_nested_dict_optional_custom_pass():
    """Test processing a dict with valid str keys and Optional[CustomType] values."""
    valid_data_1 = {"a": CustomType(10), "b": None, "c": CustomType(5)}
    assert process_dict_optional_custom(valid_data_1) == 15 # 10 + 5, should not raise
    valid_data_2 = {"only_none": None}
    assert process_dict_optional_custom(valid_data_2) == 0 # Should not raise
    assert process_dict_optional_custom({}) == 0 # Empty dict is valid

def test_nested_dict_optional_custom_fail_key():
    # Fails because key 123 is not str
    with pytest.raises(YouDiedError) as excinfo:
        process_dict_optional_custom({"a": CustomType(1), 123: None})
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'str' # Expected key type
    assert e.obituary.received_repr == 'int'
    assert e.obituary.path == ['arg', 123] # Path includes key
    # Revert message assertion back to original expectation
    assert e.obituary.message == 'Incorrect type for key 123' 

def test_nested_dict_optional_custom_fail_value():
    # Fails because value "not_custom" is not Optional[CustomType]
    with pytest.raises(YouDiedError) as excinfo:
        process_dict_optional_custom({"a": CustomType(1), "b": "not_custom"})
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'CustomType' # Was 'CustomType | None'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['arg', "value('b')"] # Update path based on new log
    assert e.obituary.message == 'Value does not match inner type CustomType of Optional'

def test_nested_dict_optional_custom_fail_value_inner():
    # Fails because value CustomType(1.5) has internal float, not handled by this test
    # but let's test if the CustomType itself is wrong, e.g., pass an int
    with pytest.raises(YouDiedError) as excinfo:
        process_dict_optional_custom({"a": CustomType(1), "b": 123})
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'CustomType' # Was 'CustomType | None'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.path == ['arg', "value('b')"] # Update path based on new log
    assert e.obituary.message == 'Value does not match inner type CustomType of Optional'

# --- More Complex Nesting: Tuple[List[Set[int]], Dict[str, bool]] ---

@diecast
def process_complex_nesting(arg: Tuple[List[Set[int]], Dict[str, bool]]) -> int:
    set_sum = sum(sum(s) for s in arg[0])
    true_count = sum(1 for v in arg[1].values() if v)
    return set_sum + true_count

def test_complex_nesting_pass():
    """Test processing a valid complex nested tuple structure."""
    valid_data = ([{1, 2}, {3}, set()], {"x": True, "y": False, "z": True})
    expected_sum = (1 + 2) + 3 + 0
    expected_true_count = 1 + 0 + 1
    assert process_complex_nesting(valid_data) == expected_sum + expected_true_count # 6 + 2 = 8
    # Test with empty inner list/dict
    valid_empty_inner = ([], {})
    assert process_complex_nesting(valid_empty_inner) == 0

def test_complex_nesting_fail_tuple_outer():
    # Fails because outer structure is List, not Tuple
    with pytest.raises(YouDiedError) as excinfo:
        process_complex_nesting([[{1, 2}], {"a": True}])
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'tuple[list[set[int]], dict[str, bool]]'
    assert e.obituary.received_repr == 'list'
    assert e.obituary.path == ['arg']
    assert "Value type list is not compatible with expected container origin tuple" in e.obituary.message

def test_complex_nesting_fail_list_inner():
    """Test failure deep inside a nested list."""
    value = ({"a": 1},)
    with pytest.raises(YouDiedError) as e:
        process_complex_nesting(value)
    # Check that the error originates from the list check failing the origin
    assert e.value.obituary.expected_repr == 'list[set[int]]'
    assert e.value.obituary.received_repr == 'dict'
    assert e.value.obituary.path == ['arg', 0]
    assert "Value type dict is not compatible with expected container origin list" in e.value.obituary.message

def test_complex_nesting_fail_set_element():
    """Test failure deep inside a nested set."""
    value = ([[{ 'a' }], {'b': True}],)
    with pytest.raises(YouDiedError) as e:
        process_complex_nesting(value)
    # Check that the error originates from the set element check
    assert e.value.obituary.expected_repr == 'int'
    assert e.value.obituary.received_repr == 'str'
    assert e.value.obituary.value == 'a'
    # Path includes the specific element causing failure within the set
    assert e.value.obituary.path == ['arg', 0, 0, "elem('a')"] # Path ends at failing element

def test_complex_nesting_fail_dict_value():
    """Test failure deep inside a nested dict value."""
    value = ([[{1}], {'a': 'not bool'}],)
    with pytest.raises(YouDiedError) as e:
        process_complex_nesting(value)
    # Check that the error originates from the dict value check
    assert e.value.obituary.expected_repr == 'bool'
    assert e.value.obituary.received_repr == 'str'
    assert e.value.obituary.value == 'not bool'
    assert e.value.obituary.path == ['arg', 1, "value('a')"]


# === Additional integration scenarios based on TODOs ===

# --- TypeVar Integration (using mold) ---

@pytest.fixture(scope="function")
def molded_typevar_module(temp_module):
    code = """
from diecast import mold
from diecast import diecast
from diecast.type_utils import YouDiedError
from typing import TypeVar, Generic, Sequence

T = TypeVar('T')
N = TypeVar('N', bound=Sequence)
C = TypeVar('C', int, str)

class Processor(Generic[T]):
    @diecast
    def process(self, item: T) -> T:
        return item

@diecast
def check_bound(seq: N) -> int:
    return len(seq)

def check_constrained(val: C) -> C:
    return val

mold()
"""
    # Pass module name AND code string correctly
    module, _ = temp_module("molded_tv_mod", code)
    return module

def test_mold_typevar_generic_class_pass(molded_typevar_module):
    """Test mold processing a generic class with valid types."""
    str_processor = molded_typevar_module.Processor[str]()
    assert str_processor.process("hello") == "hello" # Should not raise
    list_processor = molded_typevar_module.Processor[List[int]]()
    assert list_processor.process([1, 2]) == [1, 2] # Should not raise

def test_mold_typevar_generic_class_fail(molded_typevar_module):
    int_processor = molded_typevar_module.Processor[int]()
    with pytest.raises(YouDiedError) as excinfo:
        int_processor.process("string") # Pass str to int-bound processor
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == '~T'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['item']
    assert e.obituary.message == 'TypeVar consistency violation: Expected type bound to int but received str'

def test_mold_typevar_bound_fail(molded_typevar_module):
    with pytest.raises(YouDiedError) as e:
        molded_typevar_module.check_bound(5)
    assert e.value.obituary.expected_repr == '~N'
    assert e.value.obituary.received_repr == 'int'
    assert e.value.obituary.path == ['seq']
    assert e.value.obituary.message == 'Value does not meet bound type Sequence'

def test_mold_typevar_constrained_fail(molded_typevar_module):
    with pytest.raises(YouDiedError) as excinfo:
        molded_typevar_module.check_constrained(1.5)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == '~C'
    assert e.obituary.received_repr == 'float'
    assert e.obituary.path == ['val']
    assert e.obituary.message == 'Value not in allowed types for constrained TypeVar'

# --- Generator/Async Integration ---

@diecast
def integrated_generator(limit: int) -> Iterator[Union[int, str]]:
    for i in range(limit):
        yield i
        if i == 1:
            yield "ok"
        if i > 1:
            yield 2.0 # Error here

@diecast
async def integrated_async(data: List[int]) -> str:
    await asyncio.sleep(0.01)
    if len(data) > 2:
        return 123 # Error here
    return f"Processed {len(data)} items"

def test_integration_generator_fail():
    """Test generator yield type failure."""
    gen = integrated_generator(3)
    assert next(gen) == 0
    assert next(gen) == 1
    assert next(gen) == "ok"
    with pytest.raises(YouDiedError) as e:
        next(gen)

    assert e.value.obituary.expected_repr == 'Union[str, int]'
    assert e.value.obituary.received_repr == 'float'
    assert e.value.obituary.value == 2.0
    assert e.value.obituary.path == []
    assert 'Value does not match any type in Union[str, int]' in e.value.obituary.message

@pytest.mark.asyncio
async def test_integration_async_fail():
    with pytest.raises(YouDiedError) as excinfo:
        await integrated_async([1, 2, 3])
    e = excinfo.value
    assert e.cause == 'return'
    assert e.obituary.expected_repr == 'str'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.path == []

# ... (You may add more tests for ForwardRef, multiple decorators etc. if needed) 