# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from collections.abc import Sequence as ABCSequence
from typing import (
    Iterator, 
    Optional, 
    TypeVar, 
    Generic,
    Union, 
    Dict, 
    List 
)
import importlib
import asyncio
import pytest
import uuid
import sys
import gc
import os

## ===== LOCAL ===== ##
from diecast.type_utils import YouDiedError
from diecast.error_utils import Obituary
import diecast.logging
from .conftest import strip_ansi

# We import this last. Importing from a submodule of diecast
# would overwrite the decorator diecast in the namespace
from diecast import diecast

# ===== FIXTURES ===== #
@pytest.fixture(autouse=True)
def cleanup_typevar_bindings():
    """Clean up any TypeVar bindings after each test to prevent resource leaks."""
    # Run the test
    yield
    
    # Clean up - import inside the fixture to avoid circular imports
    from diecast.type_utils import _TYPEVAR_BINDINGS
    if hasattr(_TYPEVAR_BINDINGS, 'clear'):
        _TYPEVAR_BINDINGS.clear()
    
    # Force garbage collection
    gc.collect()

# ===== MOCKS ===== #
@diecast
def _err_basic_func(a: int, b: str) -> float:
    return float(a) + len(b)

@diecast
def _err_optional_func(a: Optional[str]) -> Optional[int]:
    return len(a) if a is not None else None

@diecast
def _err_union_func(a: Union[int, str]) -> Union[float, bool]:
    if isinstance(a, int):
        return float(a)
    else:
        return len(a) > 0

@diecast
def _err_forward_ref_func(target: '_ErrForwardRefTarget') -> bool:
    return isinstance(target, _ErrForwardRefTarget)

class _ErrForwardRefTarget: pass

class _ErrSimpleClass:
    @diecast
    def instance_method(self, x: int) -> str:
        return f"Value: {x}"
    @classmethod
    @diecast
    def class_method(cls, y: str) -> bool:
        return isinstance(y, str)
    @staticmethod
    @diecast
    def static_method(z: bool = True) -> Optional[bool]:
        return z

_err_instance = _ErrSimpleClass()

T_ERR_Unconstrained = TypeVar('T_ERR_Unconstrained')
T_ERR_Constrained = TypeVar('T_ERR_Constrained', int, str)
T_ERR_Bound = TypeVar('T_ERR_Bound', bound=ABCSequence)

@diecast
def _err_typevar_constrained_func(x: T_ERR_Constrained) -> T_ERR_Constrained:
    return x

@diecast
def _err_typevar_bound_func(x: T_ERR_Bound) -> int:
    return len(x)

@diecast
def _err_nested_wrong_return(a: int) -> str:
    return a # Returns int instead of str

@diecast
def _err_nested_wrong_optional_return(a: Optional[str]) -> Optional[str]:
    return 123 # Incorrect type

@diecast
def _err_nested_wrong_union_return(a: Union[int, str]) -> Union[int, str]:
    return [a] # Returns list

@diecast
async def _err_nested_bad_async() -> str:
    await asyncio.sleep(0.01)
    return 123 # Wrong type

@diecast
def _err_nested_wrong_return_typevar(x: T_ERR_Unconstrained) -> T_ERR_Unconstrained:
    return "wrong type"

class _ErrConsistentGeneric(Generic[T_ERR_Unconstrained]):
    @diecast
    def method(self, x: T_ERR_Unconstrained, y: T_ERR_Unconstrained) -> T_ERR_Unconstrained:
        return x

class _ErrParent: pass
class _ErrChild(_ErrParent): pass

@diecast
def _err_inheritance_func_b(c: _ErrChild) -> bool:
    return True

@diecast
def _err_typevar_consistency_func(x: T_ERR_Unconstrained, y: T_ERR_Unconstrained) -> T_ERR_Unconstrained:
    return x

_ErrClassWithConsistency = _ErrConsistentGeneric 

@diecast
def simple_func(value: int) -> str:
    return str(value)


class ErrorReporter:
    @diecast
    def instance_method(self, value: Dict[str, List[int]]) -> List[str]:
        return list(value.keys())

# ===== TESTS ===== #
class TestErrorReporting:
    """Tests for the error message format and caller information in DieCast."""
    
    def test_error_message_format_args(self):
        """Test that the error message for arguments follows the spec format."""
        try:
            simple_func("not an int")
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            error_msg = str(e)
            msg_stripped = strip_ansi(error_msg) # Strip ANSI
            
            # Check header components
            assert "YouDiedError:" in msg_stripped
            assert "Argument mismatch in function 'tests.test_error_reporting.simple_func'" in msg_stripped
            assert "for argument 'value'" in msg_stripped
            
            # Check core details based on _generate_error_message_core format
            assert "Expected: int" in msg_stripped # Precise check
            assert "Value: 'not an int'" in msg_stripped # Precise check
            assert "Path: `value`" in msg_stripped 
            
            # Check caller info hint
            assert "Error occurred in" in msg_stripped
            assert "test_error_message_format_args" in msg_stripped
    
    def test_error_message_format_return(self):
        """Test that the error message for return values follows the spec format."""
        @diecast
        def wrong_return(value: int) -> int:
            return str(value)  # Returns str instead of int
        
        try:
            wrong_return(42)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            error_msg = str(e)
            msg_stripped = strip_ansi(error_msg) # Strip ANSI

            # Check header components
            assert "YouDiedError:" in msg_stripped
            assert "Return value mismatch in function 'tests.test_error_reporting.wrong_return'" in msg_stripped

            # Check core details based on _generate_error_message_core format
            assert "Return value mismatch" in msg_stripped
            # Check Expected type components separately
            assert "Expected:" in msg_stripped 
            assert " int" in msg_stripped # Check for type name with potential leading space
            assert "Value: '42'" in msg_stripped

            # Check caller info hint
            assert "Error occurred in" in msg_stripped
            assert "test_error_message_format_return" in msg_stripped
    
    def test_nested_error_path_reporting(self):
        """Test path reporting for errors nested within structures."""
        reporter = ErrorReporter()
        
        try:
            # Error: "not an int" should be int in list value for key "invalid"
            # Use instance_method with the correct dictionary structure
            reporter.instance_method({"valid": [1, 2, 3], "invalid": [1, "not an int", 3]}) 
            pytest.fail("Should have raised YouDiedError for nested type mismatch")
        except YouDiedError as e:
            # Basic checks - Ensure these match the specific failure
            assert isinstance(e.obituary, Obituary)
            assert e.cause == 'argument'
            assert e.obituary.expected_repr == "int" # Inner failure is str vs int
            assert e.obituary.received_repr == "str"
            assert e.obituary.value == "not an int" # Value should be the specific failing item

            # Check formatted path string in the error message
            msg_stripped = strip_ansi(str(e))
            # Correct assertion for formatted path based on code analysis for this input
            expected_formatted_path = "Path: `value.value('invalid')[1]`"
            assert expected_formatted_path in msg_stripped

            # Check raw path list
            # Correct assertion for raw path list based on code analysis for this input
            expected_path_list = ['value', "value('invalid')", 1]
            assert e.obituary.path == expected_path_list
            assert isinstance(e.obituary.path, list)

    def test_caller_variable_name_reporting(self):
        """Test that error messages include the variable name from caller scope."""
        # NOTE: The current implementation of _generate_error_message_core
        # does NOT explicitly include the caller's variable name in the formatted
        # message, although caller *function/file/line* are included.
        # This test might need adjustment based on desired behavior vs. implementation.
        try:
            specific_variable_name = "not an int"
            simple_func(specific_variable_name)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            error_msg = str(e)
            msg_stripped = strip_ansi(error_msg) # Strip ANSI

            # Check parameter name is present (this comes from the function def)
            assert "for argument 'value'" in msg_stripped
            # Check standard formatting
            assert "Expected: int" in msg_stripped
            assert "Value: 'not an int'" in msg_stripped
            assert "Path: `value`" in msg_stripped
            # Check caller *location* info IS present
            assert "Error occurred in" in msg_stripped
            assert "test_caller_variable_name_reporting" in msg_stripped
            
            # Assert that the specific *variable name* from the caller is NOT directly in the standard message
            # assert "specific_variable_name" not in msg_stripped # This depends on final formatting decision
    
    def test_error_in_mixed_context(self):
        """Test error reporting when errors happen in nested function calls."""
        @diecast
        def outer(value: List[int]) -> int:
            return inner(value)
        
        @diecast
        def inner(value: List[str]) -> int:  # Note: different type annotation
            return len("".join(value)) # Raises TypeError internally on join
        
        try:
            test_list = [1, 2, 3]  # List[int], not List[str] for inner
            outer(test_list)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            assert e.cause == 'argument'
            assert e.__cause__ is None
            error_msg = str(e)
            msg_stripped = strip_ansi(error_msg) # Strip ANSI

            assert "YouDiedError:" in msg_stripped
            assert "Argument mismatch in function 'tests.test_error_reporting.inner'" in msg_stripped
            assert "for argument 'value'" in msg_stripped # Error is in 'inner's arg check
            assert "Expected: str" in msg_stripped # 'inner' expected str elements
            assert "Value: 1" in msg_stripped # Check for the specific failing element
            assert "Path: `value[0]`" in msg_stripped # Path to the first failing element

            assert "Error occurred in" in msg_stripped
            assert "outer" in msg_stripped # Caller function was outer
        except Exception as test_e:
            pytest.fail(f"Test setup or execution failed: {test_e}")
    
    def test_generator_yield_error_format(self):
        """Test the error format for yield type errors in generators."""
        @diecast
        def generator_func() -> Iterator[int]:
            yield 1
            yield "not an int"  # Should fail here
            yield 3
        
        gen = generator_func()
        next(gen)  # First yield is fine
        
        try:
            next(gen)  # Second yield should fail
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            error_msg = str(e)
            msg_stripped = strip_ansi(error_msg) # Strip ANSI
            
            # Check header indicates yield value mismatch
            assert "YouDiedError:" in msg_stripped
            assert "Yield value mismatch in function 'tests.test_error_reporting.generator_func'" in msg_stripped
            
            # Check core details
            assert "Expected: int" in msg_stripped
            assert "Value: 'not an int'" in msg_stripped
            # No path expected for simple yield

            # Check caller info points to the next(gen) call site
            assert "Error occurred in" in msg_stripped
            assert "test_generator_yield_error_format" in msg_stripped
    
    def test_async_error_format(self):
        """Test error format for async function return type errors."""
        @diecast
        async def async_func() -> int:
            await asyncio.sleep(0.01)
            return "not an int"  # Should fail
        
        async def run_test():
            raised_correctly = False
            try:
                await async_func()
            except YouDiedError as e:
                raised_correctly = True
                error_msg = str(e)
                msg_stripped = strip_ansi(error_msg) # Strip ANSI

                # Check header
                assert "YouDiedError:" in msg_stripped
                assert "Return value mismatch in function 'tests.test_error_reporting.async_func'" in msg_stripped
                
                # Check core details
                assert "Expected: int" in msg_stripped
                assert "Value: 'not an int'" in msg_stripped
                # No path expected for simple return

                # Check caller info points to the await call site within run_test
                assert "Error occurred in" in msg_stripped
                assert "run_test" in msg_stripped
            return raised_correctly # Return True if exception was caught and checked
        
        # Run the async test
        result = asyncio.run(run_test())
        assert result, "Async error test failed or did not raise YouDiedError"

    def test_typevar_consistency_error_format(self):
        """Test error format for TypeVar consistency violations."""
        from typing import TypeVar, Tuple
        
        TEST_T = TypeVar('TEST_T')
        
        @diecast
        def simple_consistency(first: TEST_T, second: TEST_T) -> Tuple[TEST_T, TEST_T]:
            return (first, second)
        
        test_int = 123
        test_str = "test string" 
        
        try:
            simple_consistency(test_int, test_str) # Error on second arg
            pytest.fail("Expected YouDiedError was not raised")
        except YouDiedError as e:
            error_msg = str(e)
            msg_stripped = strip_ansi(error_msg) # Strip ANSI
            print(f"\nTypeVar consistency error message: {msg_stripped}\n") # For debugging if needed

            assert "YouDiedError:" in msg_stripped
            assert "Argument mismatch in function 'tests.test_error_reporting.simple_consistency'" in msg_stripped
            assert "for argument 'second'" in msg_stripped # Error on 'second'

            # Corrected Assertion: Check the basic expected type first
            assert "Expected: ~TEST_T" in msg_stripped 
            # Check that the specific reason, including the bound type, is present
            assert "Reason: TypeVar consistency violation: Expected ~TEST_T (Bound to: int) but received str" in msg_stripped 

            assert "Value: 'test string'" in msg_stripped
            assert "Path: `second`" in msg_stripped
            assert "Error occurred in" in msg_stripped
            assert "test_typevar_consistency_error_format" in msg_stripped

    # --- Tests moved from test_decorator.py for message format validation --- 

    def test_err_msg_basic_arg_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_basic_func("bad_type", 10) # Pass str instead of int for 'a'
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "for argument 'a'" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " int" in msg_stripped # Check for type name with potential leading space
        assert "Value: 'bad_type'" in msg_stripped
        assert "Path: `a`" in msg_stripped

    def test_err_msg_basic_return_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_nested_wrong_return(123) # Returns int instead of str
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Return value mismatch" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped 
        assert " str" in msg_stripped # Check for type name with potential leading space
        assert "Value: 123" in msg_stripped
        # No path

    def test_err_msg_optional_arg_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_optional_func(123)
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "for argument 'a'" in msg_stripped
        # Check Expected type components separately
        assert "Expected: str" in msg_stripped
        assert "Value: 123" in msg_stripped
        assert "Path: `a`" in msg_stripped

    def test_err_msg_optional_return_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_nested_wrong_optional_return("hello")
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Return value mismatch" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped 
        assert " str" in msg_stripped
        assert "Value: 123" in msg_stripped
        # No path

    def test_err_msg_union_arg_fail(self):
        """Test error message format for Union argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_union_func(a=[1.0])
        # Check for key parts of the message
        err_str = str(excinfo.value)
        assert "Argument mismatch" in err_str
        assert "_err_union_func" in err_str
        assert "argument 'a'" in err_str
        assert "Expected: Union[int, str]" in strip_ansi(err_str) 
        assert "received type list" in strip_ansi(err_str)
        assert "Reason: Value does not match any type in Union[int, str]" in strip_ansi(err_str)
        assert "Value: [1.0]" in strip_ansi(err_str)
        assert "Path: `a`" in strip_ansi(err_str)

    def test_err_msg_union_return_fail(self):
        """Test error message format for Union return failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_nested_wrong_union_return(5)
        # Check for key parts of the message
        err_str = str(excinfo.value)
        assert "Return value mismatch" in err_str
        assert "_err_nested_wrong_union_return" in err_str
        assert "Expected: Union[int, str]" in strip_ansi(err_str) 
        assert "received type list" in strip_ansi(err_str)
        assert "Reason: Value does not match any type in Union[int, str]" in strip_ansi(err_str)
        assert "Value: [5]" in strip_ansi(err_str)

    def test_err_msg_forward_ref_arg_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_forward_ref_func(123)
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "for argument 'target'" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " _ErrForwardRefTarget" in msg_stripped # Check resolved type name
        assert "Value: 123" in msg_stripped
        assert "Path: `target`" in msg_stripped

    def test_err_msg_method_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_instance.instance_method("bad")
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "instance_method" in msg_stripped
        assert "for argument 'x'" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " int" in msg_stripped
        assert "Value: 'bad'" in msg_stripped
        assert "Path: `x`" in msg_stripped

    def test_err_msg_classmethod_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _ErrSimpleClass.class_method(123)
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "class_method" in msg_stripped
        assert "for argument 'y'" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " str" in msg_stripped
        assert "Value: 123" in msg_stripped
        assert "Path: `y`" in msg_stripped

    def test_err_msg_staticmethod_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _ErrSimpleClass.static_method("bad")
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "static_method" in msg_stripped
        assert "for argument 'z'" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " bool" in msg_stripped
        assert "Value: 'bad'" in msg_stripped
        assert "Path: `z`" in msg_stripped

    @pytest.mark.asyncio
    async def test_err_msg_async_return_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            await _err_nested_bad_async() # Returns 123, expects str
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Return value mismatch" in msg_stripped
        assert "_err_nested_bad_async" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " str" in msg_stripped
        assert "Value: 123" in msg_stripped
        # No path

    def test_err_msg_typevar_constrained_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_typevar_constrained_func(1.5) # Pass float, constraints are int, str
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "for argument 'x'" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " ~T_ERR_Constrained" in msg_stripped # Check for ~ prefix
        assert "Value: 1.5" in msg_stripped
        assert "Path: `x`" in msg_stripped
        # Reason might be generic like "Value not in allowed types for constrained TypeVar"
        # assert "Reason:" in msg_stripped # Add specific reason check if implemented

    def test_err_msg_typevar_bound_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_typevar_bound_func(123) # Pass int, bound is Sequence
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "for argument 'x'" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " ~T_ERR_Bound" in msg_stripped # Check for ~ prefix
        assert "Value: 123" in msg_stripped
        assert "Path: `x`" in msg_stripped
        # Corrected Assertion: Match the message from the inner _check_simple_type failure
        e = excinfo.value # Get exception info
        assert isinstance(e.obituary, Obituary)
        assert e.obituary.received_repr == "int" # Check details from inner failure
        assert e.obituary.value == 123
        assert e.obituary.path == ["x"]
        assert e.obituary.message == "Value is not an instance of expected type"
        assert e.cause == 'argument'

    def test_err_msg_typevar_consistency_arg_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_typevar_consistency_func(10, "different") # Bind T to int, pass str for y
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "for argument 'y'" in msg_stripped
        assert "Expected:" in msg_stripped
        assert " ~T_ERR_Unconstrained" in msg_stripped
        assert "(Bound to: int)" in msg_stripped
        assert "Reason: TypeVar consistency violation" in msg_stripped
        assert "Value: 'different'" in msg_stripped
        assert "Path: `y`" in msg_stripped

    def test_err_msg_typevar_consistency_return_fail(self):
        with pytest.raises(YouDiedError) as excinfo:
            _err_nested_wrong_return_typevar(42) # Binds T to int, returns str
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Return value mismatch" in msg_stripped
        # Check Expected type components separately
        assert "Expected:" in msg_stripped
        assert " ~T_ERR_Unconstrained" in msg_stripped
        assert "(Bound to: int)" in msg_stripped
        assert "Value: 'wrong type'" in msg_stripped
        # No path
        assert "Reason: TypeVar consistency violation" in msg_stripped # Check specific reason

    def test_err_msg_typevar_consistency_in_class_fail(self):
        instance = _ErrClassWithConsistency[int]() 
        with pytest.raises(YouDiedError) as excinfo:
            instance.method(100, "bad") # Bind T to int, pass str for y
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "for argument 'y'" in msg_stripped
        assert "Expected:" in msg_stripped
        assert " ~T_ERR_Unconstrained" in msg_stripped
        assert "(Bound to: int)" in msg_stripped
        assert "Reason: TypeVar consistency violation" in msg_stripped
        assert "Value: 'bad'" in msg_stripped
        assert "Path: `y`" in msg_stripped

    def test_err_msg_inheritance_fail(self):
        parent_instance = _ErrParent()
        with pytest.raises(YouDiedError) as excinfo:
            _err_inheritance_func_b(parent_instance) # Expects Child, gets Parent
        msg = str(excinfo.value)
        msg_stripped = strip_ansi(msg) # Strip ANSI
        assert "Argument mismatch" in msg_stripped
        assert "for argument 'c'" in msg_stripped
        assert "Expected: _ErrChild" in msg_stripped
        assert "Value: <tests.test_error_reporting._ErrParent object" in msg_stripped # Check value repr
        assert "Path: `c`" in msg_stripped

    # --- START REFACTOR: Tests below will check attributes, not string format --- 

    def test_obituary_details_return(self):
        """Test that YouDiedError holds correct Obituary for return value mismatch."""
        @diecast
        def wrong_return(value: int) -> int:
            return str(value)
        
        try:
            wrong_return(42)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            assert isinstance(e.obituary, Obituary), "Exception should contain an Obituary object"
            assert e.obituary.expected_repr == "int", "Incorrect expected type in Obituary"
            assert e.obituary.received_repr == "str", "Incorrect received type in Obituary"
            assert e.obituary.value == "42", "Incorrect received value in Obituary"
            assert e.obituary.path == [], "Path should be empty for simple return error"
            # assert e.obituary.message is None # Or check for specific message if generated
            assert e.cause == 'return', "Incorrect error cause"

    def test_obituary_details_nested_path(self):
        """Test that Obituary captures the correct path for nested errors."""
        reporter = ErrorReporter()
        try:
            reporter.instance_method({"valid": [1, 2, 3], "invalid": [1, "not an int", 3]})
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "int"
            assert e.obituary.received_repr == "str"
            # Assert value is the specific failing element
            assert e.obituary.value == "not an int" # <-- Correct assertion
            assert e.obituary.path == ['value', "value('invalid')", 1]
            assert e.cause == 'argument'

    def test_obituary_details_caller_variable_name(self):
        """Test Obituary attributes when caller variable info might be relevant (but not stored). """
        # Note: Obituary itself doesn't store caller variable name.
        try:
            specific_variable_name = "not an int"
            simple_func(specific_variable_name)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "int"
            assert e.obituary.received_repr == "str"
            assert e.obituary.value == "not an int"
            assert e.obituary.path == ["value"]
            assert e.cause == 'argument'

    def test_obituary_details_mixed_context(self):
        """Test Obituary details in nested function calls."""
        @diecast
        def outer(value: List[int]) -> int:
            return inner(value)
        
        @diecast
        def inner(value: List[str]) -> int:
            return len("".join(value)) # Raises TypeError internally on join
        
        try:
            test_list = [1, 2, 3]
            outer(test_list)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            # As noted before, DieCast catches the argument error first
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "str" 
            assert e.obituary.received_repr == "int" 
            # Corrected value (Element that failed)
            assert e.obituary.value == 1 
            assert e.obituary.path == ['value', 0] 
            assert e.cause == 'argument'
            assert e.__cause__ is None
        except Exception as test_e:
            pytest.fail(f"Test setup or execution failed: {test_e}")

    def test_obituary_details_generator_yield(self):
        """Test Obituary details for yield type errors."""
        @diecast
        def generator_func() -> Iterator[int]:
            yield 1
            yield "not an int" # Error here
            yield 3
        
        gen = generator_func()
        next(gen)
        try:
            next(gen)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "int"
            assert e.obituary.received_repr == "str"
            assert e.obituary.value == "not an int"
            assert e.obituary.path == [] # Path empty for simple yield
            assert e.cause == 'yield', "Incorrect error cause"
        except Exception as test_e:
            pytest.fail(f"Test setup or execution failed: {test_e}")

    def test_obituary_details_async_return(self):
        """Test Obituary details for async function return type errors."""
        @diecast
        async def async_func() -> int:
            await asyncio.sleep(0.01)
            return "not an int"
        
        async def run_test():
            try:
                await async_func()
                pytest.fail("Should have raised YouDiedError")
            except YouDiedError as e:
                assert isinstance(e.obituary, Obituary)
                assert e.obituary.expected_repr == "int"
                assert e.obituary.received_repr == "str"
                assert e.obituary.value == "not an int"
                assert e.obituary.path == []
                assert e.cause == 'return'
                return True # Indicate success
            return False

        result = asyncio.run(run_test())
        assert result, "Async error test did not raise or validate correctly"

    def test_obituary_details_typevar_consistency(self):
        """Test Obituary details for TypeVar consistency violations."""
        from typing import TypeVar, Tuple
        TEST_T = TypeVar('TEST_T')
        
        @diecast
        def simple_consistency(first: TEST_T, second: TEST_T) -> Tuple[TEST_T, TEST_T]:
            return (first, second)
        
        try:
            simple_consistency(123, "test string")
            pytest.fail("Expected YouDiedError was not raised")
        except YouDiedError as e:
            assert isinstance(e.obituary, Obituary)
            # Expected type includes the bound type from the first arg
            assert e.obituary.expected_repr == "~TEST_T" # TypeVar representation 
            assert e.obituary.received_repr == "str"
            assert e.obituary.value == "test string"
            assert e.obituary.path == ["second"] # Path points to the inconsistent arg
            assert e.obituary.message == "TypeVar consistency violation: Expected ~TEST_T (Bound to: int) but received str" # Check reason
            assert e.cause == 'argument'
        except Exception as test_e:
            pytest.fail(f"Test setup or execution failed: {test_e}")

class TestMoldErrorReporting:
    """Tests focusing on error reporting when using the mold mechanism (checking attributes)."""

    def test_mold_obituary_return_format(self, temp_module):
        """Test Obituary attributes from mold-processed return error."""
        module_name = None
        try:
            code = """
from diecast import mold

def func(x: int) -> str:
    return x
mold()
"""
            module, module_path = temp_module(code)
            module_name = module.__name__
            with pytest.raises(YouDiedError) as excinfo:
                module.func(5)
            e = excinfo.value
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "str"
            assert e.obituary.received_repr == "int"
            assert e.obituary.value == 5
            assert e.obituary.path == []
            assert e.cause == 'return'
            # Check that the error message still contains key info for manual inspection
            msg_stripped = strip_ansi(str(e))
            assert f"Return value mismatch in function '{module_name}.func'" in msg_stripped
            assert "Value: 5" in msg_stripped

        except Exception as test_e:
            pytest.fail(f"Test setup or execution failed: {test_e}")

    def test_mold_obituary_typevar_consistency(self, temp_module):
        """Test Obituary attributes for mold-processed TypeVar consistency error."""
        module_name = None
        try:
            code = """
from diecast import mold
from typing import TypeVar, Tuple
T = TypeVar('T')
def consistent_func(x: T, y: T) -> T:
    return x
mold()
"""
            module, module_path = temp_module(code)
            module_name = module.__name__
            with pytest.raises(YouDiedError) as excinfo:
                module.consistent_func(123, "abc")
            e = excinfo.value
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "~T" # Bound to int
            assert e.obituary.received_repr == "str"
            assert e.obituary.value == "abc"
            assert e.obituary.path == ["y"]
            assert e.obituary.message == "TypeVar consistency violation: Expected ~T (Bound to: int) but received str"
            assert e.cause == 'argument'

        except Exception as test_e:
             pytest.fail(f"Test setup or execution failed: {test_e}")

    def test_mold_obituary_basic_arg_fail(self, temp_module):
        """Test Obituary attributes for mold-processed basic argument error."""
        module_name = None
        try:
            code = """
from diecast import mold
def basic_func(a: int, b: str) -> None: pass
mold()
"""
            module, module_path = temp_module(code)
            module_name = module.__name__
            with pytest.raises(YouDiedError) as excinfo:
                module.basic_func("wrong", 10)
            e = excinfo.value
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "int"
            assert e.obituary.received_repr == "str"
            assert e.obituary.value == "wrong"
            assert e.obituary.path == ["a"]
            assert e.cause == 'argument'

        except Exception as test_e:
             pytest.fail(f"Test setup or execution failed: {test_e}")

    def test_mold_obituary_basic_return_fail(self, temp_module):
        """Test Obituary attributes for mold-processed basic return error."""
        module_name = None
        try:
            code = """
from diecast import mold
def basic_return_func(x: int) -> str:
     return x
mold()
"""
            module, module_path = temp_module(code)
            module_name = module.__name__
            with pytest.raises(YouDiedError) as excinfo:
                module.basic_return_func(5)
            e = excinfo.value
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "str"
            assert e.obituary.received_repr == "int"
            assert e.obituary.value == 5
            assert e.obituary.path == []
            assert e.cause == 'return'

        except Exception as test_e:
             pytest.fail(f"Test setup or execution failed: {test_e}")

    def test_mold_obituary_processes_overridden(self, temp_module):
        """Test Obituary attributes for mold-processed overridden method error."""
        base_code = """
class BaseClass:
    def method_base(self, x: int) -> None: pass
"""
        base_module, base_module_path = temp_module(base_code)
        base_module_name = base_module.__name__
        child_module_name = None
        try:
            child_code = f"""
from diecast import mold
from {base_module_name} import BaseClass
class ChildClass(BaseClass):
    def method_base(self, x: int) -> None: pass
mold()
"""
            child_module, child_module_path = temp_module(child_code)
            child_module_name = child_module.__name__
            child_instance = child_module.ChildClass()
            with pytest.raises(YouDiedError) as excinfo:
                child_instance.method_base("wrong")
            e = excinfo.value
            assert isinstance(e.obituary, Obituary)
            assert e.obituary.expected_repr == "int"
            assert e.obituary.received_repr == "str"
            assert e.obituary.value == "wrong"
            assert e.obituary.path == ["x"]
            assert e.cause == 'argument'
        except Exception as test_e:
            pytest.fail(f"Test setup or execution failed: {test_e}")
        finally:
            if base_module_name and base_module_name in sys.modules: del sys.modules[base_module_name]
            if os.path.exists(base_module_path): os.remove(base_module_path)

    def test_mold_obituary_class_methods(self, temp_module):
        """Test Obituary attributes for mold-processed class/static methods."""
        module_name = None
        try:
            code = """
from diecast import mold
class MyClass:
    @classmethod
    def class_m(cls, s: str) -> None: pass
    @staticmethod
    def static_m(i: int) -> None: pass
mold()
"""
            module, module_path = temp_module(code)
            module_name = module.__name__

            # Test classmethod
            with pytest.raises(YouDiedError) as excinfo_cm:
                module.MyClass.class_m(False)
            e_cm = excinfo_cm.value
            assert isinstance(e_cm.obituary, Obituary)
            assert e_cm.obituary.expected_repr == "str"
            assert e_cm.obituary.received_repr == "bool"
            assert e_cm.obituary.value is False
            assert e_cm.obituary.path == ["s"]
            assert e_cm.cause == 'argument'

            # Test staticmethod
            with pytest.raises(YouDiedError) as excinfo_sm:
                module.MyClass.static_m("wrong")
            e_sm = excinfo_sm.value
            assert isinstance(e_sm.obituary, Obituary)
            assert e_sm.obituary.expected_repr == "int"
            assert e_sm.obituary.received_repr == "str"
            assert e_sm.obituary.value == "wrong"
            assert e_sm.obituary.path == ["i"]
            assert e_sm.cause == 'argument'

        except Exception as test_e:
            pytest.fail(f"Test setup or execution failed: {test_e}")