# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
import importlib.util
from typing import (
    Iterator,
    TypeVar,
    Tuple,
    Dict,
    List
)

# Check if _ctypes is available
has_ctypes = importlib.util.find_spec("_ctypes") is not None

## ===== THIRD PARTY ===== ##
import pytest

## ===== LOCAL IMPORTS ===== ##
# Use consistent import path relative to the package root
import sys
import os
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diecast.type_utils import YouDiedError
from src.diecast.error_utils import Obituary
from src.diecast import diecast # We import this last. Importing from a submodule of diecast would overwrite the decorator diecast in the namespace

# ===== HELPER CLASSES/FUNCTIONS ===== #
# Mocks needed specifically for the Obituary detail tests migrated from test_error_reporting

@diecast
def _mock_err_nested_wrong_return(a: int) -> str:
    return a # Returns int instead of str

@pytest.fixture
def get_mock_reporter():
    class _MockErrorReporter:
        @diecast
        def instance_method(self, value: Dict[str, List[int]]) -> List[str]:
            return list(value.keys())
    return _MockErrorReporter

@pytest.fixture
def get_mock_reporter_instance(get_mock_reporter):
    return get_mock_reporter()

@pytest.fixture
def get_mock_simple_func():
    @diecast
    def _mock_simple_func(value: int) -> str:
        return str(value)
    return _mock_simple_func

@pytest.fixture
def get_mock_outer():
    @diecast
    def _mock_inner(value: List[str]) -> int:  # Note: different type annotation
        return len("".join(value)) # Raises TypeError internally on join
    @diecast
    def _mock_outer(value: List[int]) -> int:
        return _mock_inner(value)
    return _mock_outer

@pytest.fixture
def get_mock_generator_func():
    @diecast
    def _mock_generator_func() -> Iterator[int]: # Note: Iterator is imported via typing
        yield 1
        yield "not an int"  # Should fail here
        yield 3
    return _mock_generator_func

@pytest.fixture
def get_mock_async_func():
    @diecast
    async def _mock_async_func() -> int:
        return "not an int"  # Should fail
    return _mock_async_func

_TEST_T_Obituary = TypeVar('_TEST_T_Obituary')

@pytest.fixture
def get_mock_simple_consistency():
    @diecast
    def _mock_simple_consistency(first: _TEST_T_Obituary, second: _TEST_T_Obituary) -> Tuple[_TEST_T_Obituary, _TEST_T_Obituary]:
        return (first, second)
    return _mock_simple_consistency


# ===== TEST CLASSES ===== #

class TestErrorUtilsUnit:
    """
    Unit tests focusing on the components within diecast.error_utils,
    particularly the structure and population of the Obituary object.
    Tests migrated from test_error_reporting.py that directly inspect Obituary.
    """

    def test_obituary_population_return(self):
        """Test Obituary attributes are correctly populated for return value errors."""
        with pytest.raises(YouDiedError) as excinfo:
            _mock_err_nested_wrong_return(5) # Trigger error

        obituary = excinfo.value.obituary
        assert isinstance(obituary, Obituary)
        assert excinfo.value.cause == 'return'
        assert obituary.expected_repr == "str"
        assert obituary.received_repr == "int"
        assert obituary.value == 5
        assert obituary.path == ['return'] # Path for return value
        assert "Value is not an instance of expected type" in obituary.message # Check specific reason from check_type

    def test_obituary_population_nested_path(self, get_mock_reporter_instance):
        """Test Obituary attributes, especially path, for nested argument errors."""
        with pytest.raises(YouDiedError) as excinfo:
            # Error: "not an int" should be int in list value for key "invalid"
            get_mock_reporter_instance.instance_method({"valid": [1, 2, 3], "invalid": [1, "not an int", 3]})

        obituary = excinfo.value.obituary
        assert isinstance(obituary, Obituary)
        assert excinfo.value.cause == 'argument'
        assert obituary.expected_repr == "int" # Inner failure is str vs int
        assert obituary.received_repr == "str"
        assert obituary.value == "not an int" # Value should be the specific failing item

        # Check raw path list
        expected_path_list = ['value', "value('invalid')", 1]
        assert obituary.path == expected_path_list
        assert isinstance(obituary.path, list)
        assert "Value is not an instance of expected type" in obituary.message

    def test_obituary_population_simple_arg(self, get_mock_simple_func):
        """Test Obituary attributes are correctly populated for simple argument errors."""
        try:
            specific_variable_name = "not an int"
            get_mock_simple_func(specific_variable_name)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            obituary = e.obituary
            assert isinstance(obituary, Obituary)
            assert e.cause == 'argument'
            assert obituary.expected_repr == "int"
            assert obituary.received_repr == "str"
            assert obituary.value == "not an int"
            assert obituary.path == ['value'] # Path for simple argument
            assert "Value is not an instance of expected type" in obituary.message

    def test_obituary_population_mixed_context(self, get_mock_outer):
        """Test Obituary attributes are correctly populated for argument errors in nested calls."""
        try:
            test_list = [1, 2, 3]  # List[int], not List[str] for inner
            get_mock_outer(test_list)
            pytest.fail("Should have raised YouDiedError")
        except YouDiedError as e:
            obituary = e.obituary
            assert isinstance(obituary, Obituary)
            assert e.cause == 'argument'
            # Error happens in _mock_inner checking its 'value' argument
            assert obituary.expected_repr == "str" # _mock_inner expected str elements
            assert obituary.received_repr == "int" # but got int
            assert obituary.value == 1 # The first failing element
            assert obituary.path == ['value', 0] # Path to the first failing element in the list arg
            assert "Value is not an instance of expected type" in obituary.message

    def test_obituary_population_generator_yield(self, get_mock_generator_func):
        """Test Obituary attributes for yield type errors."""
        gen = get_mock_generator_func()
        next(gen)  # First yield is fine

        with pytest.raises(YouDiedError) as excinfo:
            next(gen)  # Second yield should fail

        obituary = excinfo.value.obituary
        assert isinstance(obituary, Obituary)
        assert excinfo.value.cause == 'yield'
        assert obituary.expected_repr == "int"
        assert obituary.received_repr == "str"
        assert obituary.value == "not an int"
        assert obituary.path == ['return[Yield]'] # Path for yield value
        assert "Value is not an instance of expected type" in obituary.message

    @pytest.mark.anyio # Keep marker if testing async interactions
    @pytest.mark.skipif(
        not has_ctypes,
        reason="Skipping test because _ctypes module is not available"
    )
    async def test_obituary_population_async_return(self, get_mock_async_func):
        """Test Obituary attributes for async function return type errors."""
        with pytest.raises(YouDiedError) as excinfo:
             await get_mock_async_func()

        obituary = excinfo.value.obituary
        assert isinstance(obituary, Obituary)
        assert excinfo.value.cause == 'return'
        assert obituary.expected_repr == "int"
        assert obituary.received_repr == "str"
        assert obituary.value == "not an int"
        assert obituary.path == ['return'] # Path for return value
        assert "Value is not an instance of expected type" in obituary.message

    def test_obituary_population_typevar_consistency(self, get_mock_simple_consistency):
        """Test Obituary attributes for TypeVar consistency argument errors."""
        test_int = 123
        test_str = "test string"

        with pytest.raises(YouDiedError) as excinfo:
            get_mock_simple_consistency(test_int, test_str) # Error on second arg

        obituary = excinfo.value.obituary
        assert isinstance(obituary, Obituary)
        assert excinfo.value.cause == 'argument'
        # Expected type reflects the TypeVar name
        assert obituary.expected_repr == "~_TEST_T_Obituary (bound to int)"
        assert obituary.received_repr == "str"
        assert obituary.value == "test string"
        assert obituary.path == ['second'] # Path for the failing second argument
        assert "TypeVar consistency violation" in obituary.message
        assert "Bound to: int in this call" in obituary.message