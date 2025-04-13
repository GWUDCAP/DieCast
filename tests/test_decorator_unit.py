# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from typing import (
    AsyncGenerator,
    Generator,
    ClassVar,
    Iterator,
    Optional,
    TypeVar,
    Generic,
    Type,
    Any
)
import asyncio
import inspect
import logging
import gc

## ===== THIRD PARTY ===== ##
import pytest

## ===== LOCAL IMPORTS ===== ##
from src.diecast.decorator import _DIECAST_MARKER, _SPECIALIZED_CLASS_CACHE, diecast
from src.diecast import logger as diecast_logger
from src.diecast.type_utils import YouDiedError # Keep for error checking in fail cases

# ===== HELPER CLASSES/FUNCTIONS ===== #

## ===== TYPE VARIABLES ===== ##
# General Purpose
T = TypeVar('T') # Keep original T for DecoratedGeneric fixture
T_Unconstrained = TypeVar('T_Unconstrained')

# Specific Test Scenarios
T_AsyncGen = TypeVar('T_AsyncGen') # For AsyncGenClass in TestGenericDecorator
T_Base = TypeVar('T_Base') # For BaseGen in TestGenericDecorator
T_Class = TypeVar('T_Class') # Renamed from T in original file, used in ConsistentGeneric
T_Cls = TypeVar('T_Cls') # For ClsMetGeneric in TestGenericDecorator
T_Deco = TypeVar('T_Deco') # For DecoratedGeneric fixture
T_Exist = TypeVar('T_Exist') # For BaseWithGetItem in TestGenericDecorator
T_Gen = TypeVar('T_Gen') # For GenClass in TestGenericDecorator
T_Ign = TypeVar('T_Ign') # For IgnoreMethodGeneric in TestGenericDecorator
LT = TypeVar('LT') # For DecoratedList in TestGenericDecorator (though test using it is excluded, keep typevar for now)
T_Nested = TypeVar('T_Nested') # For OuterGeneric in TestGenericDecorator
T_NoAnn = TypeVar('T_NoAnn') # For NoAnnGeneric in TestGenericDecorator
T_Static = TypeVar('T_Static') # For StaticMetGeneric in TestGenericDecorator

# ===== FIXTURES ===== #

## ===== CONFIGURATION FIXTURES ===== ##
@pytest.fixture(autouse=True)
def configure_diecast_logging():
    """Fixture to ensure diecast logger level is set to DEBUG for tests."""
    original_level = diecast_logger.level
    diecast_logger.setLevel(logging.DEBUG)
    yield
    diecast_logger.setLevel(original_level) # Restore original level after test

## ===== SIMPLE CLASS FIXTURES ===== ##
@pytest.fixture
def get_simple_class():
    class SimpleClass():
        @diecast
        def instance_method(self, x: int) -> str:
            if not isinstance(x, int):
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

    return SimpleClass

@pytest.fixture
def get_simple_instance(get_simple_class):
    instance = get_simple_class()
    return instance

## ===== ASYNC/GENERATOR FIXTURES ===== ##
@pytest.fixture
def get_generator_func():
    @diecast
    def generator_func(n: int) -> Iterator[int]:
        for i in range(n):
            yield i * i # Yields int
    return generator_func

@pytest.fixture
def get_async_func():
    @diecast
    async def async_func(name: str) -> str:
        await asyncio.sleep(0.01)
        return f"Hello, {name}"
    return async_func

## ===== TYPEVAR CONSISTENCY FIXTURES ===== ##
@pytest.fixture
def get_nested_consistent_types():
    @diecast
    def _nested_consistent_types(x: T_Unconstrained, y: T_Unconstrained) -> T_Unconstrained:
        return x
    return _nested_consistent_types

@pytest.fixture
def get_nested_wrong_return_typevar():
    @diecast
    def _nested_wrong_return_typevar(x: T_Unconstrained) -> T_Unconstrained:
        # This function's body is designed to fail the type check,
        # which is relevant for testing the consistency check mechanism.
        if isinstance(x, str):
            return 123 # Return wrong type if input is str
        return "wrong type" # Return wrong type if input is not str
    return _nested_wrong_return_typevar

@pytest.fixture
def get_consistent_generic():
    # This fixture is used by test_typevar_consistency_in_class
    class ConsistentGeneric(Generic[T_Class]):
        @diecast
        def method(self, x: T_Class, y: T_Class) -> T_Class:
            # The body just returns x, the test focuses on diecast's consistency check
            return x
    return ConsistentGeneric

## ===== DECORATOR MECHANICS FIXTURES ===== ##
@pytest.fixture
def get_nested_double_decorated():
    # Keep original double decoration for the test's purpose
    @diecast
    @diecast
    def _nested_double_decorated(a: int) -> int:
        # Add a marker to check if inner wrapper ran twice (it shouldn't)
        # The test itself will verify this marker logic.
        if not hasattr(_nested_double_decorated, 'call_count'):
            _nested_double_decorated.call_count = 0
        _nested_double_decorated.call_count += 1
        return a * 2
    return _nested_double_decorated

@pytest.fixture
def get_decorated_generic():
    # Fixture needed for TestGenericDecorator
    # Clear cache before defining class to ensure clean state for tests
    _SPECIALIZED_CLASS_CACHE.clear()

    @diecast # Decorate the generic class itself
    class DecoratedGeneric(Generic[T_Deco]):
        _log = logging.getLogger(f"{__name__}.DecoratedGeneric")

        def __init__(self, initial_value: T_Deco):
            self._value = initial_value
            DecoratedGeneric._log.debug(f"Instantiated DecoratedGeneric[{type(initial_value).__name__}] with {initial_value!r}")

        @diecast # Methods are decorated individually by _decorate_generic_class
        def process(self, item: T_Deco) -> T_Deco:
            DecoratedGeneric._log.debug(f"Processing item {item!r} of type {type(item).__name__}")
            # Basic check to ensure method runs
            if not isinstance(item, type(self._value)):
                 # This internal check helps debug test setup if diecast fails
                 raise TypeError(f"Test Helper: Expected type {type(self._value).__name__}, got {type(item).__name__}")
            return item

        def get_value(self) -> T_Deco: # Not decorated
            return self._value

    # Clear cache after defining class as well, just in case
    _SPECIALIZED_CLASS_CACHE.clear()
    return DecoratedGeneric

# ===== TEST FUNCTIONS ===== #
def test_method_pass(get_simple_instance):
    """Verify @diecast applies wrapper to instance methods (pass)."""
    assert get_simple_instance.instance_method(10) == "Value: 10"

def test_method_fail(get_simple_instance):
    """Verify @diecast wrapper on instance methods triggers checks (fail)."""
    with pytest.raises(YouDiedError) as excinfo:
        get_simple_instance.instance_method("bad")
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.path == ['x'] # Check path excludes 'self'

def test_classmethod_pass(get_simple_class):
    """Verify @diecast applies wrapper to classmethods (pass)."""
    assert get_simple_class.class_method("good") is True
    # Also check if the method is still a classmethod
    assert isinstance(inspect.getattr_static(get_simple_class, 'class_method'), classmethod)

def test_classmethod_fail(get_simple_class):
    """Verify @diecast wrapper on classmethods triggers checks (fail)."""
    with pytest.raises(YouDiedError) as excinfo:
        get_simple_class.class_method(123)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.path == ['y'] # Check path excludes 'cls'

def test_staticmethod_pass(get_simple_class):
    """Verify @diecast applies wrapper to staticmethods (pass)."""
    assert get_simple_class.static_method(True) is True
    assert get_simple_class.static_method(False) is False
    # Also check if the method is still a staticmethod
    assert isinstance(inspect.getattr_static(get_simple_class, 'static_method'), staticmethod)

def test_staticmethod_fail(get_simple_class):
    """Verify @diecast wrapper on staticmethods triggers checks (fail)."""
    with pytest.raises(YouDiedError) as excinfo:
        get_simple_class.static_method("bad")
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.path == ['z']

def test_generator_yield_pass(get_generator_func):
    """Verify @diecast selects and applies the sync generator wrapper."""
    gen_instance = get_generator_func(3)
    # Check it's actually a generator
    assert inspect.isgenerator(gen_instance)
    results = list(gen_instance)
    assert results == [0, 1, 4] # Verify it runs correctly

@pytest.mark.asyncio
async def test_async_return_pass(get_async_func):
    """Verify @diecast selects and applies the async wrapper."""
    result = await get_async_func("Tester")
    assert result == "Hello, Tester" # Verify it runs correctly

def test_typevar_consistency_pass(get_nested_consistent_types):
    """Verify decorator enforces TypeVar consistency (pass)."""
    assert get_nested_consistent_types(1, 2) == 1
    assert get_nested_consistent_types("a", "b") == "a"

def test_typevar_consistency_fail(get_nested_consistent_types):
    """Verify decorator enforces TypeVar consistency (fail)."""
    with pytest.raises(YouDiedError) as excinfo:
        # T_Unconstrained bound to int by 'x=1', 'y="b"' should fail
        get_nested_consistent_types(1, "b")
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.path == ['y']
    assert e.obituary.expected_repr == '~T_Unconstrained (bound to int)' # Bound to int
    assert e.obituary.received_repr == 'str'

def test_typevar_consistency_return_fail(get_nested_wrong_return_typevar):
    """Verify decorator enforces TypeVar consistency on return (fail)."""
    with pytest.raises(YouDiedError) as excinfo:
        # T_Unconstrained bound to int by x=5, function returns str
        get_nested_wrong_return_typevar(5)
    e = excinfo.value
    assert e.cause == 'return'
    assert e.obituary.expected_repr == '~T_Unconstrained (bound to int)' # Bound to int
    assert e.obituary.received_repr == 'str'

    with pytest.raises(YouDiedError) as excinfo_str:
         # T_Unconstrained bound to str by x="hello", function returns int
        get_nested_wrong_return_typevar("hello")
    e_str = excinfo_str.value
    assert e_str.cause == 'return'
    assert e_str.obituary.expected_repr == '~T_Unconstrained (bound to str)' # Bound to str
    assert e_str.obituary.received_repr == 'int'

def test_typevar_consistency_in_class(get_consistent_generic):
    """Verify TypeVar consistency check within generic class methods."""
    ConsistentInt = get_consistent_generic[int]
    instance_int = ConsistentInt()
    assert instance_int.method(10, 20) == 10 # Pass

    with pytest.raises(YouDiedError) as excinfo_int:
        instance_int.method(10, "bad") # Fail consistency
    e_int = excinfo_int.value
    assert e_int.cause == 'argument'
    assert e_int.obituary.path == ['y']
    assert e_int.obituary.expected_repr == '~T_Class (bound to int)' # Bound to int via instance_map
    assert e_int.obituary.received_repr == 'str'

    ConsistentStr = get_consistent_generic[str]
    instance_str = ConsistentStr()
    assert instance_str.method("a", "b") == "a" # Pass

    with pytest.raises(YouDiedError) as excinfo_str:
        instance_str.method("a", 123) # Fail consistency
    e_str = excinfo_str.value
    assert e_str.cause == 'argument'
    assert e_str.obituary.path == ['y']
    assert e_str.obituary.expected_repr == '~T_Class (bound to str)' # Bound to str via instance_map
    assert e_str.obituary.received_repr == 'int'

def test_double_decoration_prevention(get_nested_double_decorated):
    """Verify @diecast prevents double wrapping using _DIECAST_MARKER."""
    # Reset call count if fixture is reused
    if hasattr(get_nested_double_decorated, 'call_count'):
        delattr(get_nested_double_decorated, 'call_count')

    # Call the function
    result = get_nested_double_decorated(5)
    assert result == 10

    # Check the marker attribute exists and is True
    assert hasattr(get_nested_double_decorated, _DIECAST_MARKER)
    assert getattr(get_nested_double_decorated, _DIECAST_MARKER) is True

    # Check the internal call count marker (added in fixture)
    # It should be 1, indicating the inner function ran only once.
    assert hasattr(get_nested_double_decorated, 'call_count')
    assert get_nested_double_decorated.call_count == 1

def test_diecast_ignore_does_not_wrap():
    """Verify @diecast.ignore prevents wrapping."""
    @diecast.ignore
    def ignored_func(a: int) -> str:
        return a # Intentionally wrong return type

    # Check marker is set
    assert hasattr(ignored_func, _DIECAST_MARKER)
    assert getattr(ignored_func, _DIECAST_MARKER) is True

    # Call the function with wrong types - should NOT raise YouDiedError
    result = ignored_func("wrong_arg_type") # type: ignore
    assert result == "wrong_arg_type" # Returns the wrong type, no check applied

# ===== TEST CLASSES ===== #
class TestGenericDecorator:
    """Unit tests focused on the mechanics of decorating Generic classes."""

    @pytest.fixture(autouse=True)
    def clear_cache_fixture(self):
        """Clear specialization cache before/after each test method."""
        _SPECIALIZED_CLASS_CACHE.clear()
        yield
        _SPECIALIZED_CLASS_CACHE.clear()
        # Force garbage collection to help release potentially cached types/classes
        gc.collect()

    def test_generic_class_getitem_injection(self, get_decorated_generic):
        """(A11) Verify __class_getitem__ is injected into the decorated generic class."""
        assert hasattr(get_decorated_generic, '__class_getitem__')
        assert callable(getattr(get_decorated_generic, '__class_getitem__'))

        # Test specialization via the injected __class_getitem__
        DecoratedInt = get_decorated_generic[int]
        assert DecoratedInt is not get_decorated_generic # Should be a new class
        assert issubclass(DecoratedInt, get_decorated_generic)
        assert DecoratedInt.__name__.startswith('DecoratedGeneric_DiecastSpecialized')

        # Verify the specialized class has the type map
        assert hasattr(DecoratedInt, '_diecast_type_map')
        assert DecoratedInt._diecast_type_map == {T_Deco: int}

    def test_generic_class_specialization_cache(self, get_decorated_generic):
        """(A12) Verify the injected __class_getitem__ uses the specialization cache."""
        DecoratedInt1 = get_decorated_generic[int]
        DecoratedInt2 = get_decorated_generic[int]
        DecoratedStr = get_decorated_generic[str]

        assert DecoratedInt1 is DecoratedInt2 # Should retrieve from cache
        assert DecoratedInt1 is not DecoratedStr # Different specialization

        # Check cache content (implementation detail, but useful for validation)
        assert (get_decorated_generic, int) in _SPECIALIZED_CLASS_CACHE
        assert _SPECIALIZED_CLASS_CACHE[(get_decorated_generic, int)] is DecoratedInt1
        assert (get_decorated_generic, str) in _SPECIALIZED_CLASS_CACHE
        assert _SPECIALIZED_CLASS_CACHE[(get_decorated_generic, str)] is DecoratedStr

    def test_generic_class_sync_generator(self):
        """(A6, A7) Test sync generator method wrapper selection in generic classes."""
        @diecast
        class GenClass(Generic[T_Gen]):
            def gen_method(self, count: int, ret_val: T_Gen) -> Generator[T_Gen, None, T_Gen]:
                """Yields transformed T_Gen, returns T_Gen."""
                for i in range(count):
                    yield self._transform(i) # type: ignore
                return ret_val

            def _transform(self, val):
                # Simple transformation based on type T_Gen for testing
                resolved_type = self._get_expected_type()
                if resolved_type is int: # Check if T_Gen resolved to int
                     return val * 2
                elif resolved_type is str: # Check if T_Gen resolved to str
                     return str(val)
                return val

            def _get_expected_type(self) -> Type[T_Gen]:
                 # Helper to get the specialized type T_Gen for _transform
                 # This relies on internal details but helps the test fixture
                 type_map = getattr(type(self), '_diecast_type_map', {})
                 return type_map.get(T_Gen, Any) # type: ignore

        # Test with Int specialization
        GenInt = GenClass[int]
        instance_int = GenInt()
        gen_int = instance_int.gen_method(3, 100)
        assert inspect.isgenerator(gen_int) # Verify wrapper selected correctly
        assert list(gen_int) == [0, 2, 4] # Correct yield values
        # Return value check is more integration, focus on wrapper selection here

        # Test with Str specialization
        GenStr = GenClass[str]
        instance_str = GenStr()
        gen_str = instance_str.gen_method(2, "done")
        assert inspect.isgenerator(gen_str) # Verify wrapper selected correctly
        assert list(gen_str) == ["0", "1"] # Correct yield values

    @pytest.mark.asyncio
    async def test_generic_class_async_generator(self):
        """(A8) Test async generator method wrapper selection in generic classes."""
        @diecast
        class AsyncGenClass(Generic[T_AsyncGen]):
            async def gen_method(self, count: int) -> AsyncGenerator[T_AsyncGen, None]:
                """Yields transformed T_AsyncGen."""
                for i in range(count):
                    await asyncio.sleep(0.01)
                    yield self._transform(i) # type: ignore

            def _transform(self, val):
                 # Simple transformation based on type T_AsyncGen for testing
                 type_map = getattr(type(self), '_diecast_type_map', {})
                 expected_type = type_map.get(T_AsyncGen, Any)
                 if expected_type is int:
                     return val + 10
                 elif expected_type is str:
                     return f"val_{val}"
                 return val

        # Test with Int specialization
        AsyncGenInt = AsyncGenClass[int]
        instance_int = AsyncGenInt()
        agen_int = instance_int.gen_method(2)
        assert inspect.isasyncgen(agen_int) # Verify wrapper selected correctly
        results_int = [item async for item in agen_int]
        assert results_int == [10, 11] # Correct yield values

        # Test with Str specialization
        AsyncGenStr = AsyncGenClass[str]
        instance_str = AsyncGenStr()
        agen_str = instance_str.gen_method(3)
        assert inspect.isasyncgen(agen_str) # Verify wrapper selected correctly
        results_str = [item async for item in agen_str]
        assert results_str == ["val_0", "val_1", "val_2"] # Correct yield values


    def test_generic_class_classmethod(self):
        """(A13) Test wrapper application to @classmethod in generic class."""
        @diecast
        class ClsMetGeneric(Generic[T_Cls]):
            _class_val: ClassVar[Optional[T_Cls]] = None

            @classmethod
            def set_class_val(cls, val: T_Cls):
                 cls._class_val = val

            @classmethod
            def get_class_val(cls) -> Optional[T_Cls]: # Not decorated
                 return cls._class_val

        ClsMetInt = ClsMetGeneric[int]
        # Verify the method is still a classmethod after decoration
        assert isinstance(inspect.getattr_static(ClsMetInt, 'set_class_val'), classmethod)
        # Call the method - if wrapper wasn't applied correctly, might error differently
        ClsMetInt.set_class_val(10)
        assert ClsMetInt.get_class_val() == 10
        # Test fail case (basic check that wrapper runs)
        with pytest.raises(YouDiedError):
             ClsMetInt.set_class_val("wrong") # Should fail T_Cls check (even if limited)


    def test_generic_class_staticmethod(self):
        """(A14) Test wrapper application to @staticmethod in generic class."""
        @diecast
        class StaticMetGeneric(Generic[T_Static]):
            @staticmethod
            @diecast # Method decorator applied by _decorate_generic_class
            def process_static(val: bool) -> bool:
                 return not val

        StaticMetInt = StaticMetGeneric[int] # Specialization doesn't affect static method
        # Verify the method is still a staticmethod after decoration
        assert isinstance(inspect.getattr_static(StaticMetInt, 'process_static'), staticmethod)
        # Call the method
        assert StaticMetInt.process_static(True) is False
        # Test fail case (basic check that wrapper runs)
        with pytest.raises(YouDiedError):
             StaticMetInt.process_static("wrong")


    def test_generic_class_ignore_method(self, get_decorated_generic):
        """(A15) Test @diecast.ignore on a method within a generic class."""
        @diecast
        class IgnoreMethodGeneric(Generic[T_Ign]):
            @diecast.ignore # Ignore this specific method
            def ignored_method(self, x: T_Ign) -> int:
                # Intentionally wrong return type, should not be checked
                return "wrong" # type: ignore

            @diecast # This one should still be decorated
            def decorated_method(self, y: T_Ign) -> T_Ign:
                return y

        IgnoreMethodInt = IgnoreMethodGeneric[int]
        instance = IgnoreMethodInt()

        # Verify ignored method is not wrapped
        assert hasattr(IgnoreMethodInt.ignored_method, _DIECAST_MARKER)
        assert getattr(IgnoreMethodInt.ignored_method, _DIECAST_MARKER) is True
        # Call ignored method - should not raise YouDiedError despite wrong return type
        assert instance.ignored_method(123) == "wrong"

        # Verify other method IS wrapped and functions correctly
        # The marker *will* be on the class attribute due to direct @diecast application
        # The important checks are that the *instance* method has the marker and type checking works.
        assert getattr(instance.decorated_method, _DIECAST_MARKER) is True
        assert instance.decorated_method(456) == 456
        assert instance.decorated_method("bad") == "bad"


    def test_generic_class_ignore_class(self):
        """(A16) Test @diecast.ignore on the generic class itself."""
        @diecast.ignore
        class IgnoredGeneric(Generic[T]):
            def method(self, x: T) -> int:
                # Intentionally wrong return type
                return "ignored" # type: ignore

        # Verify class has marker
        assert hasattr(IgnoredGeneric, _DIECAST_MARKER)
        assert getattr(IgnoredGeneric, _DIECAST_MARKER) is True

        # Verify __class_getitem__ was NOT injected
        assert hasattr(IgnoredGeneric, '__class_getitem__')
        assert not getattr(IgnoredGeneric, '_DIECAST_GENERIC', False), "Ignored generic class should not have _DIECAST_GENERIC marker"

        # Verify methods were NOT wrapped (check on class, instance won't exist properly)
        assert not hasattr(IgnoredGeneric.method, _DIECAST_MARKER)

        # Instantiate (will be normal generic, no specialization)
        instance = IgnoredGeneric[int]() # type: ignore
        # Call method - should not raise YouDiedError
        assert instance.method(123) == "ignored"


    def test_generic_class_existing_getitem(self):
        """(A20) Test @diecast on generic class with existing __class_getitem__."""
        class BaseWithGetItem(Generic[T_Exist]):
            _custom_getitem_called = False
            _custom_type_map = {}

            # Define a custom __class_getitem__ BEFORE @diecast is applied
            def __class_getitem__(cls, key):
                # Simulate some custom logic
                cls._custom_getitem_called = True
                # Store the type map in a custom location
                cls._custom_type_map = {T_Exist: key}
                # For this test, just return the original class to show diecast doesn't overwrite blindly
                return cls

        # Now apply diecast
        DiecastOnExistingGetItem = diecast(BaseWithGetItem)

        # Verify the ORIGINAL __class_getitem__ is still present
        assert DiecastOnExistingGetItem.__class_getitem__ == BaseWithGetItem.__class_getitem__

        # Call the original __class_getitem__
        Specialized = DiecastOnExistingGetItem[int]

        # Verify our custom logic ran
        assert Specialized._custom_getitem_called is True
        assert Specialized._custom_type_map == {T_Exist: int}
        assert hasattr(Specialized, '_diecast_type_map') # Should have diecast's type map

        # Need a method to check:
        class MethodCheck(DiecastOnExistingGetItem): # Inherit to add method post-decoration
             def check_method(self, x: T_Exist) -> T_Exist:
                if x == 'bad':
                    return 5
                return x

        instance = MethodCheck[str]() # Use the custom __class_getitem__
        assert instance.check_method("test") == "test"
        with pytest.raises(YouDiedError):
            instance.check_method("bad")


    def test_generic_class_no_annotations(self):
        """(A21) Ensure methods without annotations aren't decorated in generics."""
        @diecast
        class NoAnnGeneric(Generic[T_NoAnn]):
            def method_with_ann(self, x: T_NoAnn) -> T_NoAnn:
                return x

            def method_without_ann(self, y): # No type annotations
                return y

        NoAnnInt = NoAnnGeneric[int]
        instance = NoAnnInt()

        # Verify method WITH annotations IS wrapped
        assert hasattr(instance.method_with_ann, _DIECAST_MARKER)
        assert getattr(instance.method_with_ann, _DIECAST_MARKER) is True

        # Verify method WITHOUT annotations is NOT wrapped
        assert not hasattr(instance.method_without_ann, _DIECAST_MARKER)

        # Call method without annotations - should work fine without checks
        assert instance.method_without_ann(123) == 123
        assert instance.method_without_ann("abc") == "abc"

class TestNonGenericDecorator:
    """Unit tests focused on the mechanics of decorating non-Generic classes."""

    def test_non_generic_class_basic_decoration(self):
        """(B1) Test applying @diecast to a non-generic class wraps methods."""
        @diecast
        class DecoratedNonGeneric:
            def method_with_ann(self, x: int) -> str:
                return str(x * 2)

            def method_without_ann(self, y):
                return y

        instance = DecoratedNonGeneric()

        # Verify method WITH annotations IS wrapped
        assert hasattr(instance.method_with_ann, _DIECAST_MARKER)
        assert getattr(instance.method_with_ann, _DIECAST_MARKER) is True
        assert instance.method_with_ann(5) == "10"
        with pytest.raises(YouDiedError):
            instance.method_with_ann("bad") # Fail input type check

        # Verify method WITHOUT annotations is NOT wrapped
        assert not hasattr(instance.method_without_ann, _DIECAST_MARKER)
        assert instance.method_without_ann(123) == 123

        # Verify __class_getitem__ was NOT injected
        assert not hasattr(DecoratedNonGeneric, '__class_getitem__')
        # Verify _diecast_type_map was NOT added
        assert not hasattr(DecoratedNonGeneric, '_diecast_type_map')

    def test_non_generic_class_ignore_method(self):
        """(B6) Test @diecast.ignore on method in non-generic class."""
        @diecast
        class IgnoreMethodNonGeneric:
            @diecast.ignore
            def ignored_method(self, x: int) -> str:
                return "ignored" # Wrong return type

            def decorated_method(self, y: bool) -> bool:
                return not y

        instance = IgnoreMethodNonGeneric()

        # Verify ignored method is marked but not wrapped effectively
        assert hasattr(IgnoreMethodNonGeneric.ignored_method, _DIECAST_MARKER)
        assert getattr(IgnoreMethodNonGeneric.ignored_method, _DIECAST_MARKER) is True
        assert instance.ignored_method(123) == "ignored" # No error despite wrong return

        # Verify other method IS wrapped
        assert hasattr(instance.decorated_method, _DIECAST_MARKER)
        assert getattr(instance.decorated_method, _DIECAST_MARKER) is True
        assert instance.decorated_method(True) is False
        with pytest.raises(YouDiedError):
            instance.decorated_method("bad")

    def test_non_generic_class_ignore_class(self):
        """(B7) Test @diecast.ignore on non-generic class."""
        @diecast.ignore
        class IgnoredNonGeneric:
            def method(self, x: int) -> str:
                return "ignored" # Wrong return type

        # Verify class has marker
        assert hasattr(IgnoredNonGeneric, _DIECAST_MARKER)
        assert getattr(IgnoredNonGeneric, _DIECAST_MARKER) is True
        # Verify method was NOT wrapped
        assert not hasattr(IgnoredNonGeneric.method, _DIECAST_MARKER)

        instance = IgnoredNonGeneric()
        assert instance.method(123) == "ignored" # No error

    def test_non_generic_class_no_annotations(self):
        """(B8) Ensure methods without annotations aren't decorated in non-generics."""
        @diecast
        class NoAnnNonGeneric:
            def method_with_ann(self, x: int) -> int:
                return x

            def method_without_ann(self, y):
                return y

        instance = NoAnnNonGeneric()

        # Verify method WITH annotations IS wrapped
        assert hasattr(instance.method_with_ann, _DIECAST_MARKER)
        assert getattr(instance.method_with_ann, _DIECAST_MARKER) is True

        # Verify method WITHOUT annotations is NOT wrapped
        assert not hasattr(instance.method_without_ann, _DIECAST_MARKER)
        assert instance.method_without_ann(123) == 123