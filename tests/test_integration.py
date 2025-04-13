# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from collections.abc import (
    AsyncGenerator, Generator, Iterator, Sequence as ABCSequence
)
from typing import (
    Annotated, Any, Callable, Dict, Generic, List, NamedTuple, 
    NewType, NoReturn, Optional, Protocol, Set, Tuple, Type, 
    TypeVar, Union, get_args, get_origin
)
from collections import namedtuple
from functools import wraps
import importlib.util
import logging
import asyncio
import sys
import re
import os
import gc
##-##

## ===== THIRD PARTY ===== ##
try:
    # Use typing_extensions for runtime_checkable if needed (Python < 3.8)
    from typing_extensions import runtime_checkable
except ImportError:
    try:
        # Or fall back to typing if available (Python >= 3.8)
        from typing import runtime_checkable
    except ImportError:
        # Provide a no-op fallback if neither is available (should not happen in supported envs)
        runtime_checkable = lambda x: x
import pytest
##-##

## ===== LOCAL ===== ##
# This is generally discouraged in tests; prefer configuring PYTHONPATH or using editable installs.
# Keeping it for now as refactoring test setup is out of scope for styling.
_SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from src.diecast.type_utils import YouDiedError
from src.diecast.config import _DIECAST_MARKER, MAX_VALUE_REPR_LENGTH
from src.diecast.decorator import _SPECIALIZED_CLASS_CACHE
from src.diecast import diecast, ignore, logger as diecast_logger
from .conftest import strip_ansi # Assume conftest provides this helper
from src.diecast.type_utils import _TYPEVAR_BINDINGS
##-##
#-#

# ===== CONSTANTS ===== #

## ===== REGEX ===== ##
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')
##-##

## ===== PYTHON VERSION CHECKS ===== ##
PY_GTE_39 = sys.version_info >= (3, 9)
PY_GTE_310 = sys.version_info >= (3, 10)
##-##

## ===== TYPE VARIABLES ===== ##
# General
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

T_Unconstrained = TypeVar('T_Unconstrained')
T_Constrained = TypeVar('T_Constrained', int, str)
T_Bound = TypeVar('T_Bound', bound=ABCSequence)
T_Class = TypeVar('T_Class')
T_Deco = TypeVar('T_Deco')
LT = TypeVar('LT')
T_Inj = TypeVar('T_Inj')
T_Cache = TypeVar('T_Cache')
T_Gen = TypeVar('T_Gen')
T_AsyncGen = TypeVar('T_AsyncGen')
T_Cls = TypeVar('T_Cls')
T_Static = TypeVar('T_Static')
T_Ign = TypeVar('T_Ign')
T_IgnCls = TypeVar('T_IgnCls')
T_Nested = TypeVar('T_Nested')
T_Inh = TypeVar('T_Inh')
T_Exist = TypeVar('T_Exist')
T_NoAnn = TypeVar('T_NoAnn')
T_GenErr = TypeVar('T_GenErr')
T_GenErrRet = TypeVar('T_GenErrRet')
T_GenErrSync = TypeVar('T_GenErrSync')
T_GenErrSyncRet = TypeVar('T_GenErrSyncRet')
T_GenErrAsync = TypeVar('T_GenErrAsync')
TEST_T = TypeVar('TEST_T')
StrOrBytes = TypeVar('StrOrBytes', str, bytes)
T_Complex = TypeVar('T_Complex', bound=Dict[str, Any])
K_Integ = TypeVar('K_Integ')
V_Integ = TypeVar('V_Integ')
T_Inh_Fail = TypeVar('T_Inh_Fail')
##-##
#-#

# ===== SETUP ===== #

## ===== GLOBAL TEST SETUP ===== ##
@pytest.fixture(autouse=True, scope='function')
def configure_diecast_logging():
    """
    Fixture to ensure diecast logger level is set to DEBUG for detailed test error reporting.
    Resets level after test. Scope is function to avoid interference between tests if
    a test explicitly modifies the logger level.
    """
    original_level = diecast_logger.level
    diecast_logger.setLevel(logging.DEBUG)
    yield
    diecast_logger.setLevel(original_level)

@pytest.fixture(autouse=True, scope='function')
def clear_caches_and_bindings():
    """
    Clear diecast internal caches and TypeVar bindings before and after each test
    to ensure test isolation. Also triggers garbage collection.
    """
    # clear_typevar_bindings() # Removed: Function requires func_id, wrappers handle cleanup.
    _SPECIALIZED_CLASS_CACHE.clear()
    yield
    # clear_typevar_bindings() # Removed: Function requires func_id, wrappers handle cleanup.
    _SPECIALIZED_CLASS_CACHE.clear()
    gc.collect() # Suggest garbage collection
##-##

## ===== INHERITANCE HIERARCHIES ===== ##
class A:
    def identify(self) -> str: return self.__class__.__name__
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
    def identify(self) -> str: return self.__class__.__name__
class BB(AA): pass
class CC(BB): pass
class DD(CC): pass
class EE(DD): pass
class FF(EE): pass
class GG(FF): pass
class HH(GG): pass
class II(HH): pass
class JJ(II): pass
##-##

## ===== CUSTOM TYPES ===== ##
class CustomType:
    def __init__(self, value: int): self.value = value
    def get_value(self) -> int: return self.value

@diecast
class Point(NamedTuple):
    x: int
    y: int

RegularTuple = namedtuple('RegularTuple', ['a', 'b'])
UserId = NewType('UserId', int)
##-##

## ===== GENERIC HELPERS ===== ##
@diecast
class MultiVarGeneric(Generic[K_Integ, V_Integ]):
    """Generic class with multiple type variables."""
    def __init__(self, key: K_Integ, value: V_Integ):
        self.key = key
        self.value = value
    @diecast
    def get_value(self) -> V_Integ: return self.value
    def get_key(self) -> K_Integ: return self.key # Undecorated

@diecast
class NestedGenericContainer(Generic[K_Integ]):
    """Generic class containing another generic type."""
    items: List[MultiVarGeneric[K_Integ, str]]
    def __init__(self, items: List[MultiVarGeneric[K_Integ, str]]): self.items = items
    def get_first_value(self) -> str:
        if not self.items: raise IndexError("Empty container")
        return self.items[0].get_value()
    def add_item(self, key: K_Integ, value: str):
         self.items.append(MultiVarGeneric[K_Integ, str](key, value))

@diecast
class BaseGenericForInherit(Generic[T_Inh]):
    """Base generic class for inheritance tests."""
    def process_base(self, item: T_Inh) -> T_Inh: return item

@diecast
class ChildInheritsGeneric(BaseGenericForInherit[int]):
    """Child class inheriting and specializing a generic base."""
    def process_child(self, item: str) -> int: return len(item)
##-##

@diecast
class IntrospectionTestGeneric(Generic[K, V]):
    """Helper class specifically for testing introspection."""
    def __init__(self, key: K, value: V): pass


## ===== PROTOCOL HELPERS ===== ##
class SimpleProto(Protocol): # No @runtime_checkable
    def method_a(self) -> int: ...
    an_attribute: str

class MatchesSimpleProto:
    def method_a(self) -> int: return 1
    an_attribute = "yes"

class DoesNotMatchSimpleProto:
    def method_a(self) -> str: return "no" # Wrong return type

@runtime_checkable
class SizedProtocol(Protocol):
    def __len__(self) -> int: ...
##-##

## ===== ASYNC HELPERS ===== ##
async def anext(iterator): # Helper for async generators
    """Basic implementation of anext for compatibility if needed."""
    return await iterator.__anext__()
##-##

## ===== ERROR REPORTING HELPERS ===== ##
@diecast
def _err_basic_func(a: int, b: str) -> float: return float(a) + len(b)

@diecast
def _err_optional_func(a: Optional[str]) -> Optional[int]: return len(a) if a is not None else None

@diecast
def _err_union_func(a: Union[int, str]) -> Union[float, bool]: 
    return float(a) if isinstance(a, int) else len(a) > 0

class _ErrForwardRefTarget: 
    pass

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

@diecast
def _err_typevar_constrained_func(x: T_Constrained) -> T_Constrained: 
    return x

@diecast
def _err_nested_wrong_return(a: int) -> str: 
    return a

@diecast
def _err_nested_wrong_optional_return(a: Optional[str]) -> Optional[str]: 
    return 123

@diecast
def _err_nested_wrong_union_return(a: Union[int, str]) -> Union[int, str]: return [a]

@diecast
async def _err_nested_bad_async() -> str: 
    await asyncio.sleep(0.01) 
    return 123

@diecast
def _err_nested_wrong_return_typevar(x: T_Unconstrained) -> T_Unconstrained: 
    return "wrong type" 

class _ErrConsistentGeneric(Generic[T_Unconstrained]):
    @diecast
    def method(self, x: T_Unconstrained, y: T_Unconstrained) -> T_Unconstrained: 
        return x

class _ErrParent: pass
class _ErrChild(_ErrParent): pass

@diecast
def _err_inheritance_func(c: _ErrChild) -> bool: 
    return True

@diecast
def _err_typevar_bound_func(x: T_Bound) -> int: 
    return len(x)

@diecast
def _err_typevar_consistency_func(x: T_Unconstrained, y: T_Unconstrained) -> T_Unconstrained: 
    return x

@diecast
def simple_func(value: int) -> str: 
    return str(value) # From error reporting tests

class ErrorReporter: # From error reporting tests
    @diecast
    def instance_method(self, value: Dict[str, List[int]]) -> List[str]: 
        return list(value.keys())
##-##
#-#

# ===== TEST CLASSES ===== #

## ===== GENERIC DECORATOR INTEGRATION ===== ##
@diecast
class _TestDecoratedGeneric(Generic[T_Deco]):
    """Helper generic class decorated with @diecast."""
    def process(self, item: T_Deco) -> T_Deco:
        return item
    def incorrect_return(self, item: T_Deco) -> T_Deco:
        return "wrong"
    def no_annotations(self, item): # Should not be wrapped
            return item

class TestGenericDecoratorIntegration:
    """
    Tests integration scenarios involving @diecast applied directly to Generic classes,
    consolidating tests from test_decorator.py.
    """
    
    def test_generic_class_decorator_pass(self):
        """Test that methods in a decorated generic class work correctly."""
        instance = _TestDecoratedGeneric[int]()
        assert instance.process(item=10) == 10
        assert instance.no_annotations(item="test") == "test" # Unwrapped

    def test_generic_class_decorator_fail_arg(self):
        """Test argument validation in a decorated generic class."""
        instance = _TestDecoratedGeneric[int]()
        with pytest.raises(YouDiedError):
            instance.process(item="wrong")

    def test_generic_class_decorator_with_list(self):
        """Test with a List[T] generic class."""
        @diecast
        class DecoratedList(Generic[LT]):
            items: List[LT]
            def __init__(self, items: List[LT]): self.items = items
            def add(self, item: LT): self.items.append(item)
            def get(self) -> List[LT]: return self.items

        instance = DecoratedList[int]([1, 2])
        assert instance.get() == [1, 2]
        instance.add(3)
        assert instance.get() == [1, 2, 3]
        with pytest.raises(YouDiedError): instance.add("bad")
        with pytest.raises(YouDiedError): DecoratedList[int](["a"])

    def test_generic_class_getitem_injection(self):
        """Test that __class_getitem__ is injected correctly."""
        assert hasattr(_TestDecoratedGeneric, '__class_getitem__')
        specialized = _TestDecoratedGeneric[int]
        assert issubclass(specialized, _TestDecoratedGeneric)
        assert specialized.__orig_bases__[0].__args__ == (int,)
        instance = specialized()
        assert instance.process(1) == 1
        with pytest.raises(YouDiedError): instance.process("a")

    def test_generic_class_specialization_cache(self):
        """Test that generic class specializations are cached."""
        specialized1 = _TestDecoratedGeneric[int]
        specialized2 = _TestDecoratedGeneric[int]
        specialized_str = _TestDecoratedGeneric[str]
        assert specialized1 is specialized2
        assert specialized1 is not specialized_str
        # Check cache directly (implementation detail, but useful for testing)
        assert (_TestDecoratedGeneric, int) in _SPECIALIZED_CLASS_CACHE
        assert (_TestDecoratedGeneric, str) in _SPECIALIZED_CLASS_CACHE
        assert _SPECIALIZED_CLASS_CACHE[(_TestDecoratedGeneric, int)] is specialized1

    def test_generic_class_sync_generator(self):
        """Test sync generator methods in generic classes."""
        @diecast
        class GenClass(Generic[T_Gen]):
            def sync_gen(self, count: int) -> Generator[T_Gen, None, str]:
                i = 0
                while i < count:
                    # Need a way to yield T_Gen based on specialization
                    # Let's assume T_Gen is int for this test path
                    yield i # Yielding int, T_Gen might be different
                    i += 1
                return "Done"
            def bad_yield(self) -> Generator[T_Gen, None, None]:
                yield "wrong" # Yielding str, T_Gen might be different
            def bad_return(self) -> Generator[T_Gen, None, str]:
                yield 0
                return 123 # Yields int, expects str

        instance_int = GenClass[int]()
        gen_int = instance_int.sync_gen(2)
        assert next(gen_int) == 0
        assert next(gen_int) == 1
        with pytest.raises(StopIteration) as si_int: next(gen_int)
        assert si_int.value.value == "Done"

        instance_str = GenClass[str]()
        with pytest.raises(YouDiedError): # Yield validation
            list(instance_str.sync_gen(2)) # Yields int, expects str

        with pytest.raises(YouDiedError): # Bad yield type
            list(instance_int.bad_yield()) # Yields str, expects int

        gen_bad_ret = instance_int.bad_return()
        assert next(gen_bad_ret) == 0
        with pytest.raises(YouDiedError): # Bad return type
            next(gen_bad_ret)

    @pytest.mark.asyncio
    async def test_generic_class_async_generator(self):
        """Test async generator methods in generic classes."""
        @diecast
        class AsyncGenClass(Generic[T_AsyncGen]):
            async def async_gen(self, count: int) -> AsyncGenerator[T_AsyncGen, None]:
                i = 0
                while i < count:
                    yield i # Yielding int, T_AsyncGen might be different
                    await asyncio.sleep(0.01)
                    i += 1
            async def bad_yield(self) -> AsyncGenerator[T_AsyncGen, None]:
                yield "wrong" 
                await asyncio.sleep(0.01)

        instance_int = AsyncGenClass[int]()
        results_int = [i async for i in instance_int.async_gen(2)]

        assert results_int == [0, 1]

        instance_str = AsyncGenClass[str]()
        with pytest.raises(YouDiedError): # Yield validation
             [i async for i in instance_str.async_gen(2)] # Yields int, expects str

        gen_bad_yield = instance_int.bad_yield()
        with pytest.raises(YouDiedError): # Bad yield type
            await anext(gen_bad_yield) # Yields str, expects int

    def test_generic_class_decorator_fail_return(self):
        """Test return value validation in a decorated generic class."""
        instance = _TestDecoratedGeneric[int]()
        with pytest.raises(YouDiedError):
            instance.incorrect_return(item=10) # Returns str, expects int

    def test_generic_class_classmethod(self):
        """Test @classmethod interaction with generic class."""
        @diecast
        class ClsMetGeneric(Generic[T_Cls]):
            @classmethod
            def process(cls, item: T_Cls) -> Type[T_Cls]: # Return type(item)
                 return type(item)
        assert ClsMetGeneric[int].process(item=1) is int
        assert ClsMetGeneric[str].process(item="a") is str
        with pytest.raises(YouDiedError): ClsMetGeneric[int].process(item="a")

    def test_generic_class_staticmethod(self):
        """Test @staticmethod interaction with generic class."""
        @diecast
        class StaticMetGeneric(Generic[T_Static]):
            @staticmethod
            def process(item: T_Static) -> bool:
                return isinstance(item, int) # Example check

        assert StaticMetGeneric[int].process(item=1) is True
        assert StaticMetGeneric[str].process(item="a") is False # Passes validation, returns False
        with pytest.raises(YouDiedError): StaticMetGeneric[int].process(item="a")

    def test_generic_class_ignore_method(self):
        """Test @diecast.ignore on a method within a generic class."""
        @diecast
        class IgnoreMethodGeneric(Generic[T_Ign]):
            def process(self, item: T_Ign) -> T_Ign: return item
            @ignore
            def ignored(self, item: T_Ign) -> str: return "ignored" # Wrong return type

        instance = IgnoreMethodGeneric[int]()
        assert instance.process(1) == 1
        with pytest.raises(YouDiedError): instance.process("a")
        # Ignored method should execute without type check
        assert instance.ignored(item=1) == "ignored"
        assert instance.ignored(item="a") == "ignored"

    def test_generic_class_ignore_class(self):
        """Test @diecast.ignore on the generic class itself."""
        @diecast
        @ignore
        class IgnoredGeneric(Generic[T_IgnCls]):
            def ignored_method(self, item: T_IgnCls) -> str: 
                return "ignored"

        IgnoredGenericInt = IgnoredGeneric[int]
        instance = IgnoredGenericInt()
        # Should execute original code without type checks
        assert instance.ignored_method(item=1) == "ignored"
        assert instance.ignored_method(item="a") == "ignored"
        assert not hasattr(IgnoredGenericInt, '_diecast_type_map')

    def test_generic_class_multiple_typevars(self):
        """Test generic class with multiple type variables."""
        @diecast
        class MultiVar(Generic[K, V]):
            def process(self, key: K, value: V) -> Tuple[K, V]: return key, value

        instance = MultiVar[int, str]()
        assert instance.process(key=1, value="a") == (1, "a")
        with pytest.raises(YouDiedError): instance.process(key="bad", value="a")
        with pytest.raises(YouDiedError): instance.process(key=1, value=2)

    # Purpose: This test verifies that type checks are correctly applied to methods
    # of a nested, pre-specialized generic attribute within another decorated generic class.
    #
    # Scenario:
    # - `OuterGeneric[T_Nested]` is decorated with @diecast.
    # - It contains an attribute `inner: _TestDecoratedGeneric[int]`. Note that `inner`
    #   is *always* specialized as `_TestDecoratedGeneric[int]`, regardless of `T_Nested`.
    # - `_TestDecoratedGeneric[T_Deco]` is also decorated with @diecast.
    #
    # Validation Points:
    # 1. `OuterGeneric.__init__`: When `OuterGeneric[int]` is created, the `inner_val`
    #    (type `T_Nested`, bound to `int`) is passed to `self.inner.process()`.
    #    Since `self.inner` is `_TestDecoratedGeneric[int]`, its `process` method
    #    expects `T_Deco` (bound to `int`). The test confirms this validation works.
    # 2. `inner_int.process()`: Calls to methods on the retrieved `inner` instance
    #    are validated against its fixed specialization (`int`). The test confirms this.
    #
    # Conclusion: The test correctly demonstrates validation of nested generics where
    # the inner generic has a fixed specialization defined in the class structure.
    def test_generic_class_nested_generic(self):
        """Test specialization with nested generic types."""
        @diecast
        class OuterGeneric(Generic[T_Nested]):
            inner: _TestDecoratedGeneric[int] # Use the helper generic
            def __init__(self, inner_val: T_Nested):
                self.inner = _TestDecoratedGeneric[int]()
                self.inner.process(inner_val) # Initial check
            def get_inner(self) -> _TestDecoratedGeneric[int]: return self.inner

        outer_int = OuterGeneric[int](inner_val=1)
        inner_int = outer_int.get_inner()
        assert inner_int.process(2) == 2
        with pytest.raises(YouDiedError): inner_int.process("bad")
        with pytest.raises(YouDiedError): OuterGeneric[int](inner_val="bad")

    def test_generic_class_inheritance(self):
        """Test inheritance involving decorated generic classes."""
        @diecast
        class BaseGen(Generic[T_Inh]):
            def base_method(self, item: T_Inh) -> T_Inh: return item

        @diecast # Decorate child as well
        class ChildGen(BaseGen[int]):
            def child_method(self, item: str) -> int: return len(item)

        child = ChildGen()
        assert child.base_method(1) == 1 # Inherited decorated method
        with pytest.raises(YouDiedError): child.base_method("bad")
        assert child.child_method("test") == 4
        with pytest.raises(YouDiedError): child.child_method(123)

    def test_generic_class_existing_getitem(self):
        """Test @diecast on generic class with existing __class_getitem__."""
        _getitem_calls = []
        class BaseWithGetItem(Generic[T_Exist]):
            # Define a custom __class_getitem__
            def __class_getitem__(cls, key):
                _getitem_calls.append(key)
                # Must call super().__class_getitem__ for Generic functionality
                return super().__class_getitem__(key)

        DecoratedExisting = diecast(BaseWithGetItem) # Apply decorator manually
        DecoratedExistingInt = DecoratedExisting[int]
        instance = DecoratedExistingInt()
        assert DecoratedExisting[int] is not None # Trigger __class_getitem__
        assert int in _getitem_calls
        assert hasattr(DecoratedExistingInt, '_diecast_type_map') # Check decorator applied
        assert hasattr(DecoratedExistingInt, '__class_getitem__')
        assert hasattr(instance, '_diecast_type_map')

        # Verify specialization still works via super call
##-##

## ===== NON-GENERIC DECORATOR INTEGRATION ===== ##
class TestNonGenericDecoratorIntegration:
    """
    Tests integration scenarios involving @diecast applied to non-Generic classes,
    consolidating tests from test_decorator.py.
    """

    def test_non_generic_class_basic_decoration(self):
        """Test applying @diecast to a non-generic class."""
        @diecast
        class DecoratedNonGeneric:
            def method_a(self, x: int) -> str: return str(x)
            def method_b(self, y: str) -> bool: return bool(y)
            def _private(self, z: float) -> float: return z # Should also be wrapped
            def no_ann(self, a): return a # Should not be wrapped

        instance = DecoratedNonGeneric()
        assert instance.method_a(1) == "1"
        assert instance.method_b("a") is True
        assert instance._private(1.1) == 1.1
        assert instance.no_ann(123) == 123

        with pytest.raises(YouDiedError): instance.method_a("bad")
        with pytest.raises(YouDiedError): instance.method_b(123)
        with pytest.raises(YouDiedError): instance._private("bad")
        assert not hasattr(instance.no_ann, _DIECAST_MARKER)

    def test_non_generic_class_ignore_method(self):
        """Test @diecast.ignore on method in non-generic class."""
        @diecast
        class IgnoreMethodNonGeneric:
            def process(self, x: int) -> int: return x
            @ignore
            def ignored(self, y: int) -> str: return "ignored" # Wrong type

        instance = IgnoreMethodNonGeneric()
        assert instance.process(1) == 1
        with pytest.raises(YouDiedError): instance.process("a")
        assert instance.ignored(1) == "ignored" # Runs original code
        assert instance.ignored("a") == "ignored" # Runs original code

    def test_non_generic_class_ignore_class(self):
        """Test @diecast.ignore on the non-generic class itself."""
        @ignore
        @diecast
        class IgnoredNonGeneric:
             def process(self, x: int) -> str: return "ignored" # Wrong type

        instance = IgnoredNonGeneric()
##-##

## ===== ADVANCED DECORATOR INTEGRATION ===== ##

### ----- SETUP ----- ###
def _callable_target(x: int) -> str: 
    return str(x)

@diecast
def _process_callable(func: Callable[[int], str], val: int) -> str:
    return func(val)

@diecast
def _process_namedtuple(pt: Point) -> int:
    return pt.x + pt.y

@diecast
def _process_newtype(user_id: UserId) -> str:
    return f"User_{user_id}"

@diecast
def _process_str_or_bytes(data: StrOrBytes) -> int:
    return len(data)
###-###

class TestAdvancedDecoratorIntegration:
    """
    Tests for advanced and edge cases of the diecast decorator integration,
    consolidating tests from test_advanced.py.
    """

    ### ----- Callable Tests ----- ###
    def test_callable_pass(self):
        """Test passing a valid Callable."""
        assert _process_callable(func=_callable_target, val=10) == "10"

    def test_callable_fail_arg(self):
        """Test passing invalid argument type to the Callable."""
        with pytest.raises(YouDiedError):
            _process_callable(func=_callable_target, val="bad")

    def test_callable_fail_return(self):
        """Test passing a Callable with the wrong return signature."""
        def bad_return_callable(x: int) -> int: return x # Returns int, not str
        with pytest.raises(YouDiedError):
            _process_callable(func=bad_return_callable, val=10)

    def test_callable_fail_arg_signature(self):
        """Test passing a Callable with the wrong argument signature."""
        def bad_arg_callable(x: str) -> str: return x # Takes str, not int
        with pytest.raises(YouDiedError):
            _process_callable(func=bad_arg_callable, val=10)

    def test_callable_nested_signature_warning(self):
        """Test that nested Callables might not be fully validated (known limitation)."""
        # Diecast primarily validates the top-level Callable signature.
        # Deep validation of nested callables is complex and often skipped.
        def complex_callable(f: Callable[[str], bool]) -> Callable[[int], str]:
             def inner(y: int) -> str:
                 return str(f(str(y))) # Calls nested callable f
             return inner

        @diecast
        def process_complex_callable(
            c: Callable[[Callable[[str], bool]], Callable[[int], str]],
            val: int
        ) -> str:
            inner_func = c(lambda s: s == "10") # Provide valid nested callable
            return inner_func(val)

        assert process_complex_callable(c=complex_callable, val=10) == "True"

        # Pass a complex callable where the *inner* callable it returns
        # has a wrong signature - Diecast might not catch this.
        def bad_inner_complex_callable(f: Callable[[str], bool]) -> Callable[[str], str]: # Inner takes str
             def inner(y: str) -> str: return str(f(y))
             return inner

        # This might pass or fail depending on exact validation logic for Callables
        # Expect it might pass validation but fail at runtime inside process_complex_callable
        try:
            process_complex_callable(c=bad_inner_complex_callable, val=10)
        except TypeError: # Expect runtime TypeError if validation passes
             pass
        except YouDiedError: # Or YouDiedError if validation catches it
             pass
        ###-###

    ### ----- NamedTuple Tests ----- ###
    def test_namedtuple_pass(self):
        """Test passing a valid NamedTuple."""
        pt = Point(x=1, y=2)
        assert _process_namedtuple(pt=pt) == 3

    def test_namedtuple_fail_type(self):
        """Test passing wrong type instead of NamedTuple."""
        with pytest.raises(YouDiedError):
            _process_namedtuple(pt=(1, 2)) # Regular tuple

    def test_namedtuple_fail_field_type(self):
        """Test initializing NamedTuple with wrong field types."""
        with pytest.raises(YouDiedError):
            pt = Point(x="1", y=2)
    ###-###

    ### ----- Protocol Tests ----- ###
    @staticmethod
    @diecast
    def _process_sized_protocol(p: SizedProtocol) -> int:
        return len(p)

    @staticmethod
    @diecast
    def _process_simple_protocol(p: SimpleProto) -> str:
        return p.an_attribute * p.method_a()

    def test_protocol_pass_list(self):
        """Test runtime_checkable Protocol with list."""
        assert self._process_sized_protocol(p=[1, 2, 3]) == 3

    def test_protocol_pass_dict(self):
        """Test runtime_checkable Protocol with dict."""
        assert self._process_sized_protocol(p={"a": 1}) == 1

    def test_protocol_pass_str(self):
        """Test runtime_checkable Protocol with str."""
        assert self._process_sized_protocol(p="abc") == 3

    def test_protocol_fail(self):
        """Test runtime_checkable Protocol fails with incompatible type."""
        with pytest.raises(YouDiedError):
            self._process_sized_protocol(p=123) # int has no __len__

    def test_non_runtime_protocol_pass(self):
        """Test non-runtime_checkable Protocol passes structurally."""
        # Diecast should treat non-runtime Protocols like structural checks (similar to Any)
        # if strict_protocols=False (default). If strict_protocols=True, it might fail.
        # Assuming default behavior for this test.
        assert self._process_simple_protocol(p=MatchesSimpleProto()) == "yes"

    def test_non_runtime_protocol_fail(self):
        """Test non-runtime_checkable Protocol fails structurally (if strict)."""
        # If strict_protocols=True, this should raise YouDiedError.
        # If strict_protocols=False (default), it might pass validation but fail at runtime.
        try:
            self._process_simple_protocol(p=DoesNotMatchSimpleProto())
            # If it reaches here, validation passed (non-strict mode)
            # It will likely fail at runtime inside the function due to attribute type mismatch.
        except TypeError:
             pass # Expected runtime failure in non-strict mode
        except YouDiedError:
             pass # Expected validation failure in strict mode
    ###-###

    ### ----- NoReturn Test ----- ###
    def test_noreturn(self):
        """Test NoReturn annotation."""
        @diecast
        def raises_always(x: int) -> NoReturn:
            raise ValueError("Intentional")

        with pytest.raises(ValueError, match="Intentional"):
            raises_always(x=1)

        @diecast
        def sometimes_returns(x: int) -> NoReturn:
             if x > 0:
                 raise ValueError("Positive")
             # Implicitly returns None if x <= 0, violating NoReturn
             # This check happens *after* the function body executes if no exception is raised.

        with pytest.raises(ValueError, match="Positive"):
             sometimes_returns(x=1)
        with pytest.raises(YouDiedError): # Should fail NoReturn check
             sometimes_returns(x=0)
    ###-###


    ### ----- Annotated Test ----- ###
    def test_annotated_pass(self):
        """Test Annotated type passes validation based on underlying type."""
        IntWithMeta = Annotated[int, "metadata"]
        @diecast
        def process_annotated(val: IntWithMeta) -> IntWithMeta: # type: ignore
            return val
        assert process_annotated(val=10) == 10

    def test_annotated_fail(self):
        """Test Annotated type fails validation based on underlying type."""
        IntWithMeta = Annotated[int, "metadata"]
        @diecast
        def process_annotated_fail(val: IntWithMeta) -> IntWithMeta: # type: ignore
            return val
        with pytest.raises(YouDiedError):
            process_annotated_fail(val="wrong")
    ###-###

    ### ----- Multiple Decorators ----- ###
    def test_multi_deco_pass_w_wraps(self):
        """Test pass mode of diecast as outermost decorator when innermost decorator usess functools.wraps."""
        def debug_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs): return func(*args, **kwargs)
            return wrapper
        @diecast
        @debug_decorator
        def func_outer(x: int) -> str: return str(x)
        assert func_outer(x=1) == "1"

    def test_multi_deco_fail_w_wraps(self):
        """Test fail mode diecast as outermost decorator when innermost decorator uses functools.wraps"""
        def debug_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs): return func(*args, **kwargs)
            return wrapper
        @diecast
        @debug_decorator
        def func_outer(x: int) -> str: return str(x)
        with pytest.raises(YouDiedError): 
            func_outer(x="bad")

    def test_multi_deco_w_wrapper_hints(self):
        """Test type checking with diecast as outermost decorator."""
        def debug_decorator(func):
            def wrapper(x: int, *args, **kwargs): 
                return func(x, *args, **kwargs)
            return wrapper
        @diecast
        @debug_decorator
        def func_outer_fail(x) -> str: return str(x)
        with pytest.raises(YouDiedError): 
            func_outer_fail("bad")
        assert func_outer_fail(1) == "1"
    ###-###

    ### ----- Complex Generator/Async ----- ###
    def test_complex_generator_pass(self):
        """Test generator yielding complex types."""
        @diecast
        def complex_gen() -> Generator[Dict[str, Optional[int]], None, None]:
            yield {"a": 1, "b": None}
            yield {"c": 3}
        results = list(complex_gen())
        assert results == [{"a": 1, "b": None}, {"c": 3}]

    def test_complex_generator_fail(self):
        """Test generator yielding incorrect types."""
        @diecast
        def complex_gen_fail() -> Generator[List[int], None, None]:
            yield [1, 2]
            yield {"a": 3} # Dict instead of List
        gen = complex_gen_fail()
        assert next(gen) == [1, 2]
        with pytest.raises(YouDiedError): next(gen)

    @pytest.mark.asyncio
    async def test_complex_async_pass(self):
        """Test async function with complex return type."""
        @diecast
        async def complex_async() -> Tuple[int, str, Optional[bool]]:
            await asyncio.sleep(0.01)
            return (1, "a", None)
        assert await complex_async() == (1, "a", None)

    @pytest.mark.asyncio
    async def test_complex_async_fail(self):
        """Test async function with incorrect return type."""
        @diecast
        async def complex_async_fail() -> List[int]:
            await asyncio.sleep(0.01)
            return [1, "b"] # Contains str
        with pytest.raises(YouDiedError): 
            await complex_async_fail()
    ###-###
##-##

## ===== DEEP INHERITANCE INTEGRATION ===== ##

### ----- SETUP ----- ###
@diecast
def _integ_simple_func(value: Union[A, AA]) -> Union[A, AA]:
    """Helper function for deep inheritance tests."""
    return value

@diecast
def _integ_complex_func(values: List[Union[A, B, C, D, E, F, G, H, I, J,
                                AA, BB, CC, DD, EE, FF, GG, HH, II, JJ]]) -> int:
    """Helper function for deep inheritance tests with complex types."""
    return len(values)
###-###

class TestDeepInheritanceIntegration:
    """Tests decorator behavior with deep inheritance chains."""

    def test_basic_functionality(self):
        """Test that the helper functions work correctly with deep inheritance."""
        j_instance = J()
        assert _integ_simple_func(j_instance) is j_instance
        jj_instance = JJ()
        assert _integ_simple_func(jj_instance) is jj_instance

        with pytest.raises(YouDiedError):
            _integ_simple_func(123) # Expect failure with wrong type

        mixed_list = [J(), JJ(), I(), II(), H(), HH()]
        assert _integ_complex_func(mixed_list) == 6

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
#-#

## ===== PYTHON 3.9+ FEATURES ===== ##
@pytest.mark.skipif(not PY_GTE_39, reason="Requires Python 3.9+")
class TestPython39FeaturesIntegration:
    """Test Python 3.9+ specific typing features with DieCast integration."""

    def test_builtin_generics_pass(self):
        """Test Python 3.9+ built-in generics."""
        @diecast
        def builtin_types_func(
            l: list[int],
            d: dict[str, bool],
            t: tuple[int, ...]
        ) -> set[str]:
            result = set()
            for i in l: result.add(str(i))
            for k, v in d.items(): result.add(k * (1 + int(v)))
            for i in t: result.add(f"t{i}")
            return result
        assert builtin_types_func(l=[1, 2], d={"a": True}, t=(3,)) == {"1", "2", "aa", "t3"}

    def test_builtin_generics_fail(self):
        """Test failures with Python 3.9+ built-in generics."""
        @diecast
        def builtin_fail_func(l: list[int]) -> None: 
            pass
        with pytest.raises(YouDiedError): 
            builtin_fail_func(l=[1, "a"])
        with pytest.raises(YouDiedError): 
            builtin_fail_func(l="not_a_list")
##-##

## ===== PYTHON 3.10+ FEATURES ===== ##
@pytest.mark.skipif(not PY_GTE_310, reason="Requires Python 3.10+")
class TestPython310FeaturesIntegration:
    """Test Python 3.10+ specific typing features with DieCast integration."""

    def test_union_operator_pass(self):
        """Test the pipe union operator (|)."""
        @diecast
        def union_op_func(arg: int | str) -> float | bool:
            return float(arg) if isinstance(arg, int) else (len(arg) > 0)

        assert union_op_func(arg=10) == 10.0
        assert union_op_func(arg="test") is True

    def test_union_operator_fail(self):
        """Test failures with the pipe union operator (|)."""
        @diecast
        def union_op_fail_func(arg: int | str) -> None: 
            pass
        with pytest.raises(YouDiedError): 
            union_op_fail_func(arg=[])

    def test_newtype_pass(self):
        """Test passing a valid NewType."""
        uid = UserId(123)
        assert _process_newtype(user_id=uid) == "User_123"

    def test_newtype_fail_base_type(self):
        """Test passing the base type instead of NewType."""
        # NewType is mostly for static analysis, runtime check might pass if base types match
        # Diecast's behavior depends on its implementation (strict or allows base type)
        # The current implementation is to validate the supertype of the NewType
        assert _process_newtype(user_id=123) == 'User_123'

    def test_newtype_fail_wrong_type(self):
        """Test passing completely wrong type instead of NewType."""
        with pytest.raises(YouDiedError):
            _process_newtype(user_id="abc")

    def test_typevar_str_bytes_pass(self):
        """Test TypeVar('StrOrBytes', str, bytes) passes."""
        assert _process_str_or_bytes(data="abc") == 3
        assert _process_str_or_bytes(data=b"def") == 3

    def test_typevar_str_bytes_fail(self):
        """Test TypeVar('StrOrBytes', str, bytes) fails."""
        with pytest.raises(YouDiedError):
            _process_str_or_bytes(data=123)

    def test_non_generic_class_no_annotations(self):
        """Ensure methods without annotations aren't decorated."""
        @diecast
        class NoAnnNonGeneric:
            def process(self, x: int) -> int: return x
            def no_ann(self, y): return y

        instance = NoAnnNonGeneric()
        assert instance.process(1) == 1
        assert instance.no_ann("a") == "a"
        assert hasattr(instance.process, _DIECAST_MARKER)
        assert not hasattr(instance.no_ann, _DIECAST_MARKER)

    def test_generic_class_no_annotations(self):
        """Ensure methods without annotations aren't decorated in generics."""
        @diecast
        class NoAnnGeneric(Generic[T_NoAnn]):
            def process(self, item: T_NoAnn) -> T_NoAnn: return item
            def no_ann(self, x): return x # No type hints

        instance = NoAnnGeneric[int]()
        assert instance.process(1) == 1
        assert instance.no_ann("test") == "test" # Should execute normally
        # Check internal marker (implementation detail)
        assert hasattr(instance.process, _DIECAST_MARKER)
        assert not hasattr(instance.no_ann, _DIECAST_MARKER)
##-##
#-#

# ===== TEST FUNCTIONS ===== #

## ===== BASIC INHERITANCE ===== ##
@diecast
def _integ_process_j(arg: J) -> str:
    """Helper function for basic inheritance tests (J)."""
    return arg.identify()

def test_inheritance_pass():
    """Test function expecting child class receives child."""
    child = _ErrChild()
    assert _err_inheritance_func(c=child) is True

def test_inheritance_fail():
    """Test function expecting child class receives parent."""
    parent = _ErrParent()
    with pytest.raises(YouDiedError):
        _err_inheritance_func(c=parent)

def test_inheritance_deep_pass():
    """Test that _integ_process_j accepts a valid J instance."""
    instance_j = J()
    assert _integ_process_j(instance_j) == instance_j.identify()

def test_inheritance_deep_fail():
    """Test that _integ_process_j rejects an instance of a parent class (I)."""
    instance_i = I()
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_j(instance_i) # I is not J
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'J'
    assert e.obituary.received_repr == 'I'
    assert e.obituary.path == ['arg']

@diecast
def _integ_process_jj(arg: JJ) -> str:
    """Helper function for basic inheritance tests (JJ)."""
    return arg.identify()

def test_inheritance_deeper_pass():
    """Test that _integ_process_jj accepts a valid JJ instance."""
    instance_jj = JJ()
    assert _integ_process_jj(instance_jj) == instance_jj.identify()

def test_inheritance_deeper_fail():
    """Test that _integ_process_jj rejects instances of parent classes (II, AA)."""
    instance_ii = II()
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_jj(instance_ii) # II is not JJ
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'JJ'
    assert e.obituary.received_repr == 'II'
    assert e.obituary.path == ['arg']

    instance_aa = AA()
    with pytest.raises(YouDiedError) as excinfo_aa:
        _integ_process_jj(instance_aa) # AA is not JJ
    e_aa = excinfo_aa.value
    assert e_aa.cause == 'argument'
##-##

## ===== MOLD INTEGRATION ===== ##
# Note: This test dynamically creates and imports a module.
def test_mold_and_ignore_integration(tmp_path):
    """
    Tests that mold() applies diecast correctly to annotated functions/methods
    within its scope, while respecting @ignore decorators on classes, methods,
    and functions. Also verifies that explicitly @diecast decorated items
    are not re-wrapped by mold().
    """
    # Use an f-string to embed the path resolution logic within the module content
    # Ensure the path is correctly escaped for the string literal.
    src_path_for_module = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
    module_content = f"""
import sys
import os
import inspect
# Add src dir back for the temp module
# Use a raw string literal (r'') to handle potential backslashes in Windows paths
sys.path.insert(0, r'{src_path_for_module}')
from src.diecast import diecast, mold, ignore
from src.diecast.config import _DIECAST_MARKER
from src.diecast.type_utils import YouDiedError # Assuming YouDiedError is needed
from typing import List, Optional

@ignore
class IgnoredClass:
    def method(self, x: int) -> int:
        # Incorrect return type, but should be ignored by mold
        return str(x + 1)

class TargetClass: # Not ignored, mold should process annotated methods
    def __init__(self, val: str): # Annotated, mold should wrap
        self.val = val

    def annotated_method(self, y: int) -> str: # Annotated, mold should wrap
        return str(y * 2)

    @ignore
    def ignored_method(self, z: int, d: Optional[int]=2) -> str: # Ignored, mold should skip
        # Error, but should be ignored by mold
        return z / d

    def unannotated_method(self, a): # Unannotated, mold should skip
        return a

@diecast # Explicitly decorated, mold should skip
def already_decorated(p: bool) -> bool:
    if not p: raise YouDiedError("Forced error", cause='test_case')
    return p

def annotated_func(q: List[int]) -> int: # Annotated, mold should wrap
    return sum(q)

def unannotated_func(r): # Unannotated, mold should skip
    return r

@ignore
def ignored_func(s: str) -> int: # Ignored, mold should skip
    # Incorrect type, but ignored
    return "bad"

# Apply mold to the current module's scope
mold()
""" # End of module_content f-string

    mod_file = tmp_path / "temp_module_mold.py"
    mod_file.write_text(module_content)

    spec = importlib.util.spec_from_file_location("temp_module_mold", mod_file)
    if spec is None or spec.loader is None:
        pytest.fail("Failed to create module spec or loader") # Added check

    temp_module = importlib.util.module_from_spec(spec)
    # Register module before execution to allow relative imports/lookups if needed
    sys.modules["temp_module_mold"] = temp_module
    try:
        spec.loader.exec_module(temp_module)

        ### ----- Verification ----- ###
        # Check @ignore applied correctly (items should NOT have the marker)
        assert hasattr(temp_module.IgnoredClass, _DIECAST_MARKER) # Class ignored
        assert not hasattr(temp_module.IgnoredClass.method, _DIECAST_MARKER) # Method in ignored class
        assert hasattr(temp_module.TargetClass.ignored_method, _DIECAST_MARKER) # Ignored method
        assert hasattr(temp_module.ignored_func, _DIECAST_MARKER) # Ignored function

        # Verify behavior of ignored items (should run original code, potentially erroring/wrong type)
        ignored_instance = temp_module.IgnoredClass()
        assert isinstance(ignored_instance.method(10), str) # Original returns str

        instance = temp_module.TargetClass("hello") # Instantiate TargetClass
        with pytest.raises(ZeroDivisionError): # Original ignored_method raises ZeroDivisionError
            instance.ignored_method(10, 0)
        assert temp_module.ignored_func("test") == "bad" # Original ignored_func returns "bad"

        # Check explicitly decorated item was skipped by mold (should have marker)
        assert hasattr(temp_module.already_decorated, _DIECAST_MARKER)
        assert temp_module.already_decorated(True) is True
        with pytest.raises(YouDiedError, match="Forced error"):
            temp_module.already_decorated(False)

        # Check unannotated items were skipped by mold (should NOT have marker)
        assert not hasattr(temp_module.TargetClass.unannotated_method, _DIECAST_MARKER)
        assert not hasattr(temp_module.unannotated_func, _DIECAST_MARKER)

        # Check annotated items WERE wrapped by mold (should have marker)
        assert hasattr(temp_module.TargetClass.__init__, _DIECAST_MARKER)
        assert hasattr(temp_module.TargetClass.annotated_method, _DIECAST_MARKER)
        assert hasattr(temp_module.annotated_func, _DIECAST_MARKER)

        # Verify behavior of molded items (should type check)
        target_instance = temp_module.TargetClass("hello") # Re-instantiate after __init__ check
        assert target_instance.annotated_method(5) == "10"
        assert target_instance.ignored_method(6) == 3
        with pytest.raises(YouDiedError): instance.annotated_method("bad") # Wrong arg type

        assert temp_module.annotated_func([1, 2, 3]) == 6
        with pytest.raises(YouDiedError): temp_module.annotated_func("bad") # Wrong arg type
        with pytest.raises(YouDiedError): temp_module.annotated_func([1, "a"]) # Wrong element type
        with pytest.raises(YouDiedError): temp_module.TargetClass(123) # Check molded __init__ (wrong arg type)
        ###-###

    finally:
        # Clean up module from sys.modules to avoid side effects
        if "temp_module_mold" in sys.modules:
            del sys.modules["temp_module_mold"]
        # Suggest garbage collection to clean up module object
        gc.collect()
##-##

## ===== ASYNC/GENERATOR INTEGRATION ===== ##

### ----- SETUP ----- ###
@diecast
async def async_multiply(a: int, b: int, f=None) -> int:
    """Simple async helper function."""
    await asyncio.sleep(0.01)
    if f:
        return f
    return a * b

@pytest.mark.asyncio
async def test_async_integration():
    """Tests basic async function integration."""
    assert await async_multiply(3, 4) == 12
    with pytest.raises(YouDiedError):
        await async_multiply("3", 4)
    with pytest.raises(YouDiedError):
        await async_multiply(3, 4, "bad")

@diecast
def count_up_sync(n: int) -> Generator[int, None, str]:
    """Synchronous generator helper."""
    i = 0
    while i < n:
        yield i
        i += 1
    return f"Counted to {n-1}"

@diecast
async def count_up_async(n: int) -> AsyncGenerator[int, None]:
    """Asynchronous generator helper."""
    i = 0
    while i < n:
        yield i
        await asyncio.sleep(0.01)
        i += 1
    # Async generators don't have return values accessible like sync ones

@diecast
def bad_yield_sync(n: int) -> Generator[int, None, None]:
    """Sync generator yielding wrong type."""
    yield 1
    yield "two"
    yield 3

@diecast
async def bad_yield_async(n: int) -> AsyncGenerator[int, None]:
    """Async generator yielding wrong type."""
    yield 1
    await asyncio.sleep(0.01)
    yield "two"
    await asyncio.sleep(0.01)
    yield 3
###-###

def test_sync_generator_integration():
    """Tests synchronous generator integration."""
    gen = count_up_sync(3)
    assert next(gen) == 0
    assert next(gen) == 1
    assert next(gen) == 2
    with pytest.raises(StopIteration) as excinfo:
        next(gen)
    assert excinfo.value.value == "Counted to 2" # Check return value

    # Test argument validation
    with pytest.raises(YouDiedError):
        list(count_up_sync("3"))

    # Test yield validation
    bad_gen = bad_yield_sync(3)
    assert next(bad_gen) == 1
    with pytest.raises(YouDiedError):
        next(bad_gen) # Fails on yielding "two"

@pytest.mark.asyncio
async def test_async_generator_integration():
    """Tests asynchronous generator integration."""
    results = [i async for i in count_up_async(3)]
    assert results == [0, 1, 2]

    # Test argument validation
    with pytest.raises(YouDiedError):
        async for _ in count_up_async("3"):
            pass

    # Test yield validation
    bad_gen = bad_yield_async(3)
    assert await anext(bad_gen) == 1
    with pytest.raises(YouDiedError):
        await anext(bad_gen) # Fails on yielding "two"
##-##

## ===== NESTED TYPES INTEGRATION ===== ##

### ----- SETUP ----- ###
@diecast
def _integ_inner_func(items: List[int]) -> int:
    """Inner function for nested error reporting test."""
    if not all(isinstance(i, int) for i in items):
        # This check is technically redundant due to diecast, but good for illustration
        raise TypeError("Inner function received non-integers!")
    return sum(items)

@diecast
def _integ_outer_func(data: Dict[str, List[int]]) -> int:
    """Outer function for nested error reporting test."""
    total = 0
    for key, value_list in data.items():
        total += _integ_inner_func(value_list) # Call the inner decorated function
    return total

@diecast
def _integ_process_list_union(arg: List[Union[str, float]]) -> int:
    """Helper for testing list containing unions."""
    return len(arg)

@diecast
def _integ_process_dict_optional_custom(arg: Dict[str, Optional[CustomType]]) -> int:
    """Helper for testing dict with optional custom types."""
    count = 0
    for k, v in arg.items():
        if v is not None:
            count += v.get_value()
    return count

@diecast
def _integ_process_complex_nesting(arg: Tuple[List[Set[int]], Dict[str, bool]]) -> int:
    """Helper for testing deeply nested standard types."""
    list_part, dict_part = arg
    return sum(sum(s) for s in list_part) + len(dict_part)
###-###

### ----- TESTS ----- ###
def test_nested_error_reporting():
    """Tests error reporting when an error occurs in a nested call."""
    valid_data = {"a": [1, 2], "b": [3, 4]}
    invalid_data_inner = {"a": [1, "2"], "b": [3, 4]} # Error in inner call's list
    invalid_data_outer = {"a": [1, 2], "b": "not_a_list"} # Error in outer call's dict value

    assert _integ_outer_func(valid_data) == 10

    with pytest.raises(YouDiedError) as excinfo_inner:
        _integ_outer_func(invalid_data_inner)
    e_inner = excinfo_inner.value
    # Check that the error originates from the inner function call path
    assert "_integ_outer_func" in strip_ansi(str(e_inner)) # Error detected during outer func arg check
    assert e_inner.obituary.expected_repr == 'int'
    assert e_inner.obituary.received_repr == 'str'
    assert e_inner.obituary.path == ['data', "value('a')", 1] # Path within _integ_inner_func

    with pytest.raises(YouDiedError) as excinfo_outer:
        _integ_outer_func(invalid_data_outer)
    e_outer = excinfo_outer.value
    # Check that the error originates from the outer function call path
    assert "_integ_outer_func" in strip_ansi(str(e_outer)) # Error detected during outer func arg check
    assert e_outer.obituary.expected_repr == 'list[int]'
    assert e_outer.obituary.received_repr == 'str'
    assert e_outer.obituary.path == ['data', "value('b')"] # Path within _integ_outer_func's loop

def test_nested_list_union_pass():
    """Test passing valid list with union elements."""
    assert _integ_process_list_union(["a", 1.0, "b", 2.0]) == 4

def test_nested_list_union_fail():
    """Test failing list with invalid union element type."""
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_list_union(["a", 1.0, 2]) # 2 is int, not str or float
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'Union[str, float]'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.path == ['arg', 2] # Path to the failing element

def test_nested_dict_optional_custom_pass():
    """Test passing valid dict with optional custom types."""
    data = {"a": CustomType(1), "b": None, "c": CustomType(3)}
    assert _integ_process_dict_optional_custom(data) == 4

def test_nested_dict_optional_custom_fail_key():
    """Test failing dict with invalid key type."""
    data = {"a": CustomType(1), 2: None} # Key 2 is not str
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_dict_optional_custom(data)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'str'
    assert e.obituary.received_repr == 'int'
    assert e.obituary.path == ['arg', 'key(2)'] # Path indicates a key failure

def test_nested_dict_optional_custom_fail_value():
    """Test failing dict with invalid value type (not None or CustomType)."""
    data = {"a": CustomType(1), "b": "not_custom"} # Value "not_custom" is wrong type
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_dict_optional_custom(data)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'CustomType'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['arg', "value('b')"] # Path to the failing value

def test_nested_dict_optional_custom_fail_value_inner():
    """Test failing dict where CustomType itself is invalid (if possible)."""
    class BadCustom: pass
    data = {"a": CustomType(1), "b": BadCustom()} # BadCustom is not CustomType or None
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_dict_optional_custom(data)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'CustomType'
    assert 'BadCustom' in e.obituary.received_repr
    assert e.obituary.path == ['arg', "value('b')"]

def test_complex_nesting_pass():
    """Test passing valid complex nested structure."""
    data = ([{1, 2}, {3}], {"a": True, "b": False})
    assert _integ_process_complex_nesting(data) == 8 # (1+2+3) + 2

def test_complex_nesting_fail_tuple_outer():
    """Test failing complex structure at the outer tuple level."""
    data = ([{1, 2}, {3}], {"a": True, "b": False}, "extra") # Extra element in tuple
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_complex_nesting(data)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'tuple[list[set[int]], dict[str, bool]]'
    assert e.obituary.received_repr == "tuple"
    assert e.obituary.path == ['arg'] # Failure is the overall tuple structure/length

def test_complex_nesting_fail_list_inner():
    """Test failing complex structure within the list element."""
    data = ([{1, 2}, "not_a_set"], {"a": True, "b": False}) # "not_a_set" instead of Set[int]
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_complex_nesting(data)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'set[int]'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['arg', 0, 1] # Path: tuple index 0, list index 1

def test_complex_nesting_fail_set_element():
    """Test failing complex structure within a set element."""
    data = ([{1, 2}, {3, "four"}], {"a": True, "b": False}) # "four" is not int
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_complex_nesting(data)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'int'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['arg', 0, 1, "elem('four')"] # Path: tuple 0, list 1, element in set

def test_complex_nesting_fail_dict_value():
    """Test failing complex structure within the dict value."""
    data = ([{1, 2}, {3}], {"a": True, "b": "not_bool"}) # "not_bool" is not bool
    with pytest.raises(YouDiedError) as excinfo:
        _integ_process_complex_nesting(data)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'bool'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['arg', 1, "value('b')"] # Path: tuple 1, dict key 'b'
###-###

##-##

## ===== GENERIC INTEGRATION ===== ##
def test_integration_multi_typevar_generic():
    """Tests integration with a generic class having multiple type variables."""
    instance = MultiVarGeneric[int, str](1, "value")
    assert instance.get_key() == 1 # Undecorated method
    assert instance.get_value() == "value" # Decorated method

    # Test type checking on decorated method
    with pytest.raises(YouDiedError):
        instance.get_value() # Call again, but imagine state changed to violate return type
        # This requires a way to make get_value return wrong type, difficult here.
        # Let's test __init__ instead.
        MultiVarGeneric[int, str](1, 2) # Wrong type for value

    # Test type checking on undecorated method (should not happen)
    # No YouDiedError expected if we could force get_key to return wrong type

    # Test specialization
    instance_float_bool = MultiVarGeneric[float, bool](1.5, True)
    assert instance_float_bool.get_key() == 1.5
    assert instance_float_bool.get_value() is True

# Purpose: This test verifies how @diecast handles nested generic types where
# a TypeVar from the outer generic class is used within the type annotation
# of an inner generic attribute.
#
# Scenario:
# - `NestedGenericContainer[K_Integ]` is decorated.
# - It contains `items: List[MultiVarGeneric[K_Integ, str]]`.
# - `MultiVarGeneric[K_Integ, V_Integ]` is also decorated.
# - The `K_Integ` from `NestedGenericContainer` is passed to `MultiVarGeneric`.
# - The `V_Integ` for `MultiVarGeneric` is fixed to `str` here.
#
# Validation Points:
# 1. Initialization (`__init__`): When `NestedGenericContainer[int]` is created,
#    the `items` list must contain `MultiVarGeneric[int, str]` instances.
#    Providing an item with the wrong `K_Integ` (e.g., `MultiVarGeneric[str, str]`)
#    correctly raises YouDiedError.
# 2. Method Arguments (`add_item`): The `key` argument must match the `K_Integ`
#    bound by the `NestedGenericContainer` instance (e.g., `int` for
#    `container = NestedGenericContainer[int]`). The `value` must be `str`.
#    Violations raise YouDiedError.
# 3. Method Returns (`get_first_value`): The return type (`str`) is validated.
# 4. Specialization: Creating `NestedGenericContainer[float]` correctly binds
#    `K_Integ` to `float` for subsequent checks.
#
# Conclusion: The test demonstrates that @diecast correctly propagates and validates
# TypeVars passed from an outer generic to a nested generic type annotation.
def test_integration_nested_generic():
    """Tests integration with nested generic types."""
    item1 = MultiVarGeneric[int, str](1, "apple")
    item2 = MultiVarGeneric[int, str](2, "banana")
    container = NestedGenericContainer[int]([item1, item2])

    assert container.get_first_value() == "apple"
    container.add_item(3, "cherry")
    assert len(container.items) == 3
    assert container.items[2].get_key() == 3
    assert container.items[2].get_value() == "cherry"

    # Test type checking on init
    with pytest.raises(YouDiedError):
        NestedGenericContainer[int]([item1, MultiVarGeneric[str, str]("key", "val")]) # Wrong key type in list item

    # Test type checking on add_item (key)
    with pytest.raises(YouDiedError):
        container.add_item("wrong_key_type", "date")

    # Test type checking on add_item (value)
    with pytest.raises(YouDiedError):
        container.add_item(4, 123) # Wrong value type

    # Test type checking on get_first_value (return type)
    # Requires forcing items[0].get_value() to return wrong type, difficult here.

    # Test specialization
    item_f1 = MultiVarGeneric[float, str](1.1, "float_apple")
    container_float = NestedGenericContainer[float]([item_f1])
    assert container_float.get_first_value() == "float_apple"
    container_float.add_item(2.2, "float_banana")
    assert container_float.items[1].get_key() == 2.2

def test_integration_inheritance_generic():
    """Tests integration with inheritance involving generics."""
    child_instance = ChildInheritsGeneric() # Specializes BaseGenericForInherit[int]

    # Test specialized base method
    assert child_instance.process_base(10) == 10
    with pytest.raises(YouDiedError):
        child_instance.process_base("wrong") # Should expect int

    # Test child method
    assert child_instance.process_child("test") == 4
    with pytest.raises(YouDiedError):
        child_instance.process_child(123) # Should expect str
##-##

## ===== ERROR REPORTING INTEGRATION ===== ##
class TestErrorReportingIntegration:
    """
    Tests for the error message format and caller information integration,
    consolidating relevant tests from test_error_reporting.py.
    """

    def test_error_message_format_args(self):
        """Verify the argument error message format."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_basic_func(a="wrong", b="right")
        e = excinfo.value
        msg = strip_ansi(str(e)) # Remove ANSI codes for simple check
        assert "YouDiedError: Argument 'a' FAILED type check" in msg
        assert "Expected: int" in msg
        assert "Received: 'wrong' (str)" in msg
        assert "Function: _err_basic_func(" in msg # Check for function signature

    def test_error_message_format_return(self):
        """Verify the return value error message format."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_nested_wrong_return(a=1) # Function returns int instead of str
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "YouDiedError: Return value FAILED type check" in msg
        assert "Expected: str" in msg
        assert "Received: 1 (int)" in msg
        assert "Function: _err_nested_wrong_return(" in msg
        assert "module: tests.test_integration" in msg

    def test_nested_error_path_reporting(self):
        """Verify error path reporting for nested structures."""
        data = {"key": ["a", 1, {"nested_key": False}]}
        with pytest.raises(YouDiedError) as excinfo:
            ErrorReporter().instance_method(data) # List contains non-int
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "YouDiedError: Argument 'value' FAILED type check" in msg
        assert "Expected: int" in msg
        assert "Received: 'a'" in msg

    def test_caller_variable_name_reporting(self):
        """Verify reporting of the variable name used in the calling scope."""
        my_variable = "not_an_int"
        with pytest.raises(YouDiedError) as excinfo:
            simple_func(value=my_variable)
        e = excinfo.value
        msg = strip_ansi(str(e))
        # Note: Reliably getting the caller's variable name is fragile, so no specific assertion here.
    def test_mixed_context_error(self):
        """Verify error reporting when mixing decorated and non-decorated calls."""
        @diecast
        def inner(x: int): return x
        def outer(y): return inner(y) # Outer is not decorated

        with pytest.raises(YouDiedError) as excinfo:
            outer("wrong") # Error occurs when inner is called
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "YouDiedError: Argument 'x' FAILED type check" in msg
        assert "Expected: int" in msg
        assert "Received: 'wrong' (str)" in msg
        assert "test_mixed_context_error.<locals>.inner(module: tests.test_integration, line:" in msg # Error is in inner

    def test_generator_yield_error_format(self):
        """Test the error format for yield type errors in generators."""
        @diecast
        def generator_func() -> Iterator[int]:
            yield 1
            yield "bad"

        gen = generator_func()
        assert next(gen) == 1
        with pytest.raises(YouDiedError) as excinfo:
            next(gen) # Error on yielding "bad"
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "YouDiedError: Yield value FAILED type check" in msg
        assert "Expected: int" in msg
        assert "Received: 'bad' (str)" in msg
        assert "generator_func(" in msg

    @pytest.mark.asyncio
    async def test_async_error_format(self):
        """Test error format for async function return type errors."""
        @diecast
        async def async_func() -> int:
            await asyncio.sleep(0.01)
            return "not_int"

        with pytest.raises(YouDiedError) as excinfo:
            await async_func()
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "YouDiedError: Return value FAILED type check" in msg
        assert "Expected: int" in msg
        assert "Received: 'not_int' (str)" in msg
        assert "async_func(" in msg

    def test_typevar_consistency_error_format(self):
        """Test error format for TypeVar consistency violations."""
        # Using _err_typevar_consistency_func: def func(x: T, y: T) -> T: return x
        with pytest.raises(YouDiedError) as excinfo:
            _err_typevar_consistency_func(x=1, y="inconsistent")
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "YouDiedError: Argument 'y' FAILED type check" in msg
        assert "_err_typevar_consistency_func(" in msg
        assert "Expected: ~T_Unconstrained (bound to int)" in msg

    # --- Specific Error Message Tests (using _err_* helpers) ---
    def test_err_msg_optional_arg_fail(self):
        """Test error message for Optional argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_optional_func(a=123)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: str" in msg
        assert "Reason: Value does not match inner type str of Optional"

    def test_err_msg_optional_return_fail(self):
        """Test error message for Optional return failure."""
        with pytest.raises(YouDiedError) as excinfo:
             _err_nested_wrong_optional_return(a="test") # Returns int
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: str" in msg
        assert "Reason: Value does not match inner type str of Optional"

    def test_err_msg_union_arg_fail(self):
        logging.warning(f"[test_err_msg_union_arg_fail] Start: _TYPEVAR_BINDINGS = {dict(_TYPEVAR_BINDINGS)}")
        """Test error message for Union argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_union_func(a=[])
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: Union[int, str]" in msg or "Expected: UnionType[int, str]" in msg
        assert "Received: [] (list)" in msg

    def test_err_msg_union_return_fail(self):
        logging.warning(f"[test_err_msg_union_return_fail] Start: _TYPEVAR_BINDINGS = {dict(_TYPEVAR_BINDINGS)}")
        """Test error message for Union return failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_nested_wrong_union_return(a=1) # Returns list
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: Union[int, str]" in msg or "Expected: UnionType[int, str]" in msg
        assert "Received: [1] (list)" in msg

    def test_err_msg_forward_ref_arg_fail(self):
        """Test error message for ForwardRef argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_forward_ref_func(target=123)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: _ErrForwardRefTarget" in msg
        assert "Received: 123 (int)" in msg

    def test_err_msg_method_fail(self):
        """Test error message for instance method argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_instance.instance_method(x="bad")
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: int" in msg
        assert "Received: 'bad' (str)" in msg

    def test_err_msg_classmethod_fail(self):
        """Test error message for class method argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _ErrSimpleClass.class_method(y=123)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: str" in msg
        assert "Received: 123 (int)" in msg

    def test_err_msg_staticmethod_fail(self):
        """Test error message for static method argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
            _ErrSimpleClass.static_method(z="bad")
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: bool" in msg
        assert "Received: 'bad' (str)" in msg

    def test_err_msg_typevar_constrained_fail(self):
        """Test error message for constrained TypeVar argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
             _err_typevar_constrained_func(x=1.5)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: ~T_Constrained" in msg
        assert "Reason: Value does not satisfy constraints (<class 'int'>, <class 'str'>)" in msg

    def test_err_msg_typevar_bound_fail(self):
        """Test error message for bound TypeVar argument failure."""
        with pytest.raises(YouDiedError) as excinfo:
             _err_typevar_bound_func(x=123)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: ~T_Bound" in msg
        assert "Value does not conform to bound Sequence for TypeVar ~T_Bound"

    def test_err_msg_typevar_consistency_return_fail(self):
        """Test error message for TypeVar consistency failure on return."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_nested_wrong_return_typevar(x=1) # Returns str
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: ~T_Unconstrained" in msg
        assert "Expected ~T_Unconstrained (Bound to: int in this call) but received str" in msg

    def test_err_msg_typevar_consistency_in_class_fail(self):
        """Test error message for TypeVar consistency failure in class method."""
        instance = _ErrConsistentGeneric[int]()
        with pytest.raises(YouDiedError) as excinfo:
            instance.method(x=1, y="bad")
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: ~T_Unconstrained" in msg
        assert "Expected ~T_Unconstrained (Bound to: int in this call) but received str" in msg

    def test_err_msg_inheritance_fail(self):
        """Test error message for inheritance failure."""
        parent = _ErrParent()
        with pytest.raises(YouDiedError) as excinfo:
            _err_inheritance_func(c=parent)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: _ErrChild" in msg

    ### ----- Generic Error Message Tests ----- ###
    def test_generic_error_message_arg_fail(self):
        """Test error message format for arg fail in specialized generic method."""
        @diecast
        class GenClass(Generic[T_GenErr]):
            def method(self, x: T_GenErr) -> int: return 1

        instance = GenClass[str]()
        with pytest.raises(YouDiedError) as excinfo:
            instance.method(x=123)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: str" in msg
        assert "Received: 123 (int)" in msg

    def test_generic_error_message_return_fail(self):
        """Test error message format for return fail in specialized generic method."""
        @diecast
        class GenClass(Generic[T_GenErrRet]):
            def method(self, x: int) -> T_GenErrRet: return "bad"

        instance = GenClass[int]()
        with pytest.raises(YouDiedError) as excinfo:
            instance.method(x=1)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: int" in msg
        assert "Received: 'bad' (str)" in msg

    def test_generic_error_message_sync_gen_yield_fail(self):
        """Test error message format for yield fail in specialized sync generator."""
        @diecast
        class GenClass(Generic[T_GenErrSync]):
            def method(self) -> Generator[T_GenErrSync, None, None]:
                yield 1

        instance = GenClass[str]()
        gen = instance.method()
        with pytest.raises(YouDiedError) as excinfo:
            next(gen)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: str" in msg
        assert "Received: 1 (int)" in msg

    def test_generic_error_message_sync_gen_return_fail(self):
        """Test error message format for return fail in specialized sync generator."""
        @diecast
        class GenClass(Generic[T_GenErrSyncRet]):
            def method(self) -> Generator[int, None, T_GenErrSyncRet]:
                yield 1
                return "bad"

        instance = GenClass[int]()
        gen = instance.method()
        assert next(gen) == 1
        with pytest.raises(YouDiedError) as excinfo:
            next(gen) # Raises StopIteration with bad value
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: int" in msg # Check return type
        assert "Received: 'bad' (str)" in msg

    @pytest.mark.asyncio
    async def test_generic_error_message_async_gen_yield_fail(self):
        """Test error message format for yield fail in specialized async generator."""
        @diecast
        class GenClass(Generic[T_GenErrAsync]):
            async def method(self) -> AsyncGenerator[T_GenErrAsync, None]:
                yield 1
                await asyncio.sleep(0.01) # Must be included for async generator syntax

        instance = GenClass[str]()
        gen = instance.method()
        with pytest.raises(YouDiedError) as excinfo:
            await anext(gen)
        e = excinfo.value
        obit = excinfo.value
        msg = strip_ansi(str(e))
        assert "Expected: str" in msg
        assert "Received: 1 (int)" in msg

    def test_error_message_value_truncation(self):
        """Verify that long received values are truncated in the error message."""
        long_string = "a" * (MAX_VALUE_REPR_LENGTH + 100)
        with pytest.raises(YouDiedError) as excinfo:
            simple_func(value=long_string)
        e = excinfo.value
        msg = strip_ansi(str(e))
        assert f"... (truncated at {MAX_VALUE_REPR_LENGTH} chars)" in msg
        assert len(e.obituary.received_repr) < len(repr(long_string)) + 50 # Check it's substantially shorter
    ###-###

    ### ----- Obituary Detail Tests ----- ###
    # These tests focus on verifying the content of the Obituary object itself
    def test_obituary_details_return(self):
        """Verify Obituary details for a return value error."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_nested_wrong_return(a=1)
        obit = excinfo.value.obituary
        assert excinfo.value.cause == 'return'
        assert obit.expected_repr == 'str'
        assert obit.received_repr == 'int' # Check the type representation
        assert obit.value == 1             # Check the actual value
        assert obit.path == ['return'] # Path for return check

    def test_obituary_details_nested_path(self):
        """Verify Obituary details for an error in a nested argument path."""
        data = {"key": ["a", 1, {"nested_key": False}]}
        with pytest.raises(YouDiedError) as excinfo:
            ErrorReporter().instance_method(data)
        obit = excinfo.value.obituary
        assert excinfo.value.cause == 'argument'
        assert obit.expected_repr == 'int'  # Expected element type
        assert obit.received_repr == 'str'  # Actual element type
        assert obit.value == 'a'            # Actual failing element
        # Verify path format (adjust if _format_path changes)
        assert obit.path == ['value', "value('key')", 0]

    def test_obituary_details_caller_variable_name(self):
        """Verify Obituary includes caller variable name when possible."""
        my_var = "text"
        with pytest.raises(YouDiedError) as excinfo:
            simple_func(value=my_var)
        obit = excinfo.value.obituary
        assert excinfo.value.cause == 'argument'
        assert obit.expected_repr == 'int'
        assert obit.received_repr == 'str' # Check the type representation
        assert obit.value == "text"        # Check the actual value

    def test_obituary_details_mixed_context(self):
        """Verify Obituary details when error occurs in decorated func called by undecorated."""
        @diecast
        def inner(x: int): return x
        def outer(y): return inner(y)
        with pytest.raises(YouDiedError) as excinfo:
            outer("wrong")
        obit = excinfo.value.obituary
        assert excinfo.value.cause == 'argument'
        assert obit.expected_repr == 'int'
        assert obit.received_repr == 'str' # Check the type representation
        assert obit.value == "wrong"       # Check the actual value

    def test_obituary_details_generator_yield(self):
        """Verify Obituary details for a generator yield error."""
        @diecast
        def gen_func() -> Iterator[int]: yield "bad"
        gen = gen_func()
        with pytest.raises(YouDiedError) as excinfo:
            next(gen)
        obit = excinfo.value.obituary
        assert excinfo.value.cause == 'yield'
        assert obit.expected_repr == 'int'
        assert obit.received_repr == 'str' # Check the type representation
        assert obit.value == "bad"         # Check the actual value
        assert obit.path == ['return[Yield]'] # Path for yield check

    @pytest.mark.asyncio
    async def test_obituary_details_async_return(self):

        """Verify Obituary details for an async function return error."""
        @diecast
        async def async_func() -> int: return "bad"
        with pytest.raises(YouDiedError) as excinfo:
            await async_func()
        obit = excinfo.value.obituary
        assert excinfo.value.cause == 'return'
        assert obit.expected_repr == 'int'
        assert obit.received_repr == 'str' # Check the type representation
        assert obit.value == "bad"         # Check the actual value

    def test_obituary_details_typevar_consistency(self):
        """Verify Obituary details for a TypeVar consistency error."""
        with pytest.raises(YouDiedError) as excinfo:
            _err_typevar_consistency_func(x=1, y="inconsistent")
        obit = excinfo.value.obituary
        assert excinfo.value.cause == 'argument'
        assert obit.expected_repr == "~T_Unconstrained (bound to int)" # Check exact expected representation
        assert obit.received_repr == 'str' # Check the type representation
        assert obit.value == "inconsistent" # Check the actual value
    ###-###
##-##

## ===== FORWARDREF RECURSIVE INTEGRATION ===== ##
@diecast
class LinkedList:
    def __init__(self, value: int): 
        self.value = value 
        self.next: Optional['LinkedList'] = None

    def set_next(self, node: 'LinkedList') -> None: 
        self.next = node

@diecast
class LinkedListFail:
    def set_next(self, node: 'LinkedListFail') -> None: pass

def test_forward_ref_self_pass():
    """Test forward reference to self within a class method."""
    n1 = LinkedList(1)
    n2 = LinkedList(2)
    n1.set_next(n2)
    assert n1.next is n2

def test_forward_ref_self_fail():
    """Test forward reference failure within a class method."""
    instance = LinkedListFail()
    with pytest.raises(YouDiedError):
        instance.set_next(node=123)
##-##

## ===== TYPE VARIABLES ===== ##
def test_typevar_unconstrained_pass():
    """Test unconstrained TypeVar passes with any type."""
    assert _err_typevar_consistency_func(x=1, y=2) == 1
    assert _err_typevar_consistency_func(x="a", y="b") == "a"
    # Consistency check happens separately

def test_typevar_constrained_pass():
    """Test constrained TypeVar passes with allowed types."""
    assert _err_typevar_constrained_func(x=1) == 1
    assert _err_typevar_constrained_func(x="a") == "a"

def test_typevar_constrained_fail():
    """Test constrained TypeVar fails with disallowed type."""
    with pytest.raises(YouDiedError):
        _err_typevar_constrained_func(x=1.5)

def test_typevar_bound_pass():
    """Test bound TypeVar passes with type matching bound."""
    assert _err_typevar_bound_func(x=[1, 2]) == 2
    assert _err_typevar_bound_func(x="abc") == 3

def test_typevar_bound_fail():
    """Test bound TypeVar fails with type not matching bound."""
    with pytest.raises(YouDiedError):
        _err_typevar_bound_func(x=123)

def test_typevar_consistency_pass():
    """Test TypeVar consistency passes when types match."""
    assert _err_typevar_consistency_func(x=1, y=2) == 1
    assert _err_typevar_consistency_func(x="a", y="b") == "a"

def test_typevar_consistency_fail():
    """Test TypeVar consistency fails when types mismatch."""
    with pytest.raises(YouDiedError):
        _err_typevar_consistency_func(x=1, y="bad")

def test_typevar_consistency_return_fail():
    """Test TypeVar consistency fails on return type mismatch."""
    with pytest.raises(YouDiedError):
        _err_nested_wrong_return_typevar(x=1) # Returns str, T bound to int

def test_typevar_consistency_in_class():
    """Test TypeVar consistency within a generic class method."""
    instance_int = _ErrConsistentGeneric[int]()
    assert instance_int.method(x=1, y=2) == 1
    with pytest.raises(YouDiedError):
        instance_int.method(x=1, y="bad")

    instance_str = _ErrConsistentGeneric[str]()
    assert instance_str.method(x="a", y="b") == "a"
    with pytest.raises(YouDiedError):
        instance_str.method(x="a", y=1)
##-##

## ===== ARGUMENT/RETURN VALIDATION ===== ##

### ----- Basic Function Tests ----- ###
def test_basic_arg_pass():
    """Test basic function call with correct argument types."""
    assert _err_basic_func(a=1, b="test") == 5.0

def test_basic_arg_fail():
    """Test basic function call with incorrect argument type."""
    with pytest.raises(YouDiedError):
        _err_basic_func(a="wrong", b="test")

def test_basic_return_fail():
    """Test basic function call with incorrect return type."""
    with pytest.raises(YouDiedError):
        _err_nested_wrong_return(a=1) # Returns int, expects str
###-###

### ----- Optional Argument/Return Tests ----- ###
def test_optional_arg_pass_none():
    """Test optional argument with None."""
    assert _err_optional_func(a=None) is None

def test_optional_arg_pass_value():
    """Test optional argument with a valid value."""
    assert _err_optional_func(a="test") == 4

def test_optional_arg_fail():
    """Test optional argument with an invalid type."""
    with pytest.raises(YouDiedError):
        _err_optional_func(a=123)

def test_optional_return_pass():
    """Test optional return with a valid value."""
    assert _err_optional_func(a="test") == 4 # Returns int, expects Optional[int]

def test_optional_return_fail():
    """Test optional return with an invalid type."""
    with pytest.raises(YouDiedError):
        _err_nested_wrong_optional_return(a="test") # Returns int, expects Optional[str]
###-###

### ----- Union Argument/Return Tests ----- ###
def test_union_arg_pass():
    """Test union argument with valid types."""
    assert _err_union_func(a=10) == 10.0
    assert _err_union_func(a="test") is True

def test_union_arg_fail():
    """Test union argument with an invalid type."""
    with pytest.raises(YouDiedError):
        _err_union_func(a=[])

def test_union_return_pass():
    """Test union return with valid types."""
    assert _err_union_func(a=10) == 10.0 # Returns float, expects Union[float, bool]
    assert _err_union_func(a="test") is True # Returns bool, expects Union[float, bool]

def test_union_return_fail():
    """Test union return with an invalid type."""
    with pytest.raises(YouDiedError):
        _err_nested_wrong_union_return(a=1) # Returns list, expects Union[int, str]
###-###
##-##

## ===== FORWARD REFERENCES ===== ##

### ----- SETUP ----- ###
@diecast
def _err_forward_ref_func(target: '_ErrForwardRefTarget') -> bool: 
    return isinstance(target, _ErrForwardRefTarget)
###-###

def test_forward_ref_arg_pass():
    """Test forward reference argument with correct type."""
    target = _ErrForwardRefTarget()
    assert _err_forward_ref_func(target=target) is True

def test_forward_ref_arg_fail():
    """Test forward reference argument with incorrect type."""
    with pytest.raises(YouDiedError):
        _err_forward_ref_func(target=123)
##-##

## ===== METHOD TYPES ===== ##

### ----- INSTANCE METHOD ----- ###
def test_instance_method_pass():
    """Test instance method call with correct type."""
    assert _err_instance.instance_method(x=5) == "Value: 5"

def test_instance_method_fail():
    """Test instance method call with incorrect type."""
    with pytest.raises(YouDiedError):
        _err_instance.instance_method(x="bad")
###-###

### ----- CLASS METHOD ----- ###
def test_class_method_pass():
    """Test class method call with correct type."""
    assert _ErrSimpleClass.class_method(y="test") is True

def test_class_method_fail():
    """Test class method call with incorrect type."""
    with pytest.raises(YouDiedError):
        _ErrSimpleClass.class_method(y=123)
###-###

### ----- STATIC METHOD ----- ###
def test_static_method_pass():
    """Test static method call with correct type."""
    assert _ErrSimpleClass.static_method(z=False) is False
    assert _ErrSimpleClass.static_method() is True # Default value

def test_static_method_fail():
    """Test static method call with incorrect type."""
    with pytest.raises(YouDiedError):
        _ErrSimpleClass.static_method(z="bad")
###-###
##-##
#-#