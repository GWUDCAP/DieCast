# ===== IMPORTS ===== #

## ===== STANDARD LIBRARY ===== ##
from typing import TypeVar, Generic, Sequence, List, Dict, Tuple, Optional # Used in various tests/fixtures
from abc import ABC, abstractmethod # Used in test_mold_skips_abstract
from numbers import Number # Used in molded_typevar_module fixture
import sys
import os

## ===== THIRD PARTY ===== ##
import pytest

## ===== LOCAL IMPORTS ===== ##
# HACK: Add src dir to path to allow importing diecast. Ideally, manage path via project setup/pytest config.
# Adjusted path relative to this file (diecast/tests/test_mold.py) to reach 'src' parent.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.diecast.decorator import _DIECAST_MARKER, diecast, ignore # Combined imports
from src.diecast.type_utils import YouDiedError # Corrected import path
from src.diecast.mold import mold


# ===== FIXTURES ===== #

# FIX: Change scope to 'function' to match temp_module dependency (Comment retained from original)
@pytest.fixture(scope="function")
def molded_typevar_module(temp_module):
    """Provides a temporary module with molded TypeVar functions/classes."""
    # NOTE: Imports within the temp module code string assume 'diecast' is importable
    # in the execution context of the temporary module.
    code = """
from src.diecast import mold, diecast, ignore # Need diecast/ignore too if used directly
from typing import TypeVar, Generic, Sequence, List, Dict, Tuple, Optional
from numbers import Number
from abc import ABC, abstractmethod # Added for completeness if needed within temp module

T = TypeVar('T')
N = TypeVar('N', bound=Number) # Use Number for bound example
C = TypeVar('C', int, str)

def consistent_func(x: T, y: T) -> T:
    return x # Mold applies

def bound_func(seq: N) -> N:
    return seq # Mold applies

def constrained_func(val: C) -> C:
    return val # Mold applies

class Processor(Generic[T]):
    def process(self, item: T) -> T:
        return item # Mold applies

# Example of direct decoration for comparison if needed
@diecast
def directly_decorated(arg: int) -> int:
    return arg

mold() # Apply mold to all applicable items
"""
    _create_mod_func = temp_module
    module_tuple = _create_mod_func("molded_tv_mod", code)
    yield module_tuple[0] # Yield only the module object
    # Cleanup is handled by temp_module fixture's teardown


# ===== TEST FUNCTIONS ===== #

def test_mold_basic_pass(temp_module):
    mod, _ = temp_module('mod_pass', """
        from src.diecast import mold
        from typing import List

        def func_molded(a: int) -> List[int]:
            return [a] * 3

        # Explicitly call mold() at module end
        mold()
    """)
    assert mod.func_molded(5) == [5, 5, 5]
    # Check if it was actually wrapped by diecast via mold
    assert hasattr(mod.func_molded, _DIECAST_MARKER)


def test_mold_basic_arg_fail(temp_module):
    mod, _ = temp_module('mod_arg_fail', """
        from src.diecast import mold
        from typing import List

        def func_arg_fail(a: int) -> List[int]:
            return [a] * 3

        # Explicitly call mold() at module end
        mold()
    """)
    with pytest.raises(YouDiedError):
        mod.func_arg_fail("bad")


def test_mold_basic_return_fail(temp_module):
    mod, _ = temp_module('mod_ret_fail', """
        from src.diecast.mold import mold # Use src path for consistency
        from typing import List

        def func_ret_fail(a: int) -> List[int]: # Annotated List[int]
            return str(a) # Returns str

        # Explicitly call mold() at module end
        mold()
    """)
    with pytest.raises(YouDiedError):
        mod.func_ret_fail(5)


def test_mold_annotated_only(temp_module):
    mod, _ = temp_module('mod_annotated', """
        from src.diecast import mold
        from typing import List

        def func_annotated(a: int) -> List[int]:
            return [a]

        def func_unannotated(a, b):
            # Should not be wrapped, no error even with bad types
            return a + b

        # Explicitly call mold() at module end
        mold()
    """)
    # Unannotated should not raise error
    assert mod.func_unannotated("a", "b") == "ab"
    assert not hasattr(mod.func_unannotated, _DIECAST_MARKER)

    # Annotated should raise error
    with pytest.raises(YouDiedError):
        mod.func_annotated("bad")
    assert hasattr(mod.func_annotated, _DIECAST_MARKER)


def test_mold_ignore_decorator(temp_module):
    mod, _ = temp_module('mod_ignore', """
        from src.diecast import mold, diecast, ignore # Ensure ignore is imported
        from typing import List

        @ignore # Use ignore directly
        def func_ignored(a: int) -> List[int]: # Annotated but ignored
            return str(a) # Return wrong type

        # Explicitly call mold() at module end
        mold()
    """)
    # Should not raise error because ignored
    assert mod.func_ignored(5) == "5"
    # Should have marker set by @ignore
    assert hasattr(mod.func_ignored, _DIECAST_MARKER)


def test_mold_ignore_class(temp_module):
    mod, _ = temp_module('mod_ignore_cls', """
        from src.diecast import mold, diecast, ignore # Ensure ignore is imported
        from typing import List

        @ignore # Use ignore directly
        class MyIgnoredClass:
            def method_annotated(self, x: int) -> str:
                 return x # Return wrong type

        # Explicitly call mold() at module end
        mold()
    """)
    instance = mod.MyIgnoredClass()
    # Should not raise error because class is ignored
    assert instance.method_annotated(123) == 123
    # Method itself might not have marker, but class does
    assert hasattr(mod.MyIgnoredClass, _DIECAST_MARKER)
    # Check method wasn't wrapped independently
    assert not hasattr(instance.method_annotated, _DIECAST_MARKER)


def test_mold_skips_imported(temp_module):
    # Module A: defines function WITHOUT mold
    mod_a, _ = temp_module('mod_a_source', """
        from typing import List
        def imported_func(a: int) -> List[int]:
            return str(a) # Returns wrong type
    """)

    # Module B: imports from A and uses mold
    mod_b, _ = temp_module('mod_b_consumer', f"""
        import sys
        # Ensure mod_a_source is importable from mod_b_consumer's perspective
        sys.path.append(r'{os.path.dirname(mod_a.__file__)}') # Add mod_a's dir to path

        from src.diecast import mold
        from mod_a_source import imported_func # Import the function

        def own_func(x: int): # Mold should wrap this
            return x

        # Explicitly call mold() at module end
        mold()
    """)

    # Calling imported func via mod_b should NOT raise error
    assert mod_b.imported_func(5) == "5"
    # Check it wasn't wrapped by mold in mod_b
    assert not hasattr(mod_b.imported_func, _DIECAST_MARKER)
    # Check own_func WAS wrapped
    assert hasattr(mod_b.own_func, _DIECAST_MARKER)


def test_mold_skips_inherited(temp_module):
    # Module Base: defines base class WITHOUT mold
    mod_base, _ = temp_module('mod_base_cls', """
        from typing import List
        class BaseClass:
            def inherited_method(self, a: int) -> List[int]:
                return str(a) # Wrong type
    """)

    # Module Derived: inherits and uses mold
    mod_derived, _ = temp_module('mod_derived_cls', f"""
        import sys
        # Ensure mod_base_cls is importable
        sys.path.append(r'{os.path.dirname(mod_base.__file__)}')

        from src.diecast import mold
        from mod_base_cls import BaseClass

        class DerivedClass(BaseClass):
            pass # Inherits method

        # Explicitly call mold() at module end
        mold()
    """)

    instance = mod_derived.DerivedClass()
    # Calling inherited method should NOT raise error
    assert instance.inherited_method(10) == "10"
    # Check method wasn't wrapped by mold
    assert not hasattr(instance.inherited_method, _DIECAST_MARKER)


def test_mold_skips_abstract(temp_module):
    mod, _ = temp_module('mod_abstract', """
        from src.diecast import mold
        from abc import ABC, abstractmethod
        from typing import List

        class AbstractStuff(ABC):
            @abstractmethod
            def stuff(self, x: int) -> List[int]: # Annotated abstract method
                pass

        # Explicitly call mold() at module end
        mold()
    """)
    # Check the abstract method itself was NOT wrapped by mold
    # Accessing via the class dict
    assert not hasattr(mod.AbstractStuff.__dict__['stuff'], _DIECAST_MARKER)


def test_mold_class_methods_fail(temp_module):
    mod, _ = temp_module('mod_cls_methods', """
        from src.diecast import mold
        from typing import Optional

        class TheClass:
            # __init__ should be wrapped if annotated
            def __init__(self, name: str):
                self.name = name

            # Regular method
            def regular(self, num: int) -> str:
                return num # Wrong type

            @classmethod
            def the_classmethod(cls, flag: bool) -> int:
                return str(flag) # Wrong type

            @staticmethod
            def the_staticmethod(val: Optional[int]) -> bool:
                return val # Wrong type (sometimes)

        # Explicitly call mold() at module end
        mold()
    """)

    # Test __init__
    with pytest.raises(YouDiedError):
        mod.TheClass(123) # Pass int instead of str

    instance = mod.TheClass("TestName") # Valid init

    # Test regular method
    with pytest.raises(YouDiedError):
        instance.regular(5)

    # Test classmethod
    with pytest.raises(YouDiedError) as excinfo_cls:
        mod.TheClass.the_classmethod(True)
    e_cls = excinfo_cls.value
    assert e_cls.cause == 'return' # Cause is return
    assert e_cls.obituary.expected_repr == 'int' # Expected return type
    assert e_cls.obituary.received_repr == 'str' # Actual returned type
    assert e_cls.obituary.path == ['return'] # Path is empty for return

    # Test staticmethod
    with pytest.raises(YouDiedError) as excinfo_static_arg:
        mod.TheClass.the_staticmethod("bad_type_for_optional_int") # Pass str to Optional[int]
    e_static_arg = excinfo_static_arg.value
    assert e_static_arg.cause == 'argument'
    assert e_static_arg.obituary.expected_repr == 'int' # Corrected expectation
    assert e_static_arg.obituary.received_repr == 'str'
    assert e_static_arg.obituary.path == ['val']

    # Test staticmethod return fail
    with pytest.raises(YouDiedError) as excinfo_static_ret:
        mod.TheClass.the_staticmethod(100) # Valid arg, but returns int instead of bool
    e_static_ret = excinfo_static_ret.value
    assert e_static_ret.cause == 'return' # FIX: Correct cause check (was already correct)
    assert e_static_ret.obituary.expected_repr == 'bool' # Expected return type
    assert e_static_ret.obituary.received_repr == 'int' # Actual returned type (100)
    assert e_static_ret.obituary.path == ['return'] # Path is empty for return

    # Check they were all wrapped
    assert hasattr(mod.TheClass.__init__, _DIECAST_MARKER)
    assert hasattr(instance.regular, _DIECAST_MARKER)
    # Class/static methods are descriptors, the wrapper is on the underlying function
    # Check the marker on the descriptor object itself, not the underlying __func__
    assert hasattr(mod.TheClass.__dict__['the_classmethod'], _DIECAST_MARKER)
    assert hasattr(mod.TheClass.__dict__['the_staticmethod'], _DIECAST_MARKER)

    # Test instance method ARGUMENT fail
    with pytest.raises(YouDiedError) as excinfo_inst_arg:
        instance.regular("bad") # Pass str where int expected
    e_inst_arg = excinfo_inst_arg.value
    assert e_inst_arg.cause == 'argument'
    assert e_inst_arg.obituary.expected_repr == 'int'
    assert e_inst_arg.obituary.received_repr == 'str'
    assert e_inst_arg.obituary.path == ['num']

    # Test instance method RETURN fail
    with pytest.raises(YouDiedError) as excinfo_inst_ret:
        instance.regular(5) # Valid arg, but returns int instead of str
    e_inst_ret = excinfo_inst_ret.value
    assert e_inst_ret.cause == 'return'
    assert e_inst_ret.obituary.expected_repr == 'str'
    assert e_inst_ret.obituary.received_repr == 'int'
    assert e_inst_ret.obituary.path == ['return']

    # Test class method ARGUMENT fail
    with pytest.raises(YouDiedError) as excinfo_cls_arg:
        mod.TheClass.the_classmethod(123) # Pass int where bool expected
    e_cls_arg = excinfo_cls_arg.value
    assert e_cls_arg.cause == 'argument'
    assert e_cls_arg.obituary.expected_repr == 'bool'
    assert e_cls_arg.obituary.received_repr == 'int'
    assert e_cls_arg.obituary.path == ['flag']

    # Test class method RETURN fail
    with pytest.raises(YouDiedError) as excinfo_cls_ret:
        mod.TheClass.the_classmethod(True) # Valid arg, but returns str instead of int
    e_cls_ret = excinfo_cls_ret.value
    assert e_cls_ret.cause == 'return'
    assert e_cls_ret.obituary.expected_repr == 'int'
    assert e_cls_ret.obituary.received_repr == 'str'
    assert e_cls_ret.obituary.path == ['return']


def test_mold_overridden_method_fail(temp_module):
    # Module Base: defines base class WITHOUT mold
    mod_base, _ = temp_module('mod_base_ovr', """
        from typing import List
        class BaseClassOvr:
            def overridden_method(self, a: int) -> List[int]:
                return [a]
    """)

    # Module Derived: overrides and uses mold
    mod_derived, _ = temp_module('mod_derived_ovr', f"""
        import sys
        # Ensure mod_base_ovr is importable
        sys.path.append(r'{os.path.dirname(mod_base.__file__)}')

        from src.diecast import mold
        from mod_base_ovr import BaseClassOvr
        from typing import List

        class DerivedClassOvr(BaseClassOvr):
            # Override with annotation, should be wrapped by mold
            def overridden_method(self, a: int) -> List[int]:
                return str(a) # Return wrong type here

        # Explicitly call mold() at module end
        mold()
    """)

    instance = mod_derived.DerivedClassOvr()
    # Calling overridden method SHOULD raise error (return fail)
    with pytest.raises(YouDiedError) as excinfo_ret:
        instance.overridden_method(10)
    e_ret = excinfo_ret.value
    assert e_ret.cause == 'return'
    assert e_ret.obituary.expected_repr == 'list[int]'
    assert e_ret.obituary.received_repr == 'str'

    # Check method WAS wrapped by mold
    assert hasattr(instance.overridden_method, _DIECAST_MARKER)

    # Test argument failure on the overridden method
    child_instance = mod_derived.DerivedClassOvr()
    with pytest.raises(YouDiedError) as excinfo_arg:
        child_instance.overridden_method("wrong") # Should fail on child's molded method
    e_arg = excinfo_arg.value
    assert e_arg.cause == 'argument'
    assert e_arg.obituary.expected_repr == 'int'
    assert e_arg.obituary.received_repr == 'str'
    assert e_arg.obituary.path == ['a']


def test_mold_with_typevars_fail_consistency(molded_typevar_module):
    """Test mold TypeVar consistency failure."""
    mod = molded_typevar_module
    with pytest.raises(YouDiedError) as excinfo:
        mod.consistent_func(100, "200") # Fail consistency (int vs str)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == '~T (bound to int)' # Expect the TypeVar itself
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['y']


def test_mold_with_typevars_fail_bound(molded_typevar_module):
    """Test mold TypeVar bound failure."""
    mod = molded_typevar_module
    with pytest.raises(YouDiedError) as excinfo:
        mod.bound_func("string") # Fail bound (str not Number)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == '~N'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['seq']


def test_mold_with_typevars_fail_constrained(molded_typevar_module):
    """Test mold TypeVar constraint failure."""
    mod = molded_typevar_module
    with pytest.raises(YouDiedError) as excinfo:
        mod.constrained_func(1.5) # Fail constraint (float not int/str)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == '~C' # Expect the TypeVar itself
    assert e.obituary.received_repr == 'float'
    assert e.obituary.path == ['val']
    assert e.obituary.message == "Value does not satisfy constraints (<class 'int'>, <class 'str'>) for TypeVar ~C"


def test_mold_generic_vs_non_generic_class(temp_module):
    """(C1, C2) Test mold() applying to generic and non-generic classes."""
    # NOTE: Adjusted import path within temp module code
    mod, _ = temp_module('mod_gen_vs_non', """
        from src.diecast import mold, diecast, ignore
        # Use relative import within the temp module if decorator is part of diecast package
        from src.diecast.decorator import _DIECAST_MARKER
        from typing import TypeVar, Generic, List, Optional
        from abc import ABC, abstractmethod # Added for completeness

        T = TypeVar('T')

        # ---- Generic Class ----
        class MyGeneric(Generic[T]):
            # Mold calls diecast() on the class if Generic, injecting __class_getitem__.
            # Methods are handled during specialization.
            def __init__(self, val: T):
                self.val = val

            def process(self, item: T) -> T:
                return item

            def unannotated(self, x):
                return x

            @ignore # Test ignore within generic class handled by mold
            def ignored_method_generic(self, item: T) -> T:
                 return item

        # ---- Non-Generic Class ----
        class MyNonGeneric:
            # Mold should decorate annotated methods here directly
            def __init__(self, val: int):
                self.val = val

            def process(self, item: str) -> str:
                return item * self.val

            def unannotated(self, x):
                return x

            @ignore # Test ignore within non-generic class handled by mold
            def ignored_method_non_generic(self, item: str) -> str:
                 return item

        # ---- Ignored Generic Class ----
        @ignore
        class IgnoredGeneric(Generic[T]):
            def method(self, item: T) -> T:
                return item

        # Apply mold
        mold()
    """)

    # --- Verification ---

    # (C1) Generic Class: Check __class_getitem__ was injected
    assert hasattr(mod.MyGeneric, '__class_getitem__'), "Mold should trigger diecast() which injects __class_getitem__ for Generic"

    # Methods within generic SHOULD NOT have the marker directly after mold()
    assert not hasattr(mod.MyGeneric.process, _DIECAST_MARKER), "Generic method shouldn't have marker after mold()"
    assert not hasattr(mod.MyGeneric.__init__, _DIECAST_MARKER), "Generic __init__ shouldn't have marker after mold()"
    assert not hasattr(mod.MyGeneric.unannotated, _DIECAST_MARKER)
    # The ignore marker IS set directly by @ignore
    assert hasattr(mod.MyGeneric.ignored_method_generic, _DIECAST_MARKER), "Ignore marker should be set"
    assert getattr(mod.MyGeneric, '_DIECAST_GENERIC', False), "Generic class should have _DIECAST_GENERIC marker"

    # Verify generic specialization works (confirming decoration pathway was set up)
    SpecializedInt = mod.MyGeneric[int]
    instance_int = SpecializedInt(10)
    assert instance_int.process(5) == 5
    with pytest.raises(YouDiedError): # Check type enforcement works
        instance_int.process("bad")
    # Ignored method within specialized generic should still be ignored
    assert instance_int.ignored_method_generic(20) == 20

    # (C2) Non-Generic Class: Check methods ARE decorated by mold()
    instance_non = mod.MyNonGeneric(3)
    assert hasattr(instance_non.process, _DIECAST_MARKER), "Non-generic method should have marker after mold()"
    assert hasattr(mod.MyNonGeneric.__init__, _DIECAST_MARKER), "Non-generic __init__ should have marker after mold()"
    assert not hasattr(instance_non.unannotated, _DIECAST_MARKER)
    # Ignore marker IS set
    assert hasattr(instance_non.ignored_method_non_generic, _DIECAST_MARKER), "Ignore marker should be set"

    # Verify non-generic behavior
    assert instance_non.process("X") == "XXX"
    with pytest.raises(YouDiedError):
        instance_non.process(123)
    # Ignored method should not raise
    assert instance_non.ignored_method_non_generic("test") == "test"

    # (C4) Ignored Generic Class (Original comment referred to C4, retaining for context)
    assert hasattr(mod.IgnoredGeneric, _DIECAST_MARKER), "Ignored generic class should have marker"
    assert not getattr(mod.IgnoredGeneric, '_DIECAST_GENERIC', False), "Ignored generic class should not have _DIECAST_GENERIC marker"
    # Instantiation should work without specialization
    ignored_inst = mod.IgnoredGeneric()
    assert ignored_inst.method(1) == 1 # No type checks