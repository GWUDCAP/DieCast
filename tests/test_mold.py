import pytest
import sys
import os
import importlib
import tempfile
import textwrap
import shutil # NEW: Import shutil for cleanup
import gc
import random
import string
from typing import List, Dict, Any, Optional, TypeVar, Generic, Sequence, Tuple, Iterator, Union
from abc import ABC, abstractmethod
from diecast import diecast, mold
from diecast.type_utils import YouDiedError, Obituary
from numbers import Number # Needed for bound fixture

# Add src dir to path to allow importing diecast
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import diecast components needed for testing mold side effects
from diecast import diecast # Need this for @diecast.ignore
from diecast.decorator import _DIECAST_MARKER # To check if decorator was applied

# FIX: Change scope to 'function' to match temp_module dependency
@pytest.fixture(scope="function")
def molded_typevar_module(temp_module):
    """Provides a temporary module with molded TypeVar functions/classes."""
    code = """
from diecast import mold, diecast # Need diecast too if used directly
from typing import TypeVar, Generic, Sequence, List, Dict, Tuple, Optional
from numbers import Number

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
    # FIX: Get the _create_module function yielded by temp_module
    _create_mod_func = temp_module
    # FIX: Call the yielded function correctly
    module = _create_mod_func("molded_tv_mod", code)
    yield module # Yield the imported module object
    # Cleanup is handled by temp_module fixture's teardown

# --- Mold Test Cases ---

def test_mold_basic_pass(temp_module):
    mod = temp_module('mod_pass', """
        from diecast import mold
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
    mod = temp_module('mod_arg_fail', """
        from diecast import mold
        from typing import List

        def func_arg_fail(a: int) -> List[int]:
            return [a] * 3
            
        # Explicitly call mold() at module end
        mold()
    """)
    with pytest.raises(YouDiedError):
        mod.func_arg_fail("bad")

def test_mold_basic_return_fail(temp_module):
    mod = temp_module('mod_ret_fail', """
        from diecast import mold
        from typing import List

        def func_ret_fail(a: int) -> List[int]: # Annotated List[int]
            return str(a) # Returns str
            
        # Explicitly call mold() at module end
        mold()
    """)
    with pytest.raises(YouDiedError):
        mod.func_ret_fail(5)

def test_mold_annotated_only(temp_module):
    mod = temp_module('mod_annotated', """
        from diecast import mold
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
    mod = temp_module('mod_ignore', """
        from diecast import mold, diecast
        from typing import List

        @diecast.ignore
        def func_ignored(a: int) -> List[int]: # Annotated but ignored
            return str(a) # Return wrong type
            
        # Explicitly call mold() at module end
        mold()
    """)
    # Should not raise error because ignored
    assert mod.func_ignored(5) == "5"
    # Should have marker set by @diecast.ignore
    assert hasattr(mod.func_ignored, _DIECAST_MARKER)

def test_mold_ignore_class(temp_module):
    mod = temp_module('mod_ignore_cls', """
        from diecast import mold, diecast
        from typing import List
        
        @diecast.ignore
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
    mod_a = temp_module('mod_a_source', """
        from typing import List
        def imported_func(a: int) -> List[int]:
            return str(a) # Returns wrong type
    """)
    
    # Module B: imports from A and uses mold
    mod_b = temp_module('mod_b_consumer', """
        from diecast import mold
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
    mod_base = temp_module('mod_base_cls', """
        from typing import List
        class BaseClass:
            def inherited_method(self, a: int) -> List[int]:
                return str(a) # Wrong type
    """)
    
    # Module Derived: inherits and uses mold
    mod_derived = temp_module('mod_derived_cls', """
        from diecast import mold
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

def test_mold_processes_overridden(temp_module):
    # Module Base: defines base class WITHOUT mold
    mod_base = temp_module('mod_base_ovr', """
        from typing import List
        class BaseClassOvr:
            def overridden_method(self, a: int) -> List[int]:
                return [a]
    """)
    
    # Module Derived: overrides and uses mold
    mod_derived = temp_module('mod_derived_ovr', """
        from diecast import mold
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
    # Calling overridden method SHOULD raise error
    with pytest.raises(YouDiedError):
        instance.overridden_method(10)
    # Check method WAS wrapped by mold
    assert hasattr(instance.overridden_method, _DIECAST_MARKER) 

def test_mold_skips_abstract(temp_module):
    mod = temp_module('mod_abstract', """
        from diecast import mold
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

def test_mold_class_methods(temp_module):
    mod = temp_module('mod_cls_methods', """
        from diecast import mold
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
    with pytest.raises(YouDiedError):
        mod.TheClass.the_classmethod(True)

    # Test staticmethod
    with pytest.raises(YouDiedError):
        mod.TheClass.the_staticmethod(10) # Pass int where Optional[int] expected -> bool returned

    # Check they were all wrapped
    assert hasattr(mod.TheClass.__init__, _DIECAST_MARKER)
    assert hasattr(instance.regular, _DIECAST_MARKER)
    # Class/static methods are descriptors, the wrapper is on the underlying function
    assert hasattr(mod.TheClass.__dict__['the_classmethod'].__func__, _DIECAST_MARKER)
    assert hasattr(mod.TheClass.__dict__['the_staticmethod'].__func__, _DIECAST_MARKER) 

def test_mold_class_methods_fail(temp_module):
    mod = temp_module('mod_cls_methods', """
        from diecast import mold
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
    assert e_cls.obituary.path == [] # Path is empty for return

    # Test staticmethod
    with pytest.raises(YouDiedError) as excinfo_static_arg:
        mod.TheClass.the_staticmethod("bad_type_for_optional_int") # Pass str to Optional[int]
    e_static_arg = excinfo_static_arg.value
    assert e_static_arg.cause == 'argument'
    assert e_static_arg.obituary.expected_repr == 'int'
    assert e_static_arg.obituary.received_repr == 'str'
    assert e_static_arg.obituary.path == ['val']

    # Test staticmethod
    with pytest.raises(YouDiedError) as excinfo_static_ret:
        mod.TheClass.the_staticmethod(100) # Valid arg, but returns int instead of bool
    e_static_ret = excinfo_static_ret.value
    assert e_static_ret.cause == 'return' # FIX: Correct cause check
    assert e_static_ret.obituary.expected_repr == 'bool' # Expected return type
    assert e_static_ret.obituary.received_repr == 'int' # Actual returned type (100)
    assert e_static_ret.obituary.path == [] # Path is empty for return

    # Check they were all wrapped
    assert hasattr(mod.TheClass.__init__, _DIECAST_MARKER)
    assert hasattr(instance.regular, _DIECAST_MARKER)
    # Class/static methods are descriptors, the wrapper is on the underlying function
    assert hasattr(mod.TheClass.__dict__['the_classmethod'].__func__, _DIECAST_MARKER)
    assert hasattr(mod.TheClass.__dict__['the_staticmethod'].__func__, _DIECAST_MARKER) 

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
    assert e_inst_ret.obituary.path == []

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
    assert e_cls_ret.obituary.path == []

def test_mold_overridden_method_fail(temp_module):
    # Module Base: defines base class WITHOUT mold
    mod_base = temp_module('mod_base_ovr', """
        from typing import List
        class BaseClassOvr:
            def overridden_method(self, a: int) -> List[int]:
                return [a]
    """)
    
    # Module Derived: overrides and uses mold
    mod_derived = temp_module('mod_derived_ovr', """
        from diecast import mold
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
    # Calling overridden method SHOULD raise error
    with pytest.raises(YouDiedError):
        instance.overridden_method(10)
    # Check method WAS wrapped by mold
    assert hasattr(instance.overridden_method, _DIECAST_MARKER) 

    child_instance = mod_derived.DerivedClassOvr()
    with pytest.raises(YouDiedError) as excinfo:
        child_instance.overridden_method("wrong") # Should fail on child's molded method
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == 'int'
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['a']

def test_mold_with_typevars_fail_consistency(molded_typevar_module):
    """Test mold TypeVar consistency failure."""
    mod = molded_typevar_module
    with pytest.raises(YouDiedError) as excinfo:
        mod.consistent_func(100, "200") # Fail consistency (int vs str)
    e = excinfo.value
    assert e.cause == 'argument'
    assert e.obituary.expected_repr == '~T' # Expect the TypeVar itself
    assert e.obituary.received_repr == 'str'
    assert e.obituary.path == ['y']
    assert e.obituary.message == "TypeVar consistency violation: Expected ~T (Bound to: int) but received str"

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
    assert e.obituary.message == 'Value is not an instance of expected type'

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
    assert e.obituary.message == 'Value not in allowed types for constrained TypeVar' 