import pytest
import time
import gc
import inspect # Needed for potential mocking/checking
from typing import List, Dict, Union, Any, Set, Type

from diecast import diecast
# Import internals for testing the cache directly
from diecast.type_utils import (
    get_cached_mro_set, 
    is_instance_optimized, 
    _mro_cache,  # Import the cache dict itself
    _mro_cache_lock # Import the lock if needed for setup/teardown
)

# --- Test Setup --- 

# Define some classes for testing
class Base: pass
class Derived(Base): pass
class Mixin:
    def mixin_method(self): pass
class ComplexDerived(Derived, Mixin): pass
class Unrelated: pass

# Fixture to clear the MRO cache before each test function
@pytest.fixture(autouse=True)
def clear_mro_cache_fixture():
    with _mro_cache_lock:
        _mro_cache.clear()
    # Also clear lru_cache on get_cached_mro_set if needed -- REMOVED as function uses manual cache
    # get_cached_mro_set.cache_clear()
    yield # Run the test
    # Teardown (optional, but good practice)
    with _mro_cache_lock:
        _mro_cache.clear()
    # get_cached_mro_set.cache_clear() -- REMOVED as function uses manual cache

# --- Direct Cache & Optimization Tests --- 

def test_get_cached_mro_set_populates_cache():
    """Verify get_cached_mro_set calculates and caches the MRO set."""
    assert Base not in _mro_cache
    mro_set = get_cached_mro_set(Base)
    assert Base in _mro_cache
    assert _mro_cache[Base] == mro_set
    assert mro_set == {Base, object} # Check content

def test_get_cached_mro_set_uses_cache():
    """Verify get_cached_mro_set uses the cache on subsequent calls."""
    # Populate cache
    first_mro_set = get_cached_mro_set(Derived)
    assert Derived in _mro_cache
    assert _mro_cache[Derived] == {Derived, Base, object}
    
    # Call again
    second_mro_set = get_cached_mro_set(Derived)
    assert second_mro_set is first_mro_set # Should return same set object due to lru_cache

def test_get_cached_mro_set_complex_inheritance():
    """Test caching with multiple inheritance."""
    assert ComplexDerived not in _mro_cache
    mro_set = get_cached_mro_set(ComplexDerived)
    assert ComplexDerived in _mro_cache
    # Order doesn't matter in the set, check presence of all bases
    assert mro_set == {ComplexDerived, Derived, Base, Mixin, object}

def test_is_instance_optimized_direct_match():
    """Test the direct type match optimization."""
    d = Derived()
    assert is_instance_optimized(d, Derived) is True

def test_is_instance_optimized_cache_hit_true():
    """Test a successful check using the MRO cache."""
    d = Derived()
    get_cached_mro_set(Derived) # Ensure cache is populated
    assert Derived in _mro_cache 
    assert is_instance_optimized(d, Base) is True
    assert is_instance_optimized(d, object) is True

def test_is_instance_optimized_cache_hit_false():
    """Test a failed check using the MRO cache."""
    d = Derived()
    get_cached_mro_set(Derived) # Ensure cache is populated
    assert Derived in _mro_cache
    assert is_instance_optimized(d, Unrelated) is False
    assert is_instance_optimized(d, Mixin) is False # Derived doesn't inherit Mixin directly

def test_is_instance_optimized_cache_miss():
    """Test that a check populates the cache if missed."""
    d = Derived()
    assert Derived not in _mro_cache
    assert is_instance_optimized(d, Base) is True # Should populate cache
    assert Derived in _mro_cache
    assert _mro_cache[Derived] == {Derived, Base, object}

def test_is_instance_optimized_complex_true():
    """Test complex inheritance success."""
    cd = ComplexDerived()
    assert is_instance_optimized(cd, Derived) is True
    assert is_instance_optimized(cd, Base) is True
    assert is_instance_optimized(cd, Mixin) is True
    assert is_instance_optimized(cd, object) is True
    # Verify cache populated
    assert ComplexDerived in _mro_cache
    assert _mro_cache[ComplexDerived] == {ComplexDerived, Derived, Base, Mixin, object}

def test_is_instance_optimized_fallback():
    """Test fallback to isinstance (mocking required for real test)."""
    # This is harder to test directly without mocking get_cached_mro_set
    # to raise an exception. For now, we rely on the fact that the 
    # implementation includes a try/except fallback.
    d = Derived()
    assert is_instance_optimized(d, Base) is True 

# --- End Direct Cache & Optimization Tests --- 