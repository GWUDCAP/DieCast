import pytest
import logging
import diecast.logging
from diecast.type_utils import _check_type_cache_obituary, _mro_cache, _TYPEVAR_BINDINGS
import sys
import os
import importlib
import tempfile
import textwrap
import shutil
import gc

@pytest.fixture(scope="function", autouse=True)
def configure_diecast_logging():
    """Ensures all logs are visible during test runs by setting the diecast logger level to debug."""
    # Get the diecast logger directly and set its level to DEBUG
    diecast.logging.set_verbosity(logging.DEBUG)
    
    # The rest is handled by pytest's log capture based on our pyproject.toml config
    yield 

# ADD FIXTURE to clear caches before each test
@pytest.fixture(scope="function", autouse=True)
def clear_caches():
    """Clears internal caches before each test function runs."""
    _check_type_cache_obituary.clear()
    _mro_cache.clear() # Also clear the MRO cache
    _TYPEVAR_BINDINGS.clear() # Also clear TypeVar bindings
    yield # Test runs here

# --- Helper: Strip ANSI Codes (String Ops) ---
def strip_ansi(text: str) -> str:
    """Removes ANSI escape sequences using string operations."""
    output = []
    i = 0
    while i < len(text):
        if text[i] == '\x1b': # Start of escape sequence
            i += 1
            if i < len(text) and text[i] == '[':
                i += 1
                # Consume digits, semicolons until terminating character (e.g., 'm')
                while i < len(text) and text[i].isdigit() or text[i] == ';':
                    i += 1
                if i < len(text): # Consume the terminating character
                    i += 1
            # Skip incomplete or malformed sequences
        else:
            output.append(text[i])
            i += 1
    return "".join(output)
# --- End Helper ---

# --- Fixture: Temporary Module ---
# Moved from test_mold.py to make it available globally
@pytest.fixture(scope="function") # Use function scope for isolation
def temp_module():
    temp_dir = tempfile.mkdtemp()
    sys.path.insert(0, temp_dir)
    
    module_files = {}
    imported_modules = []

    def _create_module(module_name: str, code: str):
        code = textwrap.dedent(code)
        module_path = os.path.join(temp_dir, f'{module_name}.py')
        with open(module_path, 'w') as f:
            f.write(code)
        
        # Ensure parent packages exist if needed (e.g., 'pkg.sub')
        parts = module_name.split('.')
        for i in range(1, len(parts)):
            pkg_path = os.path.join(temp_dir, *parts[:i], '__init__.py')
            os.makedirs(os.path.dirname(pkg_path), exist_ok=True)
            if not os.path.exists(pkg_path):
                with open(pkg_path, 'w') as f:
                    f.write('# Package marker')

        module_files[module_name] = module_path
        
        # Invalidate caches before attempting import
        importlib.invalidate_caches()

        # Add try/except with debug info for import robustness
        try:
            module = importlib.import_module(module_name)
            imported_modules.append(module_name)
            
            # If the module imports and calls mold, it will be automatically processed
            # No need to call an internal _process_module function
                
            return module, module_path # Return path for potential cleanup
        except Exception as e:
            print(f"\n!!! DieCast Test Debug !!!")
            print(f"Error importing temporary module '{module_name}' from '{temp_dir}'")
            print(f"Error: {e}")
            print(f"sys.path[0]: {sys.path[0] if sys.path else 'empty'}")
            try:
                print(f"Contents of {temp_dir}: {os.listdir(temp_dir)}")
            except Exception as list_e:
                print(f"Could not list contents of {temp_dir}: {list_e}")
            print(f"!!! End DieCast Test Debug !!!\n")
            raise

    yield _create_module # Yield the creator function

    # Cleanup
    # First ensure all created modules are removed from sys.modules
    for module_name in imported_modules:
        if module_name in sys.modules:
            # Use try-except for module deletion as it can sometimes fail
            try:
                del sys.modules[module_name]
            except KeyError:
                pass

    # Remove temp directory from sys.path
    try:
        # Ensure the path is actually the one we added
        if sys.path[0] == temp_dir:
             sys.path.pop(0)
        else:
             # Try removing it if it's elsewhere (less ideal)
             try:
                 sys.path.remove(temp_dir)
             except ValueError:
                 pass # Not found, maybe already removed
    except (IndexError):
        pass # sys.path was empty

    # Clean up any temporary files created
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Could not remove temp directory {temp_dir}: {e}")
        
    # Explicitly call garbage collection to clean up any lingering references
    gc.collect()
# --- End Fixture ---