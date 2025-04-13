# DieCast

A Python tool to enforce type hints as runtime assertions. Shape your code or watch it die.

## Description

DieCast is a type checking utility that transforms Python's optional type hints into runtime assertions. It helps catch type errors early and makes your code more robust without changing how you write type annotations or littering your code.

DieCast embodies Assertion-Driven Development - the philosophy that if your program doesn't satisfy expectations, it should crash. Code that silently invalidates expectations is a liability. Your code will adjust to the expected (asserted) shape or die trying.

## Features

- **Type checking decorator (`@diecast`)** - Apply to functions, methods, or classes to enforce type checks.
- **Automatic Module Decoration (`mold()`)** - Apply type checking to all eligible functions/classes in a module using `mold()`.
- **Support for complex types** - Works with `typing` constructs like `List`, `Dict`, `Union`, `Optional`, `TypeVar`, etc.
- **Generic Type Support** - Handles `typing.Generic` classes and resolves `TypeVar`s based on instantiation.
- **Nested type validation** - Validates nested structures like `List[Dict[str, int]]`.
- **Special case handling** - Properly handles generators, async functions/generators, forward references, etc.
- **Exclusion Decorator (`@diecast.ignore`)** - Use `@diecast.ignore` to skip specific functions/classes during automatic decoration.
- **Clean error messages** - Reports detailed information about type errors via `YouDiedError`.

## Why Assertion-Driven Development?

DieCast treats assertions as a blueprint, a mold, a scaffold, and a gauntlet. Your type hints (and inline assertions) form a contract that your code must satisfy. This architecture-first approach works even without extensive planning - add your type assumptions and state assertions as you go, and let the sum of these expectations form the mold to which your code must conform.

## Installation

```bash
pip install py-diecast
```

Or install from source:

```bash
# Replace with the actual URL if different
git clone https://github.com/GWUDCAP/diecast.git
cd diecast
pip install -e .
```

## Usage

### Basic Decorator

Apply the `@diecast` decorator directly to functions or methods:

```python
from diecast import diecast
from typing import List

@diecast
def greet(name: str) -> str:
    return f"Hello, {name}!"

@diecast
def process_numbers(numbers: List[int]) -> int:
    return sum(numbers)

# Works fine
greet("World")
process_numbers([1, 2, 3])

# Raises YouDiedError (subclass of TypeError) - wrong argument type
greet(123)
process_numbers([1, "a", 3])

# Note: For @diecast to work reliably, especially when combined with other decorators,
# it should generally be placed as the *innermost* decorator (closest to the `def` line).
```

### Automatic Module Decoration (`mold()`)

Enable type checking for an entire module by importing and **calling** the `mold()` function at the end of the module file:

```python
# In your module (e.g., my_module.py)
from typing import List, Dict, Any
from diecast import mold

# Define your functions and classes with type hints
def process_data(items: List[int]) -> Dict[str, Any]:
    return {"processed": sum(items)}

class MyClass:
    def method(self, value: str) -> bool:
        return bool(value)

# Call mold() at the end of the module
# This applies @diecast to eligible, annotated functions/classes defined above
mold()

# ---
# Now, elsewhere in your project:
# import my_module
#
# my_module.process_data([1, 2]) # OK
# # my_module.process_data(["a"]) # Raises YouDiedError
#
# instance = my_module.MyClass()
# instance.method("hello") # OK
# # instance.method(123) # Raises YouDiedError
```

### Excluding Functions/Classes (`@diecast.ignore`)

Use the `@diecast.ignore` decorator to exclude specific functions or classes when using `mold()`:

```python
from diecast import diecast, mold
from typing import List, Dict, Any

@diecast.ignore
def function_to_skip(a: int, b: str) -> int:
    # This won't be type checked by mold()
    return a + int(b)

@diecast.ignore
class ClassToSkip:
    # This class and its methods won't be type checked by mold()
    # Note: @diecast.ignore applies to the whole class here.
    # If you want to ignore only specific methods, apply @diecast to the
    # class and @diecast.ignore to the specific methods.
    def method(self, x: int) -> str:
        return str(x * 2)

# ... other functions ...

mold() # Apply to the rest of the module
```

### Working with Generics

`@diecast` correctly handles generic classes defined using `typing.Generic` and resolves `TypeVar`s based on how the generic class is specialized.

```python
from typing import TypeVar, Generic, List
from diecast import diecast

T = TypeVar('T')

@diecast
class Box(Generic[T]):
    def __init__(self, item: T):
        self.item: T = item

    def get_item(self) -> T:
        return self.item

    def process_list(self, items: List[T]) -> List[T]:
        # This will check that elements in 'items' match the T of the Box instance
        return items[:]

# Create instances with specific types
int_box = Box[int](10)
str_box = Box[str]("hello")

# Type checks are specific to the instance's type
int_box.get_item() # OK, returns int
# int_box.process_list(["a"]) # Raises YouDiedError (expects List[int])
int_box.process_list([1, 2]) # OK

str_box.get_item() # OK, returns str
# str_box.process_list([1]) # Raises YouDiedError (expects List[str])
str_box.process_list(["a", "b"]) # OK
```

### Error Handling

When a type mismatch occurs at runtime, DieCast raises a `YouDiedError`, which is a subclass of Python's built-in `TypeError`.

`YouDiedError` provides a detailed error message indicating:
*   The function or method where the error occurred.
*   The parameter name or return value involved.
*   The expected type.
*   The actual type received.
*   The value that caused the mismatch (truncated if large).
*   The location in the user's code that called the failing function.

```python
from diecast import diecast

@diecast
def add(x: int, y: int) -> int:
    return x + y

try:
    add(5, "oops")
except Exception as e:
    print(f"Caught: {type(e).__name__}")
    print(e)
    # Output will show YouDiedError and a detailed message
    # about 'y' expecting 'int' but receiving 'str'.
```

### Logging

DieCast uses the standard Python `logging` module. It logs information under the logger name `'diecast'`.

**Important:** DieCast **does not** configure any logging handlers or set logging levels by default. If you want to see DieCast's internal logging messages (e.g., for debugging decorator application or type resolution), you must configure the `'diecast'` logger using standard Python `logging` techniques:

```python
import logging

# Example: Configure basic logging to the console
logging.basicConfig(level=logging.DEBUG) # Or use logging.INFO
diecast_logger = logging.getLogger('diecast')
# You can add specific handlers, formatters, etc. to diecast_logger if needed
# e.g., handler = logging.FileHandler('diecast.log')
# e.g., diecast_logger.addHandler(handler)

# Now run your code using @diecast or mold()
# ...
```

## License

MIT License. See the `LICENSE` file for full details.

## Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` file for guidelines.