# DieCast

A Python tool to enforce type hints as runtime assertions. Shape your code or watch it die.

## Description

DieCast is a type checking utility that transforms Python's optional type hints into runtime assertions. It helps catch type errors early and makes your code more robust without changing how you write type annotations or littering your code.

DieCast embodies Assertion-Driven Development - the philosophy that if your program doesn't satisfy expectations, it should crash. Code that silently invalidates expectations is a liability. Your code will adjust to the expected shape or die trying.

## Features

- **Type checking decorator** - Apply to any function to enforce type checks
- **Automatic module import hook** - Enable type checking for entire modules
- **Support for complex types** - Works with List, Dict, Union, Optional, etc.
- **Nested type validation** - Validates nested structures like List[Dict[str, int]]
- **Special case handling** - Properly handles generators, async functions, forward references, etc.
- **Clean error messages** - Reports detailed information about type errors

## Why Assertion-Driven Development?

DieCast treats assertions as a blueprint, a mold, a scaffold, and a gauntlet. Your type hints form a contract that your code must satisfy. This architecture-first approach works even without extensive planning - add your type assumptions as you go, and let the sum of these expectations form the mold to which your code must conform.

## Installation

```bash
pip install diecast
```

Or install from source:

```bash
git clone https://github.com/yourusername/diecast.git
cd diecast
pip install -e .
```

## Usage

### Basic Decorator

```python
from diecast import diecast

@diecast
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Works fine
greet("World")  # Returns "Hello, World!"

# Raises TypeError - wrong argument type
greet(123)

**Note:** For `@diecast` to work reliably, especially when combined with other decorators, it should be placed as the **innermost** decorator (closest to the `def` line). This ensures it checks the actual arguments passed to your function and the value returned by it, before any other decorators modify them. Placing it further out might lead to checks against values modified by other decorators, which could cause unexpected results.
```

### Complex Type Annotations

```python
from typing import List, Dict, Optional, Union
from diecast import diecast

@diecast
def process_data(
    items: List[int],
    config: Dict[str, Union[int, str]],
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    # All parameters will be type checked
    result = {"processed": sum(items)}
    if user_id is not None:
        result["user_id"] = user_id
    return result
```

### Module Import Hook

Enable type checking for an entire module by importing the `mold` submodule directly:

```python
# At the top of your module
from diecast import mold

# All annotated functions in this module will now be type-checked
def process_data(items: List[int]) -> Dict[str, Any]:
    # This function will be automatically type-checked
    return {"processed": sum(items)}
```

### Excluding Functions

Use the `@diecast.ignore` decorator to exclude specific functions when using `mold`:

```python
from diecast import diecast, mold

@diecast.ignore
def function_to_skip(a: int, b: str) -> int:
    # This won't be type checked
    return a + int(b)
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 