import pytest
from typing import Any, List

from diecast import diecast

"""
Tests DieCast's handling of the `typing.Any` type.

According to PEP 484, `Any` is a special type indicating an unconstrained type.
A static type checker will treat every type as being compatible with `Any` and
`Any` as being compatible with every type.

DieCast respects this by effectively bypassing type checks when `Any` is
encountered in a type annotation (for parameters, return values, or within
nested types like List[Any]).
"""

@diecast
def func_with_any_param(x: int, y: Any) -> List:
    """Function with Any parameter annotation."""
    return [x, y]

@diecast
def func_with_any_return(x: int) -> Any:
    """Function with Any return type annotation."""
    return x

@diecast
def func_with_list_any(items: List[Any]) -> List[Any]:
    """Function with List[Any] type annotation."""
    return items

def test_any_parameter_handling():
    """Test that Any in a parameter annotation allows any type."""
    # DieCast should allow any type because y is annotated as Any
    result = func_with_any_param(1, "string")
    assert result == [1, "string"]
    
    result = func_with_any_param(1, 42)
    assert result == [1, 42]
    
    result = func_with_any_param(1, [1, 2, 3])
    assert result == [1, [1, 2, 3]]
    
    result = func_with_any_param(1, None)
    assert result == [1, None]

def test_any_return_handling():
    """Test that Any in a return annotation allows any returned type."""
    # DieCast should allow any return type because the annotation is Any
    result_int = func_with_any_return(1)
    assert result_int == 1

    @diecast
    def func_returns_str() -> Any:
        return "string"
    assert func_returns_str() == "string"

    @diecast
    def func_returns_none() -> Any:
        return None
    assert func_returns_none() is None

def test_any_in_container_handling():
    """Test that Any within a container type (List[Any]) works."""
    # DieCast should allow mixed types within the list
    items = [1, "string", True, None, 4.5, [10]]
    result = func_with_list_any(items)
    assert result == items 