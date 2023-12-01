from typing import Any, TypeVar

T = TypeVar("T")


def unwrap(x: T | None) -> T:
    if x is None:
        raise ValueError("Unexpected None")
    return x
