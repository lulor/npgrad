__all__ = ["constant_", "uniform_", "normal_"]

from typing import TypeVar

import numpy as np

from npgrad._array import Array

_Array = TypeVar("_Array", bound=Array)


def constant_(x: _Array, val: float) -> _Array:
    x.data.fill(val)
    return x


def uniform_(x: _Array, a: float = 0.0, b: float = 1.0) -> _Array:
    x.data = np.random.default_rng().uniform(a, b, size=x.shape).astype(x.dtype)
    return x


def normal_(x: _Array, mean: float = 0.0, std: float = 1.0) -> _Array:
    x.data = np.random.default_rng().normal(mean, std, size=x.shape).astype(x.dtype)
    return x
