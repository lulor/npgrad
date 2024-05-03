from __future__ import annotations

from itertools import zip_longest
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._array import Array, asarray_, implements

__all__ = [
    "add",
    "divide",
    "exp",
    "log",
    "log2",
    "log10",
    "matmul",
    "multiply",
    "negative",
    "power",
    "sqrt",
    "subtract",
    "tanh",
]


def _np_reduce_to(x: NDArray, shape: tuple[int, ...]) -> NDArray:
    """
    Reduce an ndarray to the specified shape performing an 'add' operation
    (can be seen as a reverse broadcasting).

    Parameters
    ----------
    x : ndarray
        The array to reduce.
    shape : tuple of ints
        The target shape.

    Returns
    -------
    out : ndarray
        The reduced array.
    """
    if x.shape != shape:
        assert x.ndim >= len(shape)
        dims = zip_longest(reversed(x.shape), reversed(shape), fillvalue=1)
        axes = []
        for axis, (x_dim, target_dim) in enumerate(dims, start=1):
            if x_dim != target_dim:
                assert target_dim == 1
                axes.append(-axis)
        x = x.sum(tuple(axes))
        x = np.reshape(x, shape)  # since we may have removed some dims in the sum
    return x


def _build_array(
    np_ufunc: Callable[..., ArrayLike],
    backward_func: Callable[..., None],
    *arrays: Array,
) -> Array:
    np_arrays = (x.data for x in arrays)
    prevs = tuple(x for x in arrays if x.requires_grad)
    backward = (lambda out: backward_func(out, *arrays)) if prevs else None
    return Array(
        np_ufunc(*np_arrays),
        requires_grad=bool(prevs),
        _prevs=prevs,
        _backward=backward,
    )


##### ufuncs #####


@implements(np.add)
def add(x1: ArrayLike, x2: ArrayLike) -> Array:
    x1, x2 = asarray_(x1), asarray_(x2)
    return _build_array(np.add, _add_backward, x1, x2)


def _add_backward(out: Array, x1: Array, x2: Array) -> None:
    assert out.grad is not None
    if x1.requires_grad:
        assert x1.grad is not None
        x1.grad += _np_reduce_to(out.grad, x1.shape)
    if x2.requires_grad:
        assert x2.grad is not None
        x2.grad += _np_reduce_to(out.grad, x2.shape)


@implements(np.subtract)
def subtract(x1: ArrayLike, x2: ArrayLike) -> Array:
    return add(x1, negative(x2))


@implements(np.multiply)
def multiply(x1: ArrayLike, x2: ArrayLike) -> Array:
    x1, x2 = asarray_(x1), asarray_(x2)
    return _build_array(np.multiply, _multiply_backward, x1, x2)


def _multiply_backward(out: Array, x1: Array, x2: Array) -> None:
    assert out.grad is not None
    if x1.requires_grad:
        assert x1.grad is not None
        x1.grad += _np_reduce_to(out.grad * x2.data, x1.shape)
    if x2.requires_grad:
        assert x2.grad is not None
        x2.grad += _np_reduce_to(out.grad * x1.data, x2.shape)


@implements(np.matmul)
def matmul(x1: ArrayLike, x2: ArrayLike) -> Array:
    x1, x2 = asarray_(x1), asarray_(x2)
    return _build_array(np.matmul, _matmul_backward, x1, x2)


def _matmul_backward(out: Array, x1: Array, x2: Array) -> None:
    assert out.grad is not None
    if x1.requires_grad:
        assert x1.grad is not None
        x1.grad += out.grad @ x2.data.T
    if x2.requires_grad:
        assert x2.grad is not None
        x2.grad += x1.data.T @ out.grad


@implements(np.divide)
def divide(x1: ArrayLike, x2: ArrayLike) -> Array:
    return multiply(x1, power(x2, -1))


@implements(np.power)
def power(x1: ArrayLike, x2: ArrayLike) -> Array:
    x1, x2 = asarray_(x1), asarray_(x2)
    return _build_array(np.power, _power_backward, x1, x2)


def _power_backward(out: Array, x1: Array, x2: Array) -> None:
    assert out.grad is not None
    if x1.requires_grad:
        assert x1.grad is not None
        x1.grad += _np_reduce_to(
            out.grad * x2.data * x1.data ** (x2.data - 1), x1.shape
        )
    if x2.requires_grad:
        raise NotImplementedError


@implements(np.negative)
def negative(x: ArrayLike) -> Array:
    return multiply(x, -1)


@implements(np.exp)
def exp(x: ArrayLike) -> Array:
    return _build_array(np.exp, _exp_backward, asarray_(x))


def _exp_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad * out.data


@implements(np.log)
def log(x: ArrayLike) -> Array:
    return _build_array(np.log, _log_backward, asarray_(x))


def _log_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad / x.data


@implements(np.log2)
def log2(x: ArrayLike) -> Array:
    return _build_array(np.log2, _log2_backward, asarray_(x))


def _log2_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad / (x.data * np.log(2))


@implements(np.log10)
def log10(x: ArrayLike) -> Array:
    return _build_array(np.log10, _log10_backward, asarray_(x))


def _log10_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad / (x.data * np.log(10))


@implements(np.sqrt)
def sqrt(x: ArrayLike) -> Array:
    return power(x, 0.5)


@implements(np.tanh)
def tanh(x: ArrayLike) -> Array:
    return _build_array(np.tanh, _tanh_backward, asarray_(x))


def _tanh_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad * (1 - out.data**2)
