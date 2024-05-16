from itertools import zip_longest
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from npgrad._array import Array, implements, in_array, out_array
from npgrad._grad import is_grad_enabled


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


def _dispatch_ufunc(
    np_ufunc: Callable[..., ArrayLike],
    backward_func: Callable[..., None],
    inputs: tuple[ArrayLike, ...],
    out: Array | tuple | None,  # numpy always wraps "out" in a tuple
) -> Array:
    arrays = tuple(in_array(x) for x in inputs)
    ndarrays = (x.data for x in arrays)
    prevs = tuple(x for x in arrays if x.requires_grad) if is_grad_enabled() else None

    if out is not None:
        if not isinstance(out, Array):
            if not (len(out) == 1 and isinstance(out[0], Array)):
                raise TypeError(f"out= must be single Array")
            out = out[0]
        if prevs or out.requires_grad:
            raise RuntimeError("out= is not supported for arrays requiring grad")
        np_ufunc(*ndarrays, out=out.data)
    else:
        out_data = np_ufunc(*ndarrays)
        backward = (lambda x: backward_func(x, *arrays)) if prevs else None
        out = out_array(out_data, prevs, backward)

    return out


##### ufuncs #####


@implements(np.add)
def add(x1: ArrayLike, x2: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.add, _add_backward, (x1, x2), out)


def _add_backward(out: Array, x1: Array, x2: Array) -> None:
    assert out.grad is not None
    if x1.requires_grad:
        assert x1.grad is not None
        x1.grad += _np_reduce_to(out.grad, x1.shape)
    if x2.requires_grad:
        assert x2.grad is not None
        x2.grad += _np_reduce_to(out.grad, x2.shape)


@implements(np.subtract)
def subtract(x1: ArrayLike, x2: ArrayLike, out: Array | None = None) -> Array:
    return add(x1, negative(x2), out)


@implements(np.multiply)
def multiply(x1: ArrayLike, x2: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.multiply, _multiply_backward, (x1, x2), out)


def _multiply_backward(out: Array, x1: Array, x2: Array) -> None:
    assert out.grad is not None
    if x1.requires_grad:
        assert x1.grad is not None
        x1.grad += _np_reduce_to(out.grad * x2.data, x1.shape)
    if x2.requires_grad:
        assert x2.grad is not None
        x2.grad += _np_reduce_to(out.grad * x1.data, x2.shape)


@implements(np.matmul)
def matmul(x1: ArrayLike, x2: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.matmul, _matmul_backward, (x1, x2), out)


def _matmul_backward(out: Array, x1: Array, x2: Array) -> None:
    assert out.grad is not None
    if x1.requires_grad:
        assert x1.grad is not None
        x1.grad += out.grad @ x2.data.T
    if x2.requires_grad:
        assert x2.grad is not None
        x2.grad += x1.data.T @ out.grad


@implements(np.divide)
def divide(x1: ArrayLike, x2: ArrayLike, out: Array | None = None) -> Array:
    return multiply(x1, power(x2, -1), out)


@implements(np.power)
def power(x1: ArrayLike, x2: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.power, _power_backward, (x1, x2), out)


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
def negative(x: ArrayLike, out: Array | None = None) -> Array:
    return multiply(x, -1, out)


@implements(np.exp)
def exp(x: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.exp, _exp_backward, (x,), out)


def _exp_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad * out.data


@implements(np.log)
def log(x: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.log, _log_backward, (x,), out)


def _log_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad / x.data


@implements(np.log2)
def log2(x: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.log2, _log2_backward, (x,), out)


def _log2_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad / (x.data * np.log(2))


@implements(np.log10)
def log10(x: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.log10, _log10_backward, (x,), out)


def _log10_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad / (x.data * np.log(10))


@implements(np.sqrt)
def sqrt(x: ArrayLike, out: Array | None = None) -> Array:
    return power(x, 0.5, out)


@implements(np.tanh)
def tanh(x: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.tanh, _tanh_backward, (x,), out)


def _tanh_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad * (1 - out.data**2)


@implements(np.abs)
@implements(np.absolute)
def absolute(x: ArrayLike, out: Array | None = None) -> Array:
    return _dispatch_ufunc(np.absolute, _absolute_backward, (x,), out)


abs = absolute


def _absolute_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += np.sign(x.data) * out.grad
