from typing import Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from npgrad._array import Array, asarray_, implements
from npgrad.typing import ShapeLike

_ShapeLike = TypeVar("_ShapeLike", bound=ShapeLike | None)


def _copy_if_list(x: _ShapeLike) -> _ShapeLike:
    return x.copy() if isinstance(x, list) else x


def _np_expand_dims(x: NDArray, axis: ShapeLike | None) -> NDArray:
    return x if axis is None else np.expand_dims(x, axis)


##### functions #####


@implements(np.reshape)
def reshape(x: ArrayLike, newshape: ShapeLike) -> Array:
    return _reshape(np.reshape, x, newshape)


@implements(np.squeeze)
def squeeze(x: ArrayLike, axis: ShapeLike | None = None) -> Array:
    return _reshape(np.squeeze, x, axis)


@implements(np.expand_dims)
def expand_dims(x: ArrayLike, axis: ShapeLike) -> Array:
    return _reshape(np.expand_dims, x, axis)


def _reshape(
    np_func: Callable[..., ArrayLike],
    x: ArrayLike,
    axis_or_shape: ShapeLike | None,
) -> Array:
    assert np_func in (np.reshape, np.squeeze, np.expand_dims)
    x = asarray_(x)
    if x.requires_grad:
        prevs = (x,)
        backward = lambda out: _reshape_backward(out, x)
    else:
        prevs = backward = None
    return Array(
        np_func(x.data, axis_or_shape),  # type: ignore
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _reshape_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += out.grad.reshape(x.shape)


#####


@implements(np.transpose)
def transpose(x: ArrayLike, axes: ShapeLike | None = None) -> Array:
    x = asarray_(x)
    if x.requires_grad:
        prevs = (x,)
        axes = _copy_if_list(axes)
        backward = lambda out: _transpose_backward(out, x, axes)
    else:
        prevs = backward = None
    return Array(
        np.transpose(x.data, axes),
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _transpose_backward(out: Array, x: Array, axes: ShapeLike | None) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        if isinstance(axes, (tuple, list)):
            axes_t = [0] * len(axes)
            for i, axis in enumerate(axes):
                axes_t[axis] = i
        else:
            axes_t = axes
        x.grad += np.transpose(out.grad, axes_t)


#####


@implements(np.swapaxes)
def swapaxes(x: ArrayLike, axis1: int, axis2: int) -> Array:
    return moveaxis(x, axis1, axis2)


@implements(np.moveaxis)
def moveaxis(x: ArrayLike, source: ShapeLike, destination: ShapeLike) -> Array:
    x = asarray_(x)
    if x.requires_grad:
        prevs = (x,)
        source, destination = _copy_if_list(source), _copy_if_list(destination)
        backward = lambda out: _moveaxis_backward(out, x, source, destination)
    else:
        prevs = backward = None
    return Array(
        np.moveaxis(x.data, source, destination),
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _moveaxis_backward(
    out: Array, x: Array, source: ShapeLike, destination: ShapeLike
) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad += np.moveaxis(out.grad, destination, source)


#####


@implements(np.sum)
def sum(x: ArrayLike, axis: ShapeLike | None = None, keepdims: bool = False) -> Array:
    return _sum_mean(np.sum, x, axis, keepdims)


@implements(np.mean)
def mean(x: ArrayLike, axis: ShapeLike | None = None, keepdims: bool = False) -> Array:
    return _sum_mean(np.mean, x, axis, keepdims)


def _sum_mean(
    np_func: Callable[..., ArrayLike],
    x: ArrayLike,
    axis: ShapeLike | None,
    keepdims: bool,
) -> Array:
    assert np_func in (np.sum, np.mean)
    x = asarray_(x)
    if x.requires_grad:
        prevs = (x,)
        axis_to_expand = None if keepdims else _copy_if_list(axis)
        is_mean = np_func is np.mean
        backward = lambda out: _sum_mean_backward(out, x, axis_to_expand, is_mean)
    else:
        prevs = backward = None
    return Array(
        np_func(x.data, axis=axis, keepdims=keepdims),
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _sum_mean_backward(
    out: Array, x: Array, axis: ShapeLike | None, is_mean: bool
) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        out_grad = _np_expand_dims(out.grad, axis)
        if is_mean:
            out_grad /= x.size / out.size
        x.grad += out_grad


#####


@implements(np.min)
@implements(np.amin)
def min(x: ArrayLike, axis: ShapeLike | None = None, keepdims: bool = False) -> Array:
    return _min_max(np.amin, x, axis, keepdims)


@implements(np.max)
@implements(np.amax)
def max(x: ArrayLike, axis: ShapeLike | None = None, keepdims: bool = False) -> Array:
    return _min_max(np.amax, x, axis, keepdims)


def _min_max(
    np_func: Callable[..., ArrayLike],
    x: ArrayLike,
    axis: ShapeLike | None,
    keepdims: bool,
) -> Array:
    assert np_func in (np.amin, np.amax)
    x = asarray_(x)
    if x.requires_grad:
        prevs = (x,)
        axis_to_expand = None if keepdims else _copy_if_list(axis)
        backward = lambda out: _min_max_backward(out, x, axis_to_expand)
    else:
        prevs = backward = None
    return Array(
        np_func(x.data, axis=axis, keepdims=keepdims),
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _min_max_backward(out: Array, x: Array, axis: ShapeLike | None) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        mask = x.data == _np_expand_dims(out.data, axis)
        count = np.count_nonzero(mask, axis, keepdims=True)  # type: ignore
        out_grad = _np_expand_dims(out.grad, axis) / count
        out_grad = np.broadcast_to(out_grad, x.shape)
        x.grad[mask] += out_grad[mask]
        # or
        # x.grad += np.where(mask, out_grad, 0)
