from __future__ import annotations

from math import prod as _prod

import numpy as np
import numpy.typing as npt

from ..array import Array
from ..array import asarray_ as _asarray_
from . import npfuncs
from .utils import pair as _pair

_EINSUM_OPTIM = "optimal"


### relu ###


def relu(input: npt.ArrayLike) -> Array:
    x = _asarray_(input)
    if x.requires_grad:
        prevs = (x,)
        backward = lambda out: _relu_backward(out, x)
    else:
        prevs = backward = None
    return Array(
        np.maximum(x.data, 0),
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _relu_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        mask = out.data > 0
        x.grad[mask] += out.grad[mask]


### conv2d ###


def conv2d(
    input: npt.ArrayLike,
    weight: npt.ArrayLike,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> Array:
    x, w = _asarray_(input), _asarray_(weight)

    ndim = 4
    if x.ndim != ndim or w.ndim != ndim:
        raise ValueError(
            f"expected {ndim} dimensions for input and weight arrays, but got {x.ndim} and {w.ndim}"
        )

    x_data = npfuncs.pad(x.data, padding)
    x_data = npfuncs.unfold(x_data, w.shape[-2:], stride, dilation)  # type: ignore

    out_data = np.einsum("nixyhw,oihw->noxy", x_data, w.data, optimize=_EINSUM_OPTIM)

    prevs = tuple(a for a in (x, w) if a.requires_grad)

    if prevs:
        backward = lambda out: _conv2d_backward(out, x, w, stride, padding, dilation)
    else:
        backward = None

    return Array(out_data, requires_grad=bool(prevs), _prevs=prevs, _backward=backward)


def _conv2d_backward(
    out: Array,
    x: Array,
    w: Array,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> None:
    assert out.grad is not None

    if x.requires_grad:
        assert x.grad is not None
        grad = np.einsum("noxy,oihw->nixyhw", out.grad, w.data, optimize=_EINSUM_OPTIM)
        npfuncs.fold_at(x.grad, grad, w.shape[-2:], stride, padding, dilation)  # type: ignore

    if w.requires_grad:
        assert w.grad is not None
        x_data = npfuncs.pad(x.data, padding)
        x_data = npfuncs.trim(x_data, w.shape[-2:], stride, dilation)  # type: ignore
        x_data = npfuncs.unfold(x_data, out.shape[-2:], stride=dilation, dilation=stride)  # type: ignore
        w.grad += np.einsum(
            "nixyhw,nohw->oixy", x_data, out.grad, optimize=_EINSUM_OPTIM
        )


### max_pool ###


def max_pool2d(
    input: npt.ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Array:
    if stride is None:
        stride = kernel_size

    x = _asarray_(input)
    x_data = npfuncs.pad(x.data, padding, np.NINF)
    x_data = npfuncs.unfold(x_data, kernel_size, stride)

    axis = (-2, -1)
    out_data = x_data.max(axis)

    if x.requires_grad:
        prevs = (x,)
        mask = x_data == np.expand_dims(out_data, axis)
        backward = lambda out: _max_pool2d_backward(
            out, x, mask, kernel_size, stride, padding
        )
    else:
        prevs = backward = None

    return Array(
        out_data,
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _max_pool2d_backward(
    out: Array,
    x: Array,
    mask: npt.NDArray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        axis = (-2, -1)
        count = np.count_nonzero(mask, axis, keepdims=True)  # type: ignore
        out_grad = np.expand_dims(out.grad, axis) / count
        out_grad = np.broadcast_to(out_grad, mask.shape)
        npfuncs.fold_at(x.grad, out_grad, kernel_size, stride, padding, indices=mask)


### avg_pool ###


def avg_pool2d(
    input: npt.ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Array:
    if stride is None:
        stride = kernel_size

    x = _asarray_(input)
    x_data = npfuncs.pad(x.data, padding)
    x_data = npfuncs.unfold(x_data, kernel_size, stride)

    out_data = x_data.mean((-2, -1))

    if x.requires_grad:
        prevs = (x,)
        backward = lambda out: _avg_pool2d_backward(
            out, x, kernel_size, stride, padding
        )
    else:
        prevs = backward = None

    return Array(
        out_data,
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _avg_pool2d_backward(
    out: Array,
    x: Array,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        out_grad = np.expand_dims(out.grad, (-2, -1)) / _prod(_pair(kernel_size))
        npfuncs.fold_at(x.grad, out_grad, kernel_size, stride, padding)


### softmax / cross_entropy ###


def softmax(input: npt.ArrayLike, axis: int) -> Array:
    x = _asarray_(input)
    if not x.ndim:
        raise ValueError("input must have at least 1 dimension")
    exp_ = np.exp(x - np.amax(x, axis, keepdims=True))
    return exp_ / np.sum(exp_, axis=axis, keepdims=True)


def cross_entropy(input: npt.ArrayLike, target: npt.ArrayLike) -> Array:
    x, target = _asarray_(input), np.asarray(target)

    if x.ndim == 1:
        x = np.expand_dims(x, 0)
        target = np.expand_dims(target, 0)
    elif x.ndim != 2:
        raise ValueError("input must have either 1 or 2 dimensions")

    if target.ndim != x.ndim - 1 or len(target) != len(x):
        raise ValueError(f"x and target shapes mismatch: {x.shape}, {target.shape}")

    # x is (n, c) and target is (n,)

    exp_ = np.exp(x - np.amax(x, axis=1, keepdims=True))
    out = exp_[np.arange(len(target)), target] / np.sum(exp_, axis=1)
    out = -np.log(out)

    return np.squeeze(out)
